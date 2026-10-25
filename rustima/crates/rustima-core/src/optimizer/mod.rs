//! SARIMAX parameter optimization with Basin solvers and custom trust-region methods.
//!
//! This module provides:
//! - Parameter space transformations (constrained ↔ unconstrained)
//! - Negative log-likelihood objectives shared by optimization backends
//! - `fit()` function: the main entry point for model fitting

use std::cell::RefCell;

use nalgebra::DMatrix;

use crate::css;
use crate::error::{Result, SarimaxError};
use crate::model::Prepared;
use crate::params::SarimaxParams;
use crate::start_params::compute_start_params;
use crate::types::{FitResult, SarimaxConfig};

#[cfg(test)]
mod basin_tests;
mod lbfgsb;
mod multistart;
mod objective;
mod runners;
#[cfg(test)]
mod tests;
mod transforms;
mod trust_region;

use lbfgsb::Method;
use multistart::{
    compute_n_restarts, fit_bfgs_single, fit_lbfgsb_adaptive_restart, fit_lbfgsb_hybrid,
    fit_lbfgsb_single, multistart,
};
use objective::{CssObjective, SarimaxObjective};
use runners::{compute_bounds, run_lbfgsb, run_nelder_mead};

pub use transforms::{transform_params, untransform_params};
use trust_region::{fit_profile_trust_region, fit_trust_region_single};

pub fn validate_method(method: &str) -> Result<()> {
    Method::parse(method).map(|_| ())
}

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Diagonal preconditioning scale `c` for BFGS / trust-region: the initial
/// inverse Hessian is `c · diag(1 / |∇f_i|)`. Smaller values make the first
/// step more conservative (good for ill-conditioned problems with mixed-scale
/// gradients across dimensions); larger values make the first step bolder.
/// 0.1 chosen by experiment so that the first-step displacement in each
/// dimension is ≈ 0.1 in unconstrained space — small enough to keep exog
/// coefficients near their CSS seed, large enough for ARMA dims to begin
/// curvature accumulation.
const DIAG_PRECOND_SCALE: f64 = 0.1;

// ---------------------------------------------------------------------------
// Fast paths (pure AR, maxiter=0)
// ---------------------------------------------------------------------------

/// Try pure AR fast path: for high-order non-seasonal pure AR models with
/// concentrated scale, Burg AR coefficients are asymptotically MLE-equivalent.
fn try_ar_fast_path(
    data: &Prepared,
    constrained_start: &[f64],
    method: Method,
    start_params_provided: bool,
) -> Result<Option<FitResult>> {
    let config = data.config();
    let is_pure_ar_fast = config.order.p >= 3
        && config.order.q == 0
        && config.order.qq == 0
        && config.order.pp == 0
        && config.trend.k_trend() == 0
        && config.n_exog == 0
        && config.concentrate_scale
        && !start_params_provided;

    if !is_pure_ar_fast || method != Method::Lbfgsb {
        return Ok(None);
    }

    let test_output = data.loglike(&SarimaxParams::from_flat(constrained_start, config)?)?;

    if test_output.loglike.is_finite() {
        let n_params = SarimaxParams::n_estimated_params(config);
        Ok(Some(
            FitResult {
                params: constrained_start.to_vec(),
                loglike: test_output.loglike,
                scale: test_output.scale,
                n_obs: data.endog().len(),
                n_params,
                n_iter: 0,
                converged: true,
                method: "burg-direct".to_string(),
                aic: 0.0,
                bic: 0.0,
                warnings: vec![],
            }
            .with_information_criteria(),
        ))
    } else {
        Ok(None)
    }
}

/// Build result for maxiter=0 or zero-parameter models: no optimization needed.
///
/// `converged` is `true` when `n_params == 0` (trivially converged — nothing to
/// optimize), `false` when the caller explicitly requested `maxiter=0` but there
/// are free parameters remaining.
fn build_zero_iter_result(
    data: &Prepared,
    constrained_start: &[f64],
    method: &str,
) -> Result<FitResult> {
    let config = data.config();
    let output = data.loglike(&SarimaxParams::from_flat(constrained_start, config)?)?;
    let n_params = SarimaxParams::n_estimated_params(config);
    let k_free = config.n_params();
    let converged = k_free == 0;
    Ok(FitResult {
        params: constrained_start.to_vec(),
        loglike: output.loglike,
        scale: output.scale,
        n_obs: data.endog().len(),
        n_params,
        n_iter: 0,
        converged,
        method: method.to_string(),
        aic: 0.0,
        bic: 0.0,
        warnings: vec![],
    }
    .with_information_criteria())
}

/// Build final FitResult from optimized unconstrained parameters.
fn build_fit_result(
    data: &Prepared,
    best_unconstrained: &[f64],
    n_iter: u64,
    converged: bool,
    used_method: String,
) -> Result<FitResult> {
    let config = data.config();
    let final_constrained = transform_params(best_unconstrained, config)?;
    let final_params = SarimaxParams::from_flat(&final_constrained, config)?;

    // A-2: Collect near-cancellation warning into result (VER5.2 P6, V8.5 P2-1)
    let mut warnings = Vec::new();
    if !validate_no_near_cancellation(&final_params, config, 0.05) {
        warnings.push(format!(
            "near-cancellation detected in fitted ARMA({},{}) \
             parameters (min inverted-root distance < 0.05). \
             Model may be non-identifiable.",
            config.order.p, config.order.q
        ));
    }

    let output = data.loglike(&final_params)?;
    let n_params = SarimaxParams::n_estimated_params(config);

    Ok(FitResult {
        params: final_constrained,
        loglike: output.loglike,
        scale: output.scale,
        n_obs: data.endog().len(),
        n_params,
        n_iter,
        converged,
        method: used_method,
        aic: 0.0,
        bic: 0.0,
        warnings,
    }
    .with_information_criteria())
}

// ---------------------------------------------------------------------------
// Validation + start param helpers
// ---------------------------------------------------------------------------

/// Validate observations length and resolve starting parameters.
///
/// Returns constrained start parameters, either from the caller or computed
/// via Hannan-Rissanen / Burg initialization.
fn validate_and_get_start_params(
    endog: &[f64],
    config: &SarimaxConfig,
    start_params: Option<&[f64]>,
    exog: Option<&[Vec<f64>]>,
) -> Result<Vec<f64>> {
    let min_obs = config.n_params().max(config.order.k_states().saturating_add(1));
    if endog.len() <= min_obs {
        return Err(SarimaxError::DataError(format!(
            "Not enough observations: n={} <= minimum required {} for model order",
            endog.len(),
            min_obs
        )));
    }

    match start_params {
        Some(sp) => {
            let expected_len = config.n_params();
            if sp.len() != expected_len {
                return Err(SarimaxError::ParamLengthMismatch {
                    expected: expected_len,
                    got: sp.len(),
                });
            }
            Ok(sp.to_vec())
        }
        None => compute_start_params(endog, config, exog),
    }
}

// ---------------------------------------------------------------------------
// CSS pre-optimization helpers
// ---------------------------------------------------------------------------

/// Evaluate KF log-likelihood at constrained parameters. Returns NEG_INFINITY on error.
fn eval_kf_loglike_constrained(data: &Prepared, constrained: &[f64]) -> f64 {
    match SarimaxParams::from_flat(constrained, data.config()).and_then(|p| data.loglike(&p)) {
        Ok(out) if out.loglike.is_finite() => out.loglike,
        _ => f64::NEG_INFINITY,
    }
}

/// For high-dimensional state-space models (k_states >= 40), the standard
/// optimization landscape is treacherous — many local minima.  Pre-fitting
/// with `simple_differencing=true` reduces the state dimension (removes
/// diff states), giving a smoother landscape and better convergence.
/// The resulting params are then used as warm-start for the full model.
fn run_sd_warm_start(
    endog: &[f64],
    config: &SarimaxConfig,
    exog: Option<&[Vec<f64>]>,
) -> Option<Vec<f64>> {
    let mut sd_config = config.clone();
    sd_config.simple_differencing = true;

    let sd_data = Prepared::new(endog, exog, &sd_config).ok()?;

    // Get CSS-based start params for SD model
    let mut sd_start = validate_and_get_start_params(endog, &sd_config, None, exog).ok()?;

    // CSS pre-optimization on SD path
    let n_arma = sd_config.order.p + sd_config.order.q + sd_config.order.pp + sd_config.order.qq;
    if n_arma >= 1 {
        if let Some(css_p) = run_css_optimization(endog, &sd_config, &sd_start, 200, exog) {
            let css_ll = eval_kf_loglike_constrained(&sd_data, &css_p);
            let orig_ll = eval_kf_loglike_constrained(&sd_data, &sd_start);
            if css_ll > orig_ll {
                sd_start = css_p;
            }
        }
    }

    // Quick L-BFGS-B on SD model (200 iterations — enough for warm-start)
    let sd_unc = untransform_params(&sd_start, &sd_config).ok()?;
    let sd_obj = SarimaxObjective {
        data: sd_data,
        cache: RefCell::new(None),
        ss_cache: RefCell::new(None),
    };

    let (best_unc, _, _, _, _) = fit_lbfgsb_single(&sd_obj, sd_unc, &sd_config, 200).ok()?;
    transform_params(&best_unc, &sd_config).ok()
}

// ---------------------------------------------------------------------------
// A-2: Near-cancellation detection (VER5.2 P6)
// ---------------------------------------------------------------------------

/// Compute inverted roots of an AR or MA polynomial via companion matrix eigenvalues.
///
/// `coeffs` = [φ₁, φ₂, …, φ_p] (AR) or [θ₁, θ₂, …, θ_q] (MA).
/// Returns eigenvalues of the companion matrix as (real, imag) pairs.
/// These are the inverted polynomial roots (= 1 / z_i).
///
/// Uses nalgebra real Schur decomposition to handle complex conjugate pairs.
/// Matching arima2 convention: `inv_ar_roots = 1/polyroot(c(1, -ar_pars))`.
fn polynomial_roots(coeffs: &[f64]) -> Vec<(f64, f64)> {
    let p = coeffs.len();
    if p == 0 {
        return vec![];
    }
    if p == 1 {
        return vec![(coeffs[0], 0.0)];
    }

    // Companion matrix: first row = coefficients, sub-diagonal = 1
    let mut companion = DMatrix::zeros(p, p);
    for i in 0..p {
        companion[(0, i)] = coeffs[i];
    }
    for i in 1..p {
        companion[(i, i - 1)] = 1.0;
    }

    // Real Schur decomposition → quasi-upper triangular T.
    //
    // MUST be bounded: `Schur::new` iterates the QR algorithm without an
    // iteration cap and can spin forever on ill-conditioned companion
    // matrices (observed: the degree-10 companion of a (10,0,10) fit hung
    // indefinitely while holding the GIL). The roots feed only the
    // near-cancellation warning/filter, so on non-convergence we degrade
    // gracefully by returning no roots (min_root_distance → INFINITY →
    // "no near-cancellation" → the warning is simply skipped).
    let max_niter = 100 * p.max(10);
    let Some(schur) = nalgebra::Schur::try_new(companion, 1e-12, max_niter) else {
        return vec![];
    };
    let (_, t) = schur.unpack();

    // Extract eigenvalues from diagonal blocks of T
    let mut eigenvalues = Vec::with_capacity(p);
    let mut k = 0_usize;
    while k < p {
        if k + 1 < p && t[(k + 1, k)].abs() > 1e-12 {
            // 2×2 block → complex conjugate pair
            let a = t[(k, k)];
            let b = t[(k, k + 1)];
            let c = t[(k + 1, k)];
            let d = t[(k + 1, k + 1)];
            let center = (a + d) / 2.0;
            let disc = (a - d) * (a - d) / 4.0 + b * c;
            // disc < 0 for genuine complex eigenvalues; clamp to 0 for safety
            let imag = (-disc).max(0.0).sqrt();
            eigenvalues.push((center, imag));
            eigenvalues.push((center, -imag));
            k += 2;
        } else {
            eigenvalues.push((t[(k, k)], 0.0));
            k += 1;
        }
    }
    eigenvalues
}

/// Minimum distance between AR and MA inverted roots.
///
/// Near-cancellation: AR and MA polynomials share a common root → the ARMA
/// representation is non-identifiable (Fisher information matrix singular).
///
/// Threshold guidance (arima2 convention):
///   α = 0.01 — filter random restart starting points
///   α = 0.05 — warn about final parameter estimates
fn min_root_distance(ar_coeffs: &[f64], ma_coeffs: &[f64]) -> f64 {
    let ar_roots = polynomial_roots(ar_coeffs);
    // AR is 1 - φ₁z - … so the companion takes +φ, but MA is 1 + θ₁z + …
    // (arima2: `1/polyroot(c(1, ma_pars))`), so its companion needs -θ.
    // Passing +θ compares against the roots of the sign-flipped polynomial
    // and misses genuine cancellation entirely (DIAGNOSIS_V9 follow-up N3).
    let neg_ma: Vec<f64> = ma_coeffs.iter().map(|v| -v).collect();
    let ma_roots = polynomial_roots(&neg_ma);

    let mut min_dist = f64::INFINITY;
    for &(ar_re, ar_im) in &ar_roots {
        for &(ma_re, ma_im) in &ma_roots {
            let dist = ((ar_re - ma_re).powi(2) + (ar_im - ma_im).powi(2)).sqrt();
            min_dist = min_dist.min(dist);
        }
    }
    min_dist
}

/// Returns true when ARMA parameters have no near-cancellation (dist > threshold).
///
/// AR-only or MA-only models always return true (no cancellation possible).
/// Spec (VER5.2 P6): threshold=0.01 for restart filtering, 0.05 for warnings.
fn validate_no_near_cancellation(
    sparams: &SarimaxParams,
    config: &SarimaxConfig,
    threshold: f64,
) -> bool {
    if config.order.p == 0 || config.order.q == 0 {
        return true;
    }
    min_root_distance(&sparams.ar_coeffs, &sparams.ma_coeffs) > threshold
}

/// Check whether unconstrained params exhibit near-cancellation.
///
/// Returns true if the params pass (no near-cancellation), false if they
/// should be rejected. Used to filter random restart starting points (α=0.01).
fn passes_cancellation_filter(unconstrained: &[f64], config: &SarimaxConfig) -> bool {
    // Skip check for AR-only or MA-only models
    if config.order.p == 0 || config.order.q == 0 {
        return true;
    }
    let constrained = match transform_params(unconstrained, config) {
        Ok(c) => c,
        Err(_) => return false, // invalid params → reject
    };
    let sparams = match SarimaxParams::from_flat(&constrained, config) {
        Ok(p) => p,
        Err(_) => return false,
    };
    validate_no_near_cancellation(&sparams, config, 0.01)
}

/// Run CSS pre-optimization: L-BFGS-B first, NM fallback (R-style CSS-ML).
///
/// CSS is O(n·(p'+q')) per evaluation, ~100x faster than KF for large k.
/// L-BFGS-B with finite-difference gradient on CSS matches R's optim(BFGS)
/// approach for the CSS stage.
/// Returns updated constrained parameters if optimization succeeds.
fn run_css_optimization(
    endog: &[f64],
    config: &SarimaxConfig,
    constrained_start: &[f64],
    maxiter: u64,
    exog: Option<&[Vec<f64>]>,
) -> Option<Vec<f64>> {
    let unconstrained_start = untransform_params(constrained_start, config).ok()?;
    let n = unconstrained_start.len();
    if n == 0 {
        return None;
    }

    // 1st: L-BFGS-B on CSS (R-style gradient-based optimization)
    if let Some(result) = run_lbfgsb_css(endog, config, &unconstrained_start, maxiter, exog) {
        if let Ok(constrained) = transform_params(&result, config) {
            return Some(constrained);
        }
    }

    // 2nd: NM fallback (gradient-free, more robust for ill-conditioned cases)
    run_nm_css(endog, config, &unconstrained_start, maxiter, exog)
}

/// L-BFGS-B optimization on CSS objective with finite-difference gradient.
struct CssLbfgsbObjective<'a> {
    endog: &'a [f64],
    config: &'a SarimaxConfig,
    exog: Option<&'a [Vec<f64>]>,
}

impl lbfgsb::Objective for CssLbfgsbObjective<'_> {
    fn cost(&self, x: &[f64]) -> f64 {
        transform_params(x, self.config)
            .ok()
            .and_then(|p| SarimaxParams::from_flat(&p, self.config).ok())
            .map(|p| -css::css_loglike_with_exog(self.endog, self.config, &p, self.exog))
            .filter(|c| c.is_finite())
            .unwrap_or(lbfgsb::INVALID_COST)
    }

    fn evaluate(&self, x: &[f64], gradient: &mut [f64]) -> f64 {
        let value = self.cost(x);
        if value >= f64::MAX / 4.0 {
            gradient.fill(0.0);
            return lbfgsb::INVALID_COST;
        }
        let mut trial = x.to_vec();
        for i in 0..x.len() {
            trial[i] = x[i] + 1e-7;
            let plus = self.cost(&trial);
            trial[i] = x[i] - 1e-7;
            let minus = self.cost(&trial);
            trial[i] = x[i];
            gradient[i] = (plus - minus) / 2e-7;
            if !gradient[i].is_finite() {
                gradient[i] = 0.0;
            }
        }
        value
    }
}

fn run_lbfgsb_css(
    endog: &[f64],
    config: &SarimaxConfig,
    unconstrained_start: &[f64],
    maxiter: u64,
    exog: Option<&[Vec<f64>]>,
) -> Option<Vec<f64>> {
    let objective = CssLbfgsbObjective {
        endog,
        config,
        exog,
    };
    let bounds = vec![(Some(-10.0), Some(10.0)); unconstrained_start.len()];
    lbfgsb::minimize(&objective, unconstrained_start.to_vec(), bounds, maxiter)
        .ok()
        .map(|(params, _, _, _)| params)
}

/// NM fallback for CSS optimization.
fn run_nm_css(
    endog: &[f64],
    config: &SarimaxConfig,
    unconstrained_start: &[f64],
    maxiter: u64,
    exog: Option<&[Vec<f64>]>,
) -> Option<Vec<f64>> {
    let obj = CssObjective {
        endog: endog.to_vec(),
        config: config.clone(),
        exog: exog.map(|e| e.to_vec()),
    };

    let simplex = runners::simplex(unconstrained_start, 0.1, 0.025);
    let (best_unconstrained, _, _, _) = runners::run_simplex(obj, simplex, maxiter, 1e-4).ok()?;
    transform_params(&best_unconstrained, config).ok()
}

// ---------------------------------------------------------------------------
// Public fit() entry point
// ---------------------------------------------------------------------------

/// Fit a SARIMAX model using maximum likelihood estimation.
///
/// # Arguments
/// * `endog` — Observed time series
/// * `config` — Model configuration (order, stationarity enforcement, etc.)
/// * `start_params` — Optional initial parameter values (constrained space)
/// * `method` — "lbfgsb" (default), "lbfgsb-multi", "lbfgsb-adaptive",
///   "lbfgsb-hybrid", "lbfgsb-strict", "lbfgs", "bfgs", "trust-region",
///   "profile-trust-region", or "nelder-mead" ("nm"). The `lbfgsb*` names
///   also accept a "basin-" prefix for source compatibility.
/// * `maxiter` — Maximum optimizer work units (default: 500)
/// * `exog` — Optional exogenous variables, column-major: exog[j][t]
pub fn fit(
    endog: &[f64],
    config: &SarimaxConfig,
    start_params: Option<&[f64]>,
    method: Option<&str>,
    maxiter: Option<u64>,
    exog: Option<&[Vec<f64>]>,
) -> Result<FitResult> {
    let maxiter = maxiter.unwrap_or(500);
    let method = Method::parse(method.unwrap_or("lbfgsb"))?;
    let method_label = method.label();

    // (b) start_params length: must match expected free param count if provided.
    if let Some(sp) = start_params {
        let expected = config.n_params();
        if sp.len() != expected {
            return Err(SarimaxError::InvalidInput(format!(
                "start_params length mismatch: expected {}, got {}",
                expected,
                sp.len()
            )));
        }
    }

    // simple_differencing: the Kalman filter sees the differenced series;
    // n_obs = data.endog().len() for AIC/BIC.
    let data = Prepared::new(endog, exog, config)?;

    // 0. Zero-parameter fast path: models like (0,d,0) with concentrate_scale
    //    have no free parameters to optimize. Return closed-form result directly
    //    to avoid entering L-BFGS-B FFI with n=0.
    if config.n_params() == 0 {
        return build_zero_iter_result(&data, &[], method_label);
    }

    // 1. Validate + compute start params (always use original endog: CSS init
    //    applies differencing internally via apply_differencing)
    let mut constrained_start = validate_and_get_start_params(endog, config, start_params, exog)?;

    // 2. Fast paths (AR-only, maxiter=0)
    if let Some(r) = try_ar_fast_path(&data, &constrained_start, method, start_params.is_some())? {
        return Ok(r);
    }
    if maxiter == 0 {
        return build_zero_iter_result(&data, &constrained_start, method_label);
    }

    // 2.5. CSS-ML 2-stage pre-optimization (R-style):
    // R's stats::arima(method="CSS-ML") runs CSS → ML for ALL ARMA models.
    // CSS is O(n·(p'+q')) per eval vs O(n·k³) for KF, so even with L-BFGS-B
    // gradient (2n+1 evals), CSS is much cheaper than a single KF evaluation.
    //
    // Strategy:
    //   - Seasonal models: REPLACE when CSS gives any KF improvement (R-matching)
    //   - Non-seasonal models: REPLACE only when improvement > 2.0 loglike units
    //     (avoids near-cancellation basin-trapping for borderline ARIMA cases)
    //
    // Trust-region method intentionally skips CSS pre-opt: its whole point is
    // to keep the optimizer close to a carefully-computed seed (CSS 2-stage
    // start_params) by capping per-step radius. CSS pre-opt would discard that
    // seed and start the trust-region search from a basin determined by CSS,
    // defeating the purpose of preventing first-iteration drift.
    let skip_css_preopt = method == Method::TrustRegion;
    if start_params.is_none() && !skip_css_preopt {
        let n_arma = config.order.p + config.order.q + config.order.pp + config.order.qq;
        let is_seasonal = config.order.s >= 2 && (config.order.pp > 0 || config.order.qq > 0);
        let benefit_from_css = n_arma >= 1 && (is_seasonal || n_arma >= 2);
        if benefit_from_css {
            // CSS optimization uses original endog (CSS applies differencing internally)
            if let Some(css_params) =
                run_css_optimization(endog, config, &constrained_start, 300, exog)
            {
                let css_kf_ll = eval_kf_loglike_constrained(&data, &css_params);
                let orig_kf_ll = eval_kf_loglike_constrained(&data, &constrained_start);
                let threshold = if is_seasonal { 0.0 } else { 2.0 };
                if css_kf_ll > orig_kf_ll + threshold {
                    constrained_start = css_params;
                }
            }
        }
    }

    // 2.7. High-dimensional warm-start via simple_differencing.
    // For k_states >= 40 (high-order seasonal models), the full state-space
    // optimization landscape has many local minima.  Pre-fitting with
    // simple_differencing=true gives a much smoother landscape, yielding
    // params close to the global optimum that then serve as warm-start.
    if start_params.is_none() && !config.simple_differencing && !skip_css_preopt {
        let k_states = config.order.k_states();
        if k_states >= 40 {
            if let Some(sd_params) = run_sd_warm_start(endog, config, exog) {
                let sd_ll = eval_kf_loglike_constrained(&data, &sd_params);
                let orig_ll = eval_kf_loglike_constrained(&data, &constrained_start);
                if sd_ll > orig_ll + 1.0 {
                    constrained_start = sd_params;
                }
            }
        }
    }

    // 3. Transform to unconstrained space + build objective
    let unconstrained_start = untransform_params(&constrained_start, config)?;
    let objective = SarimaxObjective {
        data,
        cache: RefCell::new(None),
        ss_cache: RefCell::new(None),
    };
    let n_restarts = compute_n_restarts(unconstrained_start.len(), config);

    // 4. Dispatch on the parsed method
    let (best_unconstrained, _best_cost, n_iter, converged, used_method) = match method {
        Method::NelderMead => {
            let (p, c, n, conv) = run_nelder_mead(objective.clone(), unconstrained_start, maxiter)
                .map_err(SarimaxError::OptimizationFailed)?;
            (p, c, n, conv, "nelder-mead".to_string())
        }
        Method::LbfgsbStrict => {
            let bounds = compute_bounds(config);
            let (p, c, n, conv) = run_lbfgsb(&objective, unconstrained_start, bounds, maxiter)?;
            (p, c, n, conv, "lbfgsb-strict".to_string())
        }
        Method::Lbfgsb => fit_lbfgsb_single(&objective, unconstrained_start, config, maxiter)?,
        Method::LbfgsbMulti | Method::Lbfgs => multistart(
            method,
            &objective,
            &unconstrained_start,
            config,
            maxiter,
            n_restarts,
        )?,
        Method::LbfgsbAdaptive => fit_lbfgsb_adaptive_restart(
            &objective,
            unconstrained_start,
            config,
            maxiter,
            n_restarts,
        )?,
        Method::LbfgsbHybrid => fit_lbfgsb_hybrid(
            &objective,
            &unconstrained_start,
            config,
            maxiter,
            n_restarts,
        )?,
        Method::Bfgs => fit_bfgs_single(&objective, unconstrained_start, maxiter)?,
        Method::TrustRegion => {
            fit_trust_region_single(&objective, unconstrained_start, config, maxiter)?
        }
        Method::ProfileTrustRegion => fit_profile_trust_region(
            &objective,
            unconstrained_start,
            config,
            maxiter,
            start_params.is_some(),
        )?,
    };
    let used_method = if used_method.starts_with("lbfgsb") {
        format!("basin-{used_method}")
    } else {
        used_method
    };

    // 5. Build result (n_obs = differenced length for correct AIC/BIC)
    build_fit_result(&objective.data, &best_unconstrained, n_iter, converged, used_method)
}
