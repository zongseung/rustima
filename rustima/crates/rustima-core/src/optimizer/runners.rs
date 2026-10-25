//! Optimizer runners (BFGS, L-BFGS, L-BFGS-B, Nelder-Mead) and gradient helpers.

use basin::core::math::DenseMatrix;
use basin::{
    Bfgs, CostFunction, Executor, FirstOrderState, Lbfgs, MoreThuente, NelderMead,
    OptimizationResult, SimplexProgress, State, Termination,
};

use crate::error::SarimaxError;
use crate::types::SarimaxConfig;

use super::objective::{ProfiledSarimaxObjective, SarimaxObjective};
use super::DIAG_PRECOND_SCALE;

// ---------------------------------------------------------------------------
// L-BFGS optimization
// ---------------------------------------------------------------------------

/// Outcome of a single optimizer run: `(params, cost, iterations, converged)`.
pub(super) type RunOutcome = std::result::Result<(Vec<f64>, f64, u64, bool), String>;

/// A single-run outcome tagged with the method that produced it.
pub(super) type MethodOutcome =
    std::result::Result<(Vec<f64>, f64, u64, bool, String), SarimaxError>;

fn basin_outcome<S: State<Param = Vec<f64>, Float = f64>>(
    result: OptimizationResult<S>,
) -> RunOutcome {
    if let Termination::Failed(reason) = &result.report.termination {
        return Err(format!("Basin solver failed: {reason:?}"));
    }
    if !result.best_cost().is_finite() {
        return Err("Basin solver found no finite objective value".into());
    }
    Ok((
        result.best_param().clone(),
        result.best_cost(),
        result.iter(),
        result.report.termination.is_converged(),
    ))
}

pub(super) fn run_bfgs(
    objective: SarimaxObjective,
    init_params: Vec<f64>,
    maxiter: u64,
) -> RunOutcome {
    let n = init_params.len();
    let grad0 = objective
        .analytical_gradient_negloglike(&init_params)
        .ok()
        .filter(|g| g.len() == n && g.iter().all(|v| v.is_finite()))
        .unwrap_or_else(|| vec![1.0; n]);
    // Retain diagonal preconditioning so high ARMA gradients do not send
    // unbounded regression coefficients far from their starting values.
    let initial_inverse = DenseMatrix::from_fn(n, n, |i, j| {
        if i == j {
            DIAG_PRECOND_SCALE / grad0[i].abs().max(1e-3)
        } else {
            0.0
        }
    });
    let solver = Bfgs::with_line_search(MoreThuente::new())
        .with_initial_inverse_hessian(initial_inverse)
        .with_absolute_gradient_tolerance(1e-5)
        .with_absolute_cost_change_tolerance(1e-9);
    let result = Executor::new(objective, solver, FirstOrderState::new(init_params))
        .max_iter(maxiter)
        .run()
        .map_err(|e| format!("Basin BFGS failed: {e}"))?;
    basin_outcome(result)
}

/// Central finite-difference gradient with relative step h = eps * (1 + |x_i|).
///
/// `f` must return the objective value to differentiate (already sign-adjusted
/// by the caller). Errors from `f` propagate.
fn central_diff_grad<F>(x: &[f64], eps: f64, mut f: F) -> std::result::Result<Vec<f64>, String>
where
    F: FnMut(&[f64]) -> std::result::Result<f64, String>,
{
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut xw = x.to_vec();
    for i in 0..n {
        let h = eps * (1.0 + x[i].abs());
        let orig = xw[i];
        xw[i] = orig + h;
        let fp = f(&xw)?;
        xw[i] = orig - h;
        let fm = f(&xw)?;
        xw[i] = orig;
        grad[i] = (fp - fm) / (2.0 * h);
    }
    Ok(grad)
}

/// Central finite-difference gradient of -LL in unconstrained space.
/// Used by trust-region BFGS — analytical gradient (via score()) returns
/// near-zero values for unbounded exog dimensions on this codebase, so we
/// fall back to finite-diff which empirically matches the true loss
/// landscape (verified against statsmodels at the same parameters).
pub(super) fn finite_diff_grad_negll(
    objective: &SarimaxObjective,
    x: &[f64],
    eps: f64,
) -> std::result::Result<Vec<f64>, String> {
    central_diff_grad(x, eps, |xi| objective.eval_loglike(xi).map(|ll| -ll))
}

pub(super) fn finite_diff_grad_profile_negll(
    objective: &ProfiledSarimaxObjective,
    x: &[f64],
    eps: f64,
) -> std::result::Result<Vec<f64>, String> {
    central_diff_grad(x, eps, |xi| objective.eval_negloglike(xi))
}

pub(super) fn run_lbfgs(
    objective: SarimaxObjective,
    init_params: Vec<f64>,
    maxiter: u64,
) -> RunOutcome {
    let solver = Lbfgs::new()
        .unbounded()
        .with_m_capacity(10)
        .with_absolute_gradient_tolerance(1e-5)
        .with_absolute_cost_change_tolerance(1e-9);
    let result = Executor::new(objective, solver, FirstOrderState::new(init_params))
        .max_iter(maxiter)
        .run()
        .map_err(|e| format!("Basin L-BFGS failed: {e}"))?;
    basin_outcome(result)
}

// ---------------------------------------------------------------------------
// Nelder-Mead (also used for CSS initialization and fallback refinement)
// ---------------------------------------------------------------------------

pub(super) fn run_simplex<P>(
    objective: P,
    simplex: Vec<Vec<f64>>,
    maxiter: u64,
    tolerance: f64,
) -> RunOutcome
where
    P: CostFunction<Param = Vec<f64>, Output = f64>,
    P::Error: std::fmt::Display,
{
    // Preserve the former cost-standard-deviation stopping rule rather than
    // adding Basin's independent simplex-size and cost-spread stopping tests.
    let solver = NelderMead::new()
        .with_absolute_simplex_size_tolerance(None)
        .with_relative_simplex_size_tolerance(None)
        .with_absolute_simplex_cost_tolerance(None)
        .with_relative_simplex_cost_tolerance(None)
        .with_absolute_simplex_cost_std_dev_tolerance(tolerance);
    let result = Executor::new(objective, solver, SimplexProgress::from_simplex(simplex))
        .max_iter(maxiter)
        .run()
        .map_err(|e| format!("Basin Nelder-Mead failed: {e}"))?;
    basin_outcome(result)
}

/// Initial simplex: the start plus one vertex per dimension, displaced by 5% of
/// the coordinate, or by `zero_step` when `|x_i| <= min_abs`.
pub(super) fn simplex(start: &[f64], min_abs: f64, zero_step: f64) -> Vec<Vec<f64>> {
    let mut simplex = vec![start.to_vec()];
    for i in 0..start.len() {
        let mut vertex = start.to_vec();
        vertex[i] += if vertex[i].abs() > min_abs {
            vertex[i] * 0.05
        } else {
            zero_step
        };
        simplex.push(vertex);
    }
    simplex
}

pub(super) fn run_nelder_mead(
    objective: SarimaxObjective,
    init_params: Vec<f64>,
    maxiter: u64,
) -> RunOutcome {
    run_simplex(objective, simplex(&init_params, 1e-8, 0.00025), maxiter, 1e-6)
}

// ---------------------------------------------------------------------------
// L-BFGS-B optimization (box-constrained)
// ---------------------------------------------------------------------------

/// Compute box bounds for each parameter based on config.
///
/// Layout: `[trend | exog | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]`
pub(super) fn compute_bounds(config: &SarimaxConfig) -> Vec<(Option<f64>, Option<f64>)> {
    let kt = config.trend.k_trend();
    let n_exog = config.n_exog;
    let mut bounds = Vec::new();

    // trend + exog: unbounded
    for _ in 0..(kt + n_exog) {
        bounds.push((None, None));
    }

    // AR coefficients: unbounded when enforce_stationarity (Monahan/Jones transform
    // maps any real to stationary roots). This matches statsmodels which passes no
    // bounds to scipy L-BFGS-B when enforce_stationarity=True.
    for _ in 0..config.order.p {
        if config.enforce_stationarity {
            bounds.push((None, None));
        } else {
            bounds.push((Some(-0.999), Some(0.999)));
        }
    }

    // MA coefficients: unbounded when enforce_invertibility (same reasoning)
    for _ in 0..config.order.q {
        if config.enforce_invertibility {
            bounds.push((None, None));
        } else {
            bounds.push((Some(-0.999), Some(0.999)));
        }
    }

    // Seasonal AR: unbounded when enforce_stationarity
    for _ in 0..config.order.pp {
        if config.enforce_stationarity {
            bounds.push((None, None));
        } else {
            bounds.push((Some(-0.999), Some(0.999)));
        }
    }

    // Seasonal MA: unbounded when enforce_invertibility
    for _ in 0..config.order.qq {
        if config.enforce_invertibility {
            bounds.push((None, None));
        } else {
            bounds.push((Some(-0.999), Some(0.999)));
        }
    }

    // sigma2 (unconstrained space: exp/log transform, so any real maps to positive σ²)
    // Lower bound -50.0 maps to σ² ≈ 1.9e-22, preventing extreme values
    if !config.concentrate_scale {
        bounds.push((Some(-50.0), None));
    }

    bounds
}

struct KalmanObjective<'a>(&'a SarimaxObjective);

impl super::lbfgsb::Objective for KalmanObjective<'_> {
    fn cost(&self, x: &[f64]) -> f64 {
        self.0
            .eval_negloglike(x)
            .ok()
            .filter(|c| c.is_finite())
            .unwrap_or(super::lbfgsb::INVALID_COST)
    }

    fn evaluate(&self, x: &[f64], gradient: &mut [f64]) -> f64 {
        evaluate_with_fallback(
            x,
            gradient,
            || self.0.eval_negloglike_with_gradient(x),
            |point| self.0.eval_negloglike(point),
        )
    }
}

/// Recover from unavailable analytical gradients using finite differences.
fn evaluate_with_fallback<A, C>(x: &[f64], gradient: &mut [f64], analytical: A, cost: C) -> f64
where
    A: FnOnce() -> Result<(f64, Vec<f64>), String>,
    C: Fn(&[f64]) -> Result<f64, String>,
{
    if let Ok((value, g)) = analytical() {
        if value.is_finite() && g.len() == gradient.len() && g.iter().all(|v| v.is_finite()) {
            gradient.copy_from_slice(&g);
            return value;
        }
    }
    let value = match cost(x) {
        Ok(c) if c.is_finite() => c,
        _ => {
            gradient.fill(0.0);
            return super::lbfgsb::INVALID_COST;
        }
    };
    let mut trial = x.to_vec();
    for i in 0..x.len() {
        let step = (1.0 + x[i].abs()) * 1e-7;
        trial[i] = x[i] + step;
        let plus = cost(&trial).ok().filter(|c| c.is_finite()).unwrap_or(value);
        trial[i] = x[i];
        gradient[i] = (plus - value) / step;
        if !gradient[i].is_finite() {
            gradient[i] = 0.0;
        }
    }
    value
}

pub(super) fn run_lbfgsb(
    objective: &SarimaxObjective,
    init_params: Vec<f64>,
    bounds: Vec<(Option<f64>, Option<f64>)>,
    maxiter: u64,
) -> super::lbfgsb::Outcome {
    let obj = objective.clone();
    super::lbfgsb::minimize(&KalmanObjective(&obj), init_params, bounds, maxiter)
}

pub(super) fn consume_budget(remaining: &mut u64, total_work: &mut u64, n: u64) {
    let used = n.min(*remaining);
    *total_work = total_work.saturating_add(used);
    *remaining = remaining.saturating_sub(used);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_without_a_finite_candidate_returns_an_error() {
        struct Invalid;
        impl CostFunction for Invalid {
            type Param = Vec<f64>;
            type Output = f64;
            type Error = std::convert::Infallible;
            fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
                Ok(f64::INFINITY)
            }
        }
        let result = run_simplex(Invalid, vec![vec![0.0], vec![0.1]], 1, 1e-6);
        assert!(result.is_err());
    }

    #[test]
    fn numerical_gradient_recovers_from_invalid_analytical_gradient() {
        let mut gradient = [0.0];
        let cost = evaluate_with_fallback(
            &[2.0],
            &mut gradient,
            || Ok((4.0, vec![f64::NAN])),
            |x| Ok(x[0].powi(2)),
        );
        assert_eq!(cost, 4.0);
        assert!((gradient[0] - 4.0).abs() < 1e-6);
    }
}
