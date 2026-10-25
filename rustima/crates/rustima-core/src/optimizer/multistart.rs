//! Multi-start orchestration and per-method fit strategies.

use rayon::prelude::*;

use std::cell::RefCell;

use crate::types::SarimaxConfig;

use super::lbfgsb::{Method, MethodOutcome, Outcome, RunError};
use super::objective::SarimaxObjective;
use super::passes_cancellation_filter;
use super::runners::{
    compute_bounds, consume_budget, run_bfgs, run_lbfgs, run_lbfgsb, run_nelder_mead,
};
use super::transforms::untransform_params;

/// Incumbent: `(params, cost, converged, label)`.
type Best = Option<(Vec<f64>, f64, bool, String)>;
type Bounds = [(Option<f64>, Option<f64>)];

// ---------------------------------------------------------------------------
// Multi-start helpers (shared by lbfgsb-multi and lbfgs methods)
// ---------------------------------------------------------------------------

/// Seasonal MA grid initialization via Nelder-Mead.
///
/// For models with enforced invertibility and seasonal MA terms, tries
/// NM optimization from a grid of constrained MA/SMA starting points.
/// Gradient-free NM avoids boundary traps near invertibility constraints.
///
/// Returns updated `(best, remaining, total_work)` via mutable references.
fn grid_ma_initialization(
    objective: &SarimaxObjective,
    config: &SarimaxConfig,
    n_params_total: usize,
    remaining: &mut u64,
    total_work: &mut u64,
    best: &mut Best,
    method_label: &str,
) {
    if !config.enforce_invertibility || config.order.qq == 0 || *remaining == 0 {
        return;
    }

    let kt = config.trend.k_trend();
    let n_exog = config.n_exog;
    let ma_start = kt + n_exog + config.order.p;
    let sma_start = ma_start + config.order.q + config.order.pp;

    let grid_vals = [-0.3, -0.6, -0.9];
    for &ma_val in &grid_vals {
        if *remaining == 0 {
            break;
        }
        let mut grid_constrained = vec![0.0; n_params_total];
        for i in 0..config.order.q {
            grid_constrained[ma_start + i] = ma_val;
        }
        for i in 0..config.order.qq {
            grid_constrained[sma_start + i] = ma_val;
        }
        if let Ok(grid_uncons) = untransform_params(&grid_constrained, config) {
            if let Ok((p, c, n, conv)) = run_nelder_mead(objective.clone(), grid_uncons, *remaining)
            {
                consume_budget(remaining, total_work, n);
                try_update_best(best, p, c, conv, method_label);
            }
        }
    }
}

/// Generate LCG-perturbed starting points and optimize each.
///
/// Uses a deterministic linear congruential generator (LCG) seeded at 12345
/// for reproducible multi-start perturbations of the unconstrained start.
fn lcg_perturbed_starts(
    unconstrained_start: &[f64],
    n_restarts: usize,
    remaining: &u64,
) -> Vec<Vec<f64>> {
    let mut starts = Vec::with_capacity(n_restarts);
    let mut rng_state: u64 = 12345;

    for _ in 0..n_restarts {
        if *remaining == 0 {
            break;
        }
        let mut perturbed = unconstrained_start.to_vec();
        for v in perturbed.iter_mut() {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            let scale = if v.abs() > 0.1 { v.abs() * 0.5 } else { 0.1 };
            *v += u * scale;
        }
        starts.push(perturbed);
    }
    starts
}

/// Update `best` if `(p, c)` is better (lower cost) than the current best.
fn try_update_best(best: &mut Best, p: Vec<f64>, c: f64, conv: bool, method_name: &str) {
    match best {
        Some((_, best_cost, _, _)) if c < *best_cost => {
            *best = Some((p, c, conv, method_name.to_string()));
        }
        None => {
            *best = Some((p, c, conv, method_name.to_string()));
        }
        _ => {}
    }
}

/// Book a run's work (successful or not) and offer its result to `best`.
fn consider(outcome: Outcome, label: &str, best: &mut Best, remaining: &mut u64, total_work: &mut u64) {
    match outcome {
        Ok((p, c, n, conv)) => {
            consume_budget(remaining, total_work, n);
            try_update_best(best, p, c, conv, label);
        }
        Err(error) => consume_budget(remaining, total_work, error.work),
    }
}

/// Run L-BFGS-B from every start in parallel. Each rayon worker gets its own
/// objective (fresh caches): `SarimaxObjective` holds `RefCell`s and is `!Sync`.
fn par_lbfgsb(
    objective: &SarimaxObjective,
    starts: Vec<Vec<f64>>,
    bounds: &Bounds,
    budget: u64,
) -> Vec<Outcome> {
    let data = &objective.data;
    starts
        .into_par_iter()
        .map(|start| {
            let obj = SarimaxObjective {
                data: data.clone(),
                cache: RefCell::new(None),
                ss_cache: RefCell::new(None),
            };
            run_lbfgsb(&obj, start, bounds.to_vec(), budget)
        })
        .collect()
}

/// NM refinement: try polishing the best solution with Nelder-Mead.
///
/// Returns the final `(params, cost, total_work, converged, method_name)` tuple.
fn nm_refinement(
    objective: &SarimaxObjective,
    best_p: Vec<f64>,
    best_c: f64,
    best_conv: bool,
    method_name: String,
    n_params_total: usize,
    remaining: &mut u64,
    total_work: &mut u64,
) -> (Vec<f64>, f64, u64, bool, String) {
    if n_params_total >= 2 && *remaining > 0 {
        match run_nelder_mead(objective.clone(), best_p.clone(), *remaining) {
            Ok((nm_p, nm_c, nm_n, nm_conv)) if nm_c < best_c => {
                consume_budget(remaining, total_work, nm_n);
                (
                    nm_p,
                    nm_c,
                    *total_work,
                    nm_conv,
                    format!("{}+nm", method_name),
                )
            }
            Ok((_, _, nm_n, _)) => {
                consume_budget(remaining, total_work, nm_n);
                (best_p, best_c, *total_work, best_conv, method_name)
            }
            Err(_) => (best_p, best_c, *total_work, best_conv, method_name),
        }
    } else {
        (best_p, best_c, *total_work, best_conv, method_name)
    }
}

// ---------------------------------------------------------------------------
// Method-specific dispatch functions
// ---------------------------------------------------------------------------

/// Determine number of multi-start restarts based on model complexity.
pub(super) fn compute_n_restarts(n_params_total: usize, config: &SarimaxConfig) -> usize {
    let has_seasonal = config.order.pp > 0 || config.order.qq > 0;
    if n_params_total >= 4 {
        3
    } else if n_params_total >= 3 || has_seasonal {
        2
    } else if n_params_total >= 2 {
        1
    } else {
        0
    }
}

/// Multi-start: seed run, zero start, seasonal-MA grid (NM), LCG perturbations
/// of the seed, then NM refinement of the incumbent; Nelder-Mead from the seed
/// if nothing converged.
///
/// `method` is `Lbfgs` (unbounded Basin L-BFGS, perturbations run sequentially)
/// or `LbfgsbMulti` (bounded L-BFGS-B, perturbations run in parallel via rayon).
pub(super) fn multistart(
    method: Method,
    objective: &SarimaxObjective,
    unconstrained_start: &[f64],
    config: &SarimaxConfig,
    maxiter: u64,
    n_restarts: usize,
) -> MethodOutcome {
    let bounds = compute_bounds(config);
    let label = if method == Method::Lbfgs { "lbfgs" } else { "lbfgsb-multi" };
    let run = |start: Vec<f64>, budget: u64| -> Outcome {
        if method == Method::Lbfgs {
            run_lbfgs(objective.clone(), start, budget).map_err(RunError::from)
        } else {
            run_lbfgsb(objective, start, bounds.clone(), budget)
        }
    };
    let n_params_total = unconstrained_start.len();
    let mut remaining = maxiter;
    let mut total_work: u64 = 0;
    let mut best: Best = None;

    consider(
        run(unconstrained_start.to_vec(), remaining),
        label,
        &mut best,
        &mut remaining,
        &mut total_work,
    );

    if n_restarts > 0 && remaining > 0 {
        // 1. Zero-start
        consider(
            run(vec![0.0; n_params_total], remaining),
            label,
            &mut best,
            &mut remaining,
            &mut total_work,
        );

        // 2. Seasonal MA grid (NM, gradient-free for boundary avoidance)
        let grid_label = if method == Method::Lbfgs {
            label.to_string()
        } else {
            format!("{label}+nm")
        };
        grid_ma_initialization(
            objective,
            config,
            n_params_total,
            &mut remaining,
            &mut total_work,
            &mut best,
            &grid_label,
        );

        // 3. LCG perturbations (with P6 near-cancellation filter, α=0.01)
        let perturbations: Vec<_> =
            lcg_perturbed_starts(unconstrained_start, n_restarts, &remaining)
                .into_iter()
                .filter(|start| passes_cancellation_filter(start, config))
                .collect();
        // Skip the rayon block in a fork()ed child — the inherited pool would deadlock (C1/C2).
        let parallel = method == Method::LbfgsbMulti
            && perturbations.len() >= 2
            && remaining > 0
            && crate::batch::guard_rayon_fork().is_ok();
        if parallel {
            let per_start_budget = remaining / perturbations.len() as u64;
            if per_start_budget > 0 {
                for outcome in par_lbfgsb(objective, perturbations, &bounds, per_start_budget) {
                    consider(outcome, label, &mut best, &mut remaining, &mut total_work);
                }
            }
        } else {
            for perturbed in perturbations {
                if remaining == 0 {
                    break;
                }
                consider(run(perturbed, remaining), label, &mut best, &mut remaining, &mut total_work);
            }
        }
    }

    match best {
        Some((best_p, best_c, best_conv, method_name)) => Ok(nm_refinement(
            objective,
            best_p,
            best_c,
            best_conv,
            method_name,
            n_params_total,
            &mut remaining,
            &mut total_work,
        )),
        None => {
            if remaining == 0 {
                return Err(RunError {
                    message: "L-BFGS-B failed within the evaluation budget".into(),
                    work: total_work,
                });
            }
            // All optimizer attempts failed, fallback to Nelder-Mead
            let (p, c, n, conv) =
                run_nelder_mead(objective.clone(), unconstrained_start.to_vec(), remaining)
                    .map_err(|message| RunError {
                        message,
                        work: maxiter,
                    })?;
            consume_budget(&mut remaining, &mut total_work, n);
            Ok((p, c, total_work, conv, "nelder-mead (fallback)".to_string()))
        }
    }
}

/// Single-run BFGS (no multi-start) with Nelder-Mead fallback.
pub(super) fn fit_bfgs_single(
    objective: &SarimaxObjective,
    unconstrained_start: Vec<f64>,
    maxiter: u64,
) -> MethodOutcome {
    match run_bfgs(objective.clone(), unconstrained_start.clone(), maxiter) {
        Ok((p, c, n, conv)) => Ok((p, c, n, conv, "bfgs".to_string())),
        Err(_) => {
            let (p, c, n, conv) =
                run_nelder_mead(objective.clone(), unconstrained_start, maxiter)
                    .map_err(RunError::from)?;
            Ok((p, c, n, conv, "nelder-mead (fallback)".to_string()))
        }
    }
}

/// Single-run L-BFGS-B; on failure, escalates to multi-start with the remaining budget.
pub(super) fn fit_lbfgsb_single(
    objective: &SarimaxObjective,
    unconstrained_start: Vec<f64>,
    config: &SarimaxConfig,
    maxiter: u64,
) -> MethodOutcome {
    let bounds = compute_bounds(config);
    match run_lbfgsb(objective, unconstrained_start.clone(), bounds, maxiter) {
        Ok((p, c, n, conv)) => Ok((p, c, n, conv, "lbfgsb".into())),
        Err(error) => {
            let remaining = maxiter.saturating_sub(error.work);
            if remaining == 0 {
                return Err(error);
            }
            let n_restarts = compute_n_restarts(unconstrained_start.len(), config);
            match multistart(
                Method::LbfgsbMulti,
                objective,
                &unconstrained_start,
                config,
                remaining,
                n_restarts,
            ) {
                Ok((p, c, n, conv, method)) => {
                    Ok((p, c, n + error.work, conv, format!("{method} (fallback)")))
                }
                Err(mut failure) => {
                    failure.work += error.work;
                    Err(failure)
                }
            }
        }
    }
}

/// Gradient-weighted perturbations around `anchor` (basin hopping).
///
/// Directions with large |∂L/∂θ_i| at the seed carry most of the loss-landscape
/// information and are most likely to harbor distinct basins, so they are
/// perturbed more aggressively; near-zero-gradient dimensions stay close to the
/// anchor. The per-dimension scale is max(|seed|, |anchor|), floored, so exog
/// coefficients that drifted far from their seed get proportionally wider
/// exploration. With `nudge_seed`, the seed itself and a lightly kicked copy of
/// it are added as candidates (they may sit in a different basin than the
/// anchor). The LCG draw order is part of the contract: k·n hop draws, then the
/// nudge draws.
#[allow(clippy::too_many_arguments)]
fn hop_candidates(
    objective: &SarimaxObjective,
    seed: &[f64],
    anchor: &[f64],
    config: &SarimaxConfig,
    alpha: f64,
    lcg_seed: u64,
    k: usize,
    nudge_seed: bool,
) -> Vec<Vec<f64>> {
    let n_dim = seed.len();
    let grad = match objective.analytical_gradient_negloglike(seed) {
        Ok(g) if g.len() == n_dim => g,
        _ => vec![1.0; n_dim], // fall back to uniform perturbation
    };
    // Per-dim relative magnitudes ∈ [0, 1], floored so no dim is fully frozen
    let max_abs = grad
        .iter()
        .map(|g| g.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-12);
    let rel_grad: Vec<f64> = grad.iter().map(|g| (g.abs() / max_abs).max(0.1)).collect();

    let mut lcg_state = lcg_seed;
    let mut next_u = || -> f64 {
        lcg_state = lcg_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((lcg_state >> 32) as u32) as f64 / (u32::MAX as f64);
        2.0 * u - 1.0
    };
    let dim_scale: Vec<f64> = (0..n_dim)
        .map(|i| (0.6 * seed[i].abs().max(anchor[i].abs())).max(0.5))
        .collect();
    let mut candidates: Vec<Vec<f64>> = (0..k)
        .map(|_| -> Vec<f64> {
            (0..n_dim)
                .map(|i| anchor[i] + alpha * dim_scale[i] * rel_grad[i] * next_u())
                .collect()
        })
        .filter(|p: &Vec<f64>| passes_cancellation_filter(p, config))
        .collect();
    if nudge_seed {
        if passes_cancellation_filter(seed, config) {
            candidates.push(seed.to_vec());
        }
        let nudged: Vec<f64> = (0..n_dim)
            .map(|i| seed[i] + 0.1 * dim_scale[i] * next_u())
            .collect();
        if passes_cancellation_filter(&nudged, config) {
            candidates.push(nudged);
        }
    }
    candidates
}

/// Run L-BFGS-B from every candidate in parallel and offer each result to `best`.
fn polish_in_parallel(
    objective: &SarimaxObjective,
    candidates: Vec<Vec<f64>>,
    bounds: &Bounds,
    budget: u64,
    label: &str,
    best: &mut Best,
    total_work: &mut u64,
) {
    if candidates.is_empty() || budget == 0 || crate::batch::guard_rayon_fork().is_err() {
        return;
    }
    for result in par_lbfgsb(objective, candidates, bounds, budget) {
        match result {
            Ok((p, c, n, conv)) => {
                *total_work += n;
                try_update_best(best, p, c, conv, label);
            }
            Err(error) => *total_work += error.work,
        }
    }
}

/// Gradient-informed basin hopping for SARIMAX MLE.
///
/// **Novel contribution**: adapts the basin hopping / iterated local search
/// framework from large-scale non-convex optimization (Wales & Doye 1997) to
/// time-series MLE. Two key departures from prior ARIMA multi-start work:
///
///   1. **Anchor at the converged local optimum, not the seed.**
///      Standard multi-start perturbs the *seed* (CSS-style start params).
///      But all small perturbations of the seed fall into the same basin of
///      attraction the seed itself lies in — they re-converge to the same
///      local optimum, learning nothing new. Perturbing the *converged*
///      point lets us escape into neighboring basins, which is the whole
///      point of basin hopping.
///
///   2. **Weight perturbation magnitude per dimension by the seed gradient.**
///      See [`hop_candidates`].
///
/// Algorithm:
///   1. Run L-BFGS-B from seed → converged anchor θ★ with cost c★.
///   2. Generate K gradient-weighted perturbations around θ★ (plus the seed
///      and a nudged seed).
///   3. Run them in parallel via Rayon.
///   4. Return the solution with lowest cost across {θ★, θ_1, …, θ_K}.
///
/// Wall-time cost: ~2× single-start (1× baseline + 1× parallel for K runs).
pub(super) fn fit_lbfgsb_adaptive_restart(
    objective: &SarimaxObjective,
    unconstrained_start: Vec<f64>,
    config: &SarimaxConfig,
    maxiter: u64,
    n_restarts: usize,
) -> MethodOutcome {
    let bounds = compute_bounds(config);
    let k = n_restarts.max(3);
    let baseline_budget = (maxiter / 2).max(1);
    let mut total_work = 0_u64;

    // 1. Baseline from seed — run to convergence. The result is our anchor for
    //    basin hopping (or the seed itself if the baseline failed).
    let mut best: Best = match fit_lbfgsb_single(
        objective,
        unconstrained_start.clone(),
        config,
        baseline_budget,
    ) {
        Ok((p, c, n, conv, name)) => {
            total_work += n;
            Some((p, c, conv, name))
        }
        Err(error) => {
            total_work += error.work;
            None
        }
    };
    let anchor: Vec<f64> = best
        .as_ref()
        .map(|(p, _, _, _)| p.clone())
        .unwrap_or_else(|| unconstrained_start.clone());

    // 2. Hop around the anchor, weighted by the seed gradient.
    let perturbations = hop_candidates(
        objective,
        &unconstrained_start,
        &anchor,
        config,
        1.0,
        0x9E37_79B9_7F4A_7C15,
        k,
        true,
    );

    // 3. Run them in parallel with the remaining budget split evenly.
    let per_restart_budget = maxiter.saturating_sub(total_work) / perturbations.len().max(1) as u64;
    polish_in_parallel(
        objective,
        perturbations,
        &bounds,
        per_restart_budget,
        "lbfgsb-adaptive",
        &mut best,
        &mut total_work,
    );

    match best {
        Some((p, c, conv, name)) => Ok((p, c, total_work, conv, name)),
        None => {
            let remaining = maxiter.saturating_sub(total_work);
            fit_lbfgsb_single(objective, unconstrained_start, config, remaining)
                .map(|(p, c, n, conv, name)| (p, c, total_work + n, conv, name))
                .map_err(|mut error| {
                    error.work += total_work;
                    error
                })
        }
    }
}

/// Hybrid: multi-start followed by gradient-informed adaptive polish.
///
/// Combines the diversity of [`multistart`] (zero-start, MA grid, LCG
/// perturbations of the seed — explores many basins) with the basin hopping of
/// [`fit_lbfgsb_adaptive_restart`] (perturb around the *converged* best,
/// gradient-weighted — escapes the best-so-far basin into neighboring ones).
///
/// Wall-time: ~1.5× multi-start (multi + parallel K polish).
pub(super) fn fit_lbfgsb_hybrid(
    objective: &SarimaxObjective,
    unconstrained_start: &[f64],
    config: &SarimaxConfig,
    maxiter: u64,
    n_restarts: usize,
) -> MethodOutcome {
    // Stage 1: standard multi-start
    let multi_budget = (maxiter / 2).max(1);
    let (multi_p, multi_c, multi_iter, multi_conv, orig_name) = multistart(
        Method::LbfgsbMulti,
        objective,
        unconstrained_start,
        config,
        multi_budget,
        n_restarts,
    )?;
    let mut total_work = multi_iter;
    let k = n_restarts.max(3);
    let bounds = compute_bounds(config);

    // Stage 2: hop around the multi-start best. alpha is larger than for
    // adaptive-alone because multi already supplied a strong seed.
    let perturbations = hop_candidates(
        objective,
        unconstrained_start,
        &multi_p,
        config,
        1.5,
        0xA7F3_5B2E_1290_C84B,
        k,
        false,
    );

    let mut best: Best = Some((
        multi_p,
        multi_c,
        multi_conv,
        if orig_name.contains("nm") {
            "lbfgsb-hybrid+nm"
        } else {
            "lbfgsb-hybrid"
        }
        .into(),
    ));

    let polish_budget = maxiter.saturating_sub(total_work) / perturbations.len().max(1) as u64;
    polish_in_parallel(
        objective,
        perturbations,
        &bounds,
        polish_budget,
        "lbfgsb-hybrid",
        &mut best,
        &mut total_work,
    );

    match best {
        Some((p, c, conv, name)) => Ok((p, c, total_work, conv, name)),
        None => unreachable!("multi-start supplied an incumbent"),
    }
}
