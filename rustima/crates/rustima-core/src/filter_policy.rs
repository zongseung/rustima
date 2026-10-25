//! Per-step decisions shared by the Kalman recursions.
//!
//! Three loops exist on purpose — `kalman_core`, `kalman_filter_batched` (shared-P
//! batch specialisation) and `score::tlkf_pass` (tangent-linear) — plus the
//! profiled GLS objective. Merging them was evaluated and rejected twice
//! (2026-07-22, d80260d). What they must agree on lives here instead: when F_t is
//! usable, what the fallback variance is, and when the covariance recursion is
//! frozen at its steady state. A loglike and its gradient that freeze at different
//! steps disagree by ~1e-9, below any value-based test, so the decision is shared
//! rather than copied.

use nalgebra::{DMatrix, DVector};

/// Steady-state convergence tolerance for the gain vector pz = P*Z.
///
/// We check pz convergence (k-vector) because F_t = Z'*pz (scalar) can
/// converge while pz changes in directions orthogonal to Z, leading to
/// incorrect cached K_inf = T*pz/F. Error analysis shows the loglike
/// error from a converged-but-imprecise K_inf is O(eps * n^2) due to
/// compounding through the state recursion (unit-root states integrate
/// the error), so eps must be tight.
///
/// At 1e-9, loglike error stays below ~1e-5 for n=1000.
/// For models where pz converges slowly, steady-state won't trigger
/// and the sparse T*P*T' path provides the primary speedup instead.
pub(crate) const STEADY_STATE_TOL: f64 = 1e-9;

/// Minimum steps past burn-in before checking for convergence.
pub(crate) const STEADY_STATE_MIN_STEPS: usize = 5;

/// Number of consecutive converged pz values required.
pub(crate) const STEADY_STATE_CONSEC: usize = 3;

/// Penalty variance used when the Kalman innovation variance F_t is not
/// usable at observation t (a sign the model is at a near-non-stationary
/// boundary, e.g. AR roots ≈ unit circle). Instead of aborting, we contribute
/// `log F_safe` to the log-likelihood for that observation, yielding a smooth,
/// finite-but-bad cost the optimizer can back away from. Value ≈ 1e10 means
/// each offending observation costs ~23 log-likelihood units — small enough
/// to be meaningful, large enough to dominate any real innovation magnitude.
pub(crate) const KF_FT_FALLBACK_VARIANCE: f64 = 1.0e10;

/// Whether F_t can weight an update. NaN and F_t <= 0 cannot; +inf passes and
/// trips the non-finite loglike error downstream, as it always has.
#[inline(always)]
pub(crate) fn f_usable(f_t: f64) -> bool {
    f_t > 0.0
}

/// Detects pz = P·Z convergence and hands back F_inf once it is safe to freeze.
///
/// Call [`observe`](Self::observe) on usable-F steps only, right after the
/// covariance predict (so `p` is P_{t+1|t}); fallback steps leave it untouched.
pub(crate) struct SteadyStateDetector {
    pz_prev: DVector<f64>,
    consec: usize,
    /// Step at which the steady state was adopted, if it was.
    pub(crate) onset: Option<usize>,
}

impl SteadyStateDetector {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            pz_prev: DVector::zeros(k),
            consec: 0,
            onset: None,
        }
    }

    /// Overwrites `pz` with P·Z. `Some(f_inf)` means freeze now; `pz` then holds
    /// pz_inf. Dense gemv/dot on purpose: this is the one place the number is
    /// computed, so every loop sees the same bits.
    #[inline]
    pub(crate) fn observe(
        &mut self,
        t: usize,
        burn: usize,
        p: &DMatrix<f64>,
        z: &DVector<f64>,
        pz: &mut DVector<f64>,
    ) -> Option<f64> {
        if t < burn + STEADY_STATE_MIN_STEPS {
            return None;
        }
        pz.gemv(1.0, p, z, 0.0);
        let converged = check_convergence(pz, &self.pz_prev, &mut self.consec);
        self.pz_prev.copy_from(pz);
        if !converged {
            return None;
        }
        let f_steady = z.dot(pz);
        // Only adopt if F_inf is positive and finite; otherwise stay on the non-converged path.
        (f_steady > 0.0 && f_steady.is_finite()).then(|| {
            self.onset = Some(t);
            f_steady
        })
    }
}

/// Returns true once the relative change in pz has been below `STEADY_STATE_TOL`
/// for `STEADY_STATE_CONSEC` consecutive steps.
#[inline]
fn check_convergence(pz: &DVector<f64>, pz_prev: &DVector<f64>, consec_count: &mut usize) -> bool {
    let pz_diff_sq: f64 = pz
        .iter()
        .zip(pz_prev.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum();
    let pz_norm_sq: f64 = pz_prev.iter().map(|v| v * v).sum();
    let pz_norm = pz_norm_sq.sqrt().max(1e-15);

    if pz_diff_sq.sqrt() / pz_norm < STEADY_STATE_TOL {
        *consec_count += 1;
        *consec_count >= STEADY_STATE_CONSEC
    } else {
        *consec_count = 0;
        false
    }
}
