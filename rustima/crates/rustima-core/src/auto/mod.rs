//! Shared Auto ARIMA differencing, candidate search, and selection for all bindings.

mod differencing;
#[cfg(test)]
mod tests;

use std::collections::HashSet;

use crate::batch::grid_search_fit;
use crate::error::{Result, SarimaxError};
use crate::types::{FitResult, SarimaxConfig, SarimaxOrder};
use crate::{optimizer, validation};

#[derive(Debug, Clone, Copy)]
pub enum Criterion {
    Aic,
    Bic,
    Hqic,
}

impl Criterion {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "aic" => Ok(Self::Aic),
            "bic" => Ok(Self::Bic),
            "hqic" => Ok(Self::Hqic),
            _ => Err(SarimaxError::InvalidInput(format!(
                "Unknown criterion: {name}"
            ))),
        }
    }

    fn value(self, fit: &FitResult) -> f64 {
        let ic = match self {
            Self::Aic => fit.aic,
            Self::Bic => fit.bic,
            Self::Hqic if fit.n_obs > 1 => {
                -2.0 * fit.loglike + 2.0 * fit.n_params as f64 * (fit.n_obs as f64).ln().ln()
            }
            Self::Hqic => f64::INFINITY,
        };
        if ic.is_finite() {
            ic
        } else {
            f64::INFINITY
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub max_p: usize,
    pub max_q: usize,
    pub max_d: usize,
    pub max_pp: usize,
    pub max_qq: usize,
    pub max_dd: usize,
    pub period: usize,
    pub d: Option<usize>,
    pub dd: Option<usize>,
    pub stepwise: bool,
    pub criterion: Criterion,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            max_p: 5,
            max_q: 5,
            max_d: 2,
            max_pp: 2,
            max_qq: 2,
            max_dd: 1,
            period: 0,
            d: None,
            dd: None,
            stepwise: true,
            criterion: Criterion::Aic,
        }
    }
}

/// Select nonseasonal and seasonal differences independently of model fitting.
pub fn difference_orders(
    y: &[f64],
    max_d: usize,
    max_dd: usize,
    period: usize,
) -> Result<(usize, usize)> {
    validation::validate_endog(y)?;
    validation::validate_order(
        0,
        max_d,
        0,
        0,
        if period > 1 { max_dd } else { 0 },
        0,
        period,
        0,
    )?;
    Ok((
        differencing::ndiffs(y, max_d)?,
        differencing::nsdiffs(y, period, max_dd),
    ))
}

#[derive(Debug, Clone)]
pub struct CandidateEvaluation {
    pub order: SarimaxOrder,
    pub ic: Option<f64>,
    pub converged: bool,
    pub error: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SelectedModel {
    pub order: SarimaxOrder,
    pub fit: FitResult,
    pub ic: f64,
}

#[derive(Debug)]
pub struct SearchResult {
    pub best: Option<SelectedModel>,
    pub history: Vec<CandidateEvaluation>,
    pub d: usize,
    pub dd: usize,
}

#[derive(Default)]
struct Selection {
    any: Option<SelectedModel>,
    converged: Option<SelectedModel>,
    history: Vec<CandidateEvaluation>,
}

impl Selection {
    fn accept(
        &mut self,
        order: SarimaxOrder,
        result: Result<FitResult>,
        criterion: Criterion,
    ) -> bool {
        let fit = match result {
            Ok(fit) => fit,
            Err(e) => {
                self.history.push(CandidateEvaluation {
                    order,
                    ic: None,
                    converged: false,
                    error: Some(e.to_string()),
                    warnings: vec![],
                });
                return false;
            }
        };
        let ic = criterion.value(&fit);
        self.history.push(CandidateEvaluation {
            order: order.clone(),
            ic: ic.is_finite().then_some(ic),
            converged: fit.converged,
            error: None,
            warnings: fit.warnings.clone(),
        });
        if !ic.is_finite() {
            return false;
        }
        let improved = self.any.as_ref().is_none_or(|b| ic < b.ic);
        let selected = SelectedModel { order, fit, ic };
        if selected.fit.converged && self.converged.as_ref().is_none_or(|b| ic < b.ic) {
            self.converged = Some(selected.clone());
        }
        if improved {
            self.any = Some(selected);
        }
        improved
    }

    fn finish(self, d: usize, dd: usize) -> SearchResult {
        // Preserve the existing 5-IC-unit soft preference for converged models.
        let best = match (self.any, self.converged) {
            (Some(any), Some(converged)) if converged.ic - any.ic < 5.0 => Some(converged),
            (any, _) => any,
        };
        SearchResult {
            best,
            history: self.history,
            d,
            dd,
        }
    }
}

/// Search fits the same candidate batches in deterministic order in either binding.
/// The base configuration supplies flags, trend and exogenous-variable count;
/// its order is replaced by the search options.
pub fn search(
    y: &[f64],
    base: &SarimaxConfig,
    options: &SearchOptions,
    method: Option<&str>,
    maxiter: Option<u64>,
    exog: Option<&[Vec<f64>]>,
) -> Result<SearchResult> {
    validation::validate_endog(y)?;
    let s = options.period;
    if s == 1 {
        return Err(SarimaxError::InvalidInput(
            "seasonal period must be 0 or at least 2".into(),
        ));
    }
    let max_pp = if s == 0 { 0 } else { options.max_pp };
    let max_qq = if s == 0 { 0 } else { options.max_qq };
    // Check bounds before differencing or allocating the candidate grid.
    validation::validate_order(
        options.max_p,
        0,
        options.max_q,
        max_pp,
        0,
        max_qq,
        s,
        base.n_exog,
    )?;
    validation::validate_order(0, options.max_d, 0, 0, 0, 0, 0, 0)?;
    validation::validate_order(0, 0, 0, 0, if s == 0 { 0 } else { options.max_dd }, 0, 2, 0)?;
    let d = match options.d {
        Some(d) => d,
        None => differencing::ndiffs(y, options.max_d)?,
    };
    let dd = options
        .dd
        .unwrap_or_else(|| differencing::nsdiffs(y, s, options.max_dd));
    let mut config = base.clone();
    config.order = SarimaxOrder::new(options.max_p, d, options.max_q, max_pp, dd, max_qq, s);
    validation::validate_request(y, &config, exog)?;
    let method = method.unwrap_or(if base.n_exog > 0 {
        "profile-trust-region"
    } else {
        "lbfgsb"
    });
    optimizer::validate_method(method)?;
    let max = [options.max_p, options.max_q, max_pp, max_qq];
    let mut pending = Vec::new();
    if options.stepwise {
        pending.extend([
            [0, 0, 0, 0],
            if max[0] >= 2 && max[1] >= 2 {
                [2, 2, 0, 0]
            } else {
                [1, 1, 0, 0]
            },
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]);
        if s > 0 {
            pending.extend([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]);
        }
        for key in &mut pending {
            for i in 0..4 {
                key[i] = key[i].min(max[i]);
            }
        }
    } else {
        for p in 0..=max[0] {
            for q in 0..=max[1] {
                for pp in 0..=max[2] {
                    for qq in 0..=max[3] {
                        pending.push([p, q, pp, qq]);
                    }
                }
            }
        }
    }
    let mut visited = HashSet::new();
    let mut selection = Selection::default();
    loop {
        // Keep initial candidate ordering, including clamped duplicates, for history compatibility.
        let fresh: Vec<_> = pending
            .into_iter()
            .filter(|key| !visited.contains(key))
            .collect();
        if fresh.is_empty() {
            break;
        }
        visited.extend(fresh.iter().copied());
        let configs: Vec<_> = fresh
            .into_iter()
            .map(|[p, q, pp, qq]| {
                let mut c = config.clone();
                c.order = SarimaxOrder::new(p, d, q, pp, dd, qq, s);
                c
            })
            .collect();
        let fits = grid_search_fit(y, &configs, Some(method), maxiter, exog);
        let mut improved = false;
        for (c, fit) in configs.into_iter().zip(fits) {
            improved |= selection.accept(c.order, fit, options.criterion);
        }
        if !options.stepwise || !improved {
            break;
        }
        let Some(best) = &selection.any else {
            break;
        };
        let o = &best.order;
        let anchor = [o.p, o.q, o.pp, o.qq];
        pending = Vec::new();
        for delta in [
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [1, 1, 0, 0],
            [-1, -1, 0, 0],
        ] {
            let key: [isize; 4] = std::array::from_fn(|i| anchor[i] as isize + delta[i]);
            if (0..4).all(|i| key[i] >= 0 && key[i] <= max[i] as isize) {
                pending.push(key.map(|v| v as usize));
            }
        }
    }
    Ok(selection.finish(d, dd))
}
