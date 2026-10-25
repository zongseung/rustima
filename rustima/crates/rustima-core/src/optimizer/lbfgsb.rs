//! Basin L-BFGS-B evaluation and method aliases.

use std::cell::Cell;

use crate::error::SarimaxError;

pub(super) const INVALID_COST: f64 = f64::MAX / 2.0;

/// A parsed optimizer method. `fit` parses the name once; everything downstream
/// matches on the variant, never on the string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Method {
    Lbfgsb,
    LbfgsbMulti,
    LbfgsbAdaptive,
    LbfgsbHybrid,
    LbfgsbStrict,
    Lbfgs,
    Bfgs,
    TrustRegion,
    ProfileTrustRegion,
    NelderMead,
}

/// `(name, method, accepts the "basin-" prefix)`.
const METHODS: &[(&str, Method, bool)] = &[
    ("lbfgsb", Method::Lbfgsb, true),
    ("lbfgsb-multi", Method::LbfgsbMulti, true),
    ("lbfgsb-adaptive", Method::LbfgsbAdaptive, true),
    ("lbfgsb-hybrid", Method::LbfgsbHybrid, true),
    ("lbfgsb-strict", Method::LbfgsbStrict, true),
    ("lbfgsb_single", Method::LbfgsbStrict, false),
    ("lbfgs", Method::Lbfgs, false),
    ("bfgs", Method::Bfgs, false),
    ("trust-region", Method::TrustRegion, false),
    ("profile-trust-region", Method::ProfileTrustRegion, false),
    ("nelder-mead", Method::NelderMead, false),
    ("nm", Method::NelderMead, false),
];

impl Method {
    /// Keep validation ahead of fast paths so unavailable methods never succeed silently.
    pub(super) fn parse(method: &str) -> Result<Self, SarimaxError> {
        let (name, basin) = match method.strip_prefix("basin-") {
            Some(rest) => (rest, true),
            None => (method, false),
        };
        METHODS
            .iter()
            .find(|(n, _, basin_ok)| *n == name && (!basin || *basin_ok))
            .map(|(_, m, _)| *m)
            .ok_or_else(|| {
                let names = |basin: bool| {
                    METHODS
                        .iter()
                        .filter(|(_, _, ok)| !basin || *ok)
                        .map(|(n, _, _)| if basin { format!("basin-{n}") } else { n.to_string() })
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                SarimaxError::OptimizationFailed(format!(
                    "unknown method: '{method}'. Valid methods: {}, {}",
                    names(false),
                    names(true),
                ))
            })
    }

    /// Label reported when no optimizer ran (zero-parameter or `maxiter=0` results).
    pub(super) fn label(self) -> &'static str {
        match self {
            Method::Lbfgsb => "basin-lbfgsb",
            Method::LbfgsbMulti => "basin-lbfgsb-multi",
            Method::LbfgsbAdaptive => "basin-lbfgsb-adaptive",
            Method::LbfgsbHybrid => "basin-lbfgsb-hybrid",
            Method::LbfgsbStrict => "basin-lbfgsb-strict",
            Method::Lbfgs => "lbfgs",
            Method::Bfgs => "bfgs",
            Method::TrustRegion => "trust-region",
            Method::ProfileTrustRegion => "profile-trust-region",
            Method::NelderMead => "nelder-mead",
        }
    }
}

impl From<String> for RunError {
    /// Basin solver errors carry no work count.
    fn from(message: String) -> Self {
        RunError { message, work: 0 }
    }
}

impl From<RunError> for SarimaxError {
    fn from(e: RunError) -> Self {
        SarimaxError::OptimizationFailed(e.message)
    }
}

pub(super) trait Objective {
    fn cost(&self, x: &[f64]) -> f64;
    fn evaluate(&self, x: &[f64], gradient: &mut [f64]) -> f64;
}

#[derive(Debug)]
pub(super) struct RunError {
    pub(super) message: String,
    pub(super) work: u64,
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(f)
    }
}

pub(super) type Outcome = Result<(Vec<f64>, f64, u64, bool), RunError>;
pub(super) type MethodOutcome = Result<(Vec<f64>, f64, u64, bool, String), RunError>;

pub(super) fn minimize<E: Objective>(
    objective: &E,
    start: Vec<f64>,
    bounds: Vec<(Option<f64>, Option<f64>)>,
    limit: u64,
) -> Outcome {
    if limit == 0 {
        return Err(RunError {
            message: "L-BFGS-B evaluation budget exhausted".into(),
            work: 0,
        });
    }
    basin_backend::minimize(objective, start, bounds, limit)
}

mod basin_backend {
    use super::*;
    use basin::{
        BoxBounds, ConstraintDescription, ConstraintSet, ConstraintSolveError, Constraints,
        CostFunction, Executor, FirstOrderState, Gradient, Lbfgsb, State, StepOutcome, Termination,
    };

    #[derive(Debug, thiserror::Error)]
    #[error("L-BFGS-B evaluation budget exhausted")]
    struct Exhausted;

    struct Problem<'a, E> {
        objective: &'a E,
        lower: Vec<f64>,
        upper: Vec<f64>,
        work: &'a Cell<u64>,
        limit: u64,
    }

    impl<E> Constraints for Problem<'_, E> {
        type Set = ConstraintSet<BoxBounds<Vec<f64>>>;
        fn constraints(&self) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
            ConstraintSet::new().with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
        }
    }

    impl<E: Objective> Problem<'_, E> {
        fn evaluate(&self, x: &[f64]) -> Result<(f64, Vec<f64>), Exhausted> {
            if self.work.get() >= self.limit {
                return Err(Exhausted);
            }
            self.work.set(self.work.get() + 1);
            let mut gradient = vec![0.0; x.len()];
            let cost = self.objective.evaluate(x, &mut gradient);
            // Basin rejects non-finite trials. The legacy finite penalty must
            // never turn an invalid point into apparent gradient convergence.
            let cost = if cost < INVALID_COST && cost.is_finite() {
                cost
            } else {
                f64::INFINITY
            };
            Ok((cost, gradient))
        }
    }

    impl<E: Objective> CostFunction for Problem<'_, E> {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Exhausted;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Exhausted> {
            self.evaluate(x).map(|(cost, _)| cost)
        }
    }

    impl<E: Objective> Gradient for Problem<'_, E> {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Exhausted> {
            self.evaluate(x).map(|(_, gradient)| gradient)
        }
        fn cost_and_gradient(&self, x: &Vec<f64>) -> Result<(f64, Vec<f64>), Exhausted> {
            self.evaluate(x)
        }
    }

    pub(super) fn minimize<E: Objective>(
        objective: &E,
        start: Vec<f64>,
        bounds: Vec<(Option<f64>, Option<f64>)>,
        limit: u64,
    ) -> Outcome {
        let work = Cell::new(0);
        let (lower, upper) = bounds
            .into_iter()
            .map(|(lower, upper)| {
                (
                    lower.unwrap_or(f64::NEG_INFINITY),
                    upper.unwrap_or(f64::INFINITY),
                )
            })
            .unzip();
        let problem = Problem {
            objective,
            lower,
            upper,
            work: &work,
            limit,
        };
        let solver = Lbfgsb::new()
            .with_m_capacity(10)
            .with_absolute_projected_gradient_tolerance(1e-5)
            .with_relative_cost_reduction_tolerance(1e7 * f64::EPSILON);
        let mut stepper = Executor::new(problem, solver, FirstOrderState::new(start))
            .max_iter(u64::MAX)
            .into_stepper()
            .map_err(|error| RunError {
                message: format!("Basin L-BFGS-B failed: {error}"),
                work: work.get(),
            })?;
        loop {
            // A callback error consumes Basin's state. Save the accepted point
            // before stepping so mid-search exhaustion can return it intact.
            let accepted = (stepper.state().param().clone(), stepper.state().cost());
            match stepper.step() {
                Ok(StepOutcome::Continue) => {}
                Ok(StepOutcome::Stopped(report)) => {
                    if let Termination::Failed(reason) = report.termination {
                        return Err(RunError {
                            message: format!("Basin L-BFGS-B failed: {reason:?}"),
                            work: work.get(),
                        });
                    }
                    let state = stepper.state();
                    if !state.cost().is_finite() || state.cost() >= INVALID_COST {
                        return Err(RunError {
                            message: "Basin L-BFGS-B returned an invalid objective".into(),
                            work: work.get(),
                        });
                    }
                    return Ok((
                        state.param().clone(),
                        state.cost(),
                        work.get(),
                        matches!(report.termination, Termination::Converged(_)),
                    ));
                }
                Err(ConstraintSolveError::Evaluation(Exhausted)) => {
                    return Ok((accepted.0, accepted.1, work.get(), false));
                }
                Err(error) => {
                    return Err(RunError {
                        message: format!("Basin L-BFGS-B failed: {error}"),
                        work: work.get(),
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_selection_and_validation() {
        assert_eq!(Method::parse("lbfgsb").unwrap(), Method::Lbfgsb);
        assert!(Method::parse("basin-bfgs").is_err());
        assert!(Method::parse("basin-lbfgsb_single").is_err());
        assert_eq!(Method::parse("basin-lbfgsb-multi").unwrap(), Method::LbfgsbMulti);
        assert_eq!(Method::parse("lbfgsb_single").unwrap().label(), "basin-lbfgsb-strict");
        assert_eq!(Method::parse("nm").unwrap().label(), "nelder-mead");
    }

    mod basin {
        use super::*;

        struct Quadratic(Vec<f64>);
        impl Objective for Quadratic {
            fn cost(&self, x: &[f64]) -> f64 {
                x.iter()
                    .zip(&self.0)
                    .map(|(x, target)| 0.5 * (x - target).powi(2))
                    .sum()
            }
            fn evaluate(&self, x: &[f64], gradient: &mut [f64]) -> f64 {
                for ((g, x), target) in gradient.iter_mut().zip(x).zip(&self.0) {
                    *g = x - target;
                }
                self.cost(x)
            }
        }

        #[test]
        fn bounds_include_absent_and_one_sided_endpoints() {
            let objective = Quadratic(vec![2.0, -2.0, 3.0, 4.0]);
            let bounds = vec![
                (None, None),
                (Some(0.0), None),
                (None, Some(1.0)),
                (Some(-1.0), Some(1.0)),
            ];
            let (params, cost, work, converged) =
                minimize(&objective, vec![0.0, -3.0, 2.0, 0.0], bounds, 100).unwrap();
            assert!(converged);
            assert!(work <= 100);
            for (got, expected) in params.iter().zip([2.0, 0.0, 1.0, 1.0]) {
                assert!((got - expected).abs() < 1e-8);
            }
            assert_eq!(cost, objective.cost(&params));
        }

        struct Rosenbrock {
            calls: Cell<u64>,
        }
        impl Objective for Rosenbrock {
            fn cost(&self, x: &[f64]) -> f64 {
                (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
            }
            fn evaluate(&self, x: &[f64], g: &mut [f64]) -> f64 {
                self.calls.set(self.calls.get() + 1);
                g[0] = 2.0 * (x[0] - 1.0) - 400.0 * x[0] * (x[1] - x[0].powi(2));
                g[1] = 200.0 * (x[1] - x[0].powi(2));
                self.cost(x)
            }
        }

        #[test]
        fn budget_exhaustion_inside_line_search_retains_accepted_point() {
            for limit in [1, 2] {
                let objective = Rosenbrock {
                    calls: Cell::new(0),
                };
                let start = vec![-1.2, 1.0];
                let (params, cost, work, converged) =
                    minimize(&objective, start.clone(), vec![(None, None); 2], limit).unwrap();
                assert_eq!(params, start);
                assert_eq!(cost, objective.cost(&start));
                assert!(!converged);
                assert_eq!(work, limit);
                assert_eq!(objective.calls.get(), limit);
            }
        }

        #[test]
        fn stationary_initial_point_can_converge_with_one_evaluation() {
            let (_, cost, work, converged) =
                minimize(&Quadratic(vec![1.0]), vec![1.0], vec![(None, None)], 1).unwrap();
            assert!(converged);
            assert_eq!(cost, 0.0);
            assert_eq!(work, 1);
        }

        struct BadObjective {
            invalid: bool,
        }
        impl Objective for BadObjective {
            fn cost(&self, x: &[f64]) -> f64 {
                if self.invalid {
                    INVALID_COST
                } else {
                    x[0].powi(2)
                }
            }
            fn evaluate(&self, x: &[f64], g: &mut [f64]) -> f64 {
                g[0] = if self.invalid { 0.0 } else { f64::NAN };
                self.cost(x)
            }
        }

        #[test]
        fn numerical_failure_and_invalid_objective_do_not_converge() {
            for invalid in [false, true] {
                let error = minimize(
                    &BadObjective { invalid },
                    vec![1.0],
                    vec![(None, None)],
                    100,
                )
                .unwrap_err();
                assert!(error.work > 0 && error.work <= 100);
                assert!(error.message.contains("failed"));
            }
        }
    }
}
