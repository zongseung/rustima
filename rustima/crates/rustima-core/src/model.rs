//! The model: validated config + raw data, and the observations as the Kalman
//! filter sees them.
//!
//! `simple_differencing` is applied in [`Prepared::new`] and nowhere else: every
//! state space, Kalman init and filter pass is built from a [`Prepared`], so fit,
//! forecast, residuals and inference cannot disagree about which series they filter.
//! Input validation happens once, in [`SarimaxModel::new`]; the Python and R
//! adapters only convert types.

use crate::css::apply_differencing;
use crate::error::{Result, SarimaxError};
use crate::forecast::{ForecastResult, ResidualOutput, RollingForecastOutput};
use crate::inference::{DiagnosticsResult, InferenceResult};
use crate::initialization::KalmanInit;
use crate::kalman::{self, KalmanFilterOutput, KalmanOutput};
use crate::params::SarimaxParams;
use crate::state_space::StateSpace;
use crate::types::{FitResult, SarimaxConfig};

/// Endog/exog after `simple_differencing` (or the raw series when it is off).
#[derive(Clone)]
pub(crate) struct Prepared {
    config: SarimaxConfig,
    endog: Vec<f64>,
    exog: Option<Vec<Vec<f64>>>,
}

impl Prepared {
    /// Takes the raw, undifferenced series. Under `simple_differencing` endog and
    /// every exog column are differenced the same way (seasonal D times, then d times).
    pub(crate) fn new(
        endog: &[f64],
        exog: Option<&[Vec<f64>]>,
        config: &SarimaxConfig,
    ) -> Result<Self> {
        if !config.simple_differencing {
            return Ok(Self {
                config: config.clone(),
                endog: endog.to_vec(),
                exog: exog.map(<[_]>::to_vec),
            });
        }
        let eff = apply_differencing(endog, config);
        let eff_exog = exog
            .map(|cols| {
                cols.iter()
                    .enumerate()
                    .map(|(j, col)| {
                        let xj = apply_differencing(col, config);
                        if xj.len() != eff.len() {
                            return Err(SarimaxError::InvalidInput(format!(
                                "exog column {} has {} rows after simple_differencing, but y has {} observations",
                                j, xj.len(), eff.len()
                            )));
                        }
                        Ok(xj)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        Ok(Self {
            config: config.clone(),
            endog: eff,
            exog: eff_exog,
        })
    }

    pub(crate) fn config(&self) -> &SarimaxConfig {
        &self.config
    }

    pub(crate) fn endog(&self) -> &[f64] {
        &self.endog
    }

    pub(crate) fn exog(&self) -> Option<&[Vec<f64>]> {
        self.exog.as_deref()
    }

    /// State-space matrices and Kalman init at `params` — the only way to build them.
    pub(crate) fn system(&self, params: &SarimaxParams) -> Result<(StateSpace, KalmanInit)> {
        let ss = StateSpace::new(&self.config, params, &self.endog, self.exog())?;
        let init = KalmanInit::from_config_default(&ss, &self.config);
        Ok((ss, init))
    }

    pub(crate) fn loglike(&self, params: &SarimaxParams) -> Result<KalmanOutput> {
        let (ss, init) = self.system(params)?;
        kalman::kalman_loglike(&self.endog, &ss, &init, self.config.concentrate_scale)
    }

    pub(crate) fn filter(&self, params: &SarimaxParams) -> Result<KalmanFilterOutput> {
        let (ss, init) = self.system(params)?;
        kalman::kalman_filter(&self.endog, &ss, &init, self.config.concentrate_scale)
    }
}

/// Difference future exog rows the way [`Prepared::new`] differenced the in-sample
/// columns, seeding each column with its last `d + s·D` in-sample rows.
pub(crate) fn difference_future_exog(
    past: &[Vec<f64>],
    future: &[Vec<f64>],
    config: &SarimaxConfig,
) -> Vec<Vec<f64>> {
    let n_drop = config.order.k_states_diff();
    past.iter()
        .zip(future)
        .map(|(p, f)| {
            let mut joined = p[p.len().saturating_sub(n_drop)..].to_vec();
            joined.extend_from_slice(f);
            let x = apply_differencing(&joined, config);
            x[x.len().saturating_sub(f.len())..].to_vec()
        })
        .collect()
}

/// A SARIMAX specification bound to its data. Params are always the constrained
/// flat vector `[trend | exog | ar | ma | sar | sma | sigma2?]` (see
/// [`SarimaxConfig::param_names`]); every method checks its length and finiteness.
pub struct SarimaxModel {
    endog: Vec<f64>,
    exog: Option<Vec<Vec<f64>>>,
    data: Prepared,
}

impl SarimaxModel {
    /// Validates the request (order bounds, finite data, exog shape) and prepares the data.
    /// `exog` is column-major: `exog[j][t]`.
    pub fn new(endog: Vec<f64>, exog: Option<Vec<Vec<f64>>>, config: SarimaxConfig) -> Result<Self> {
        crate::validation::validate_request(&endog, &config, exog.as_deref())?;
        let data = Prepared::new(&endog, exog.as_deref(), &config)?;
        Ok(Self { endog, exog, data })
    }

    pub fn config(&self) -> &SarimaxConfig {
        self.data.config()
    }

    /// Observations the filter sees: `n - d - s·D` under simple_differencing, else `n`.
    pub fn n_obs(&self) -> usize {
        self.data.endog().len()
    }

    fn params(&self, flat: &[f64]) -> Result<SarimaxParams> {
        SarimaxParams::from_flat(flat, self.config())
    }

    pub fn loglike(&self, params: &[f64]) -> Result<f64> {
        Ok(self.data.loglike(&self.params(params)?)?.loglike)
    }

    pub fn fit(
        &self,
        start_params: Option<&[f64]>,
        method: Option<&str>,
        maxiter: Option<u64>,
    ) -> Result<FitResult> {
        crate::optimizer::fit(
            &self.endog,
            self.config(),
            start_params,
            method,
            maxiter,
            self.exog.as_deref(),
        )
    }

    pub fn forecast(
        &self,
        params: &[f64],
        steps: usize,
        alpha: f64,
        future_exog: Option<&[Vec<f64>]>,
    ) -> Result<ForecastResult> {
        crate::forecast::forecast_pipeline(
            &self.endog,
            self.config(),
            &self.params(params)?,
            steps,
            alpha,
            self.exog.as_deref(),
            future_exog,
        )
    }

    pub fn rolling_forecast(
        &self,
        params: &[f64],
        start: usize,
        step: usize,
        horizon: usize,
        alpha: f64,
    ) -> Result<RollingForecastOutput> {
        crate::forecast::rolling_forecast_pipeline(
            &self.endog,
            self.config(),
            &self.params(params)?,
            start,
            step,
            horizon,
            alpha,
            self.exog.as_deref(),
        )
    }

    pub fn residuals(&self, params: &[f64]) -> Result<ResidualOutput> {
        let fo = self.data.filter(&self.params(params)?)?;
        Ok(crate::forecast::compute_residuals(&fo, self.config().concentrate_scale))
    }

    /// `method` is `"hessian"` or `"opg"`.
    pub fn inference(&self, params: &[f64], method: &str, alpha: f64) -> Result<InferenceResult> {
        self.params(params)?;
        crate::inference::compute_inference(
            &self.endog,
            self.config(),
            params,
            method,
            alpha,
            self.exog.as_deref(),
        )
    }

    pub fn diagnostics(&self, params: &[f64]) -> Result<DiagnosticsResult> {
        let resid = self.residuals(params)?;
        Ok(crate::inference::compute_diagnostics(&resid.standardized_residuals, params.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::load_fixtures;
    use crate::types::SarimaxOrder;

    fn sd_config(order: SarimaxOrder) -> SarimaxConfig {
        SarimaxConfig {
            order,
            n_exog: 1,
            simple_differencing: true,
            ..SarimaxConfig::default()
        }
    }

    #[test]
    fn prepared_differences_exog_with_y() {
        let cfg = sd_config(SarimaxOrder::new(0, 1, 0, 0, 0, 0, 0));
        let y = vec![1.0, 3.0, 6.0, 10.0];
        let x = vec![vec![2.0, 5.0, 9.0, 14.0]];
        let p = Prepared::new(&y, Some(&x), &cfg).unwrap();
        assert_eq!(p.endog(), &[2.0, 3.0, 4.0]);
        assert_eq!(p.exog().unwrap()[0], vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn prepared_seasonal_differences_exog_with_y() {
        let cfg = sd_config(SarimaxOrder::new(0, 0, 0, 0, 1, 0, 2));
        let y = vec![1.0, 2.0, 4.0, 7.0, 11.0];
        let x = vec![vec![3.0, 5.0, 10.0, 18.0, 31.0]];
        let p = Prepared::new(&y, Some(&x), &cfg).unwrap();
        assert_eq!(p.endog(), &[3.0, 5.0, 7.0]);
        assert_eq!(p.exog().unwrap()[0], vec![7.0, 13.0, 21.0]);
    }

    #[test]
    fn model_validates_once_and_checks_params() {
        let y: Vec<f64> = (0..30).map(|t| (t as f64 * 0.4).sin()).collect();
        let cfg = SarimaxConfig {
            order: SarimaxOrder::new(1, 0, 0, 0, 0, 0, 0),
            ..SarimaxConfig::default()
        };
        assert!(SarimaxModel::new(vec![], None, cfg.clone()).is_err());
        assert!(SarimaxModel::new(y.clone(), Some(vec![vec![1.0; 3]]), cfg.clone()).is_err());
        let m = SarimaxModel::new(y, None, cfg).unwrap();
        assert_eq!(m.n_obs(), 30);
        assert!(m.loglike(&[0.5]).is_err(), "wrong length must be rejected");
        assert!(m.inference(&[0.5], "hessian", 0.05).is_err());
        assert!(m.loglike(&[0.5, 1.0]).unwrap().is_finite());
    }

    #[test]
    fn forecast_validates_future_exog_before_differencing() {
        let y: Vec<f64> = (0..30).map(|t| (t as f64 * 0.4).sin()).collect();
        let x = vec![(0..30).map(|t| t as f64).collect()];
        let config = sd_config(SarimaxOrder::new(1, 1, 0, 0, 0, 0, 0));
        let model = SarimaxModel::new(y, Some(x), config).unwrap();
        let params = [0.3, 0.4, 1.0];
        for future in [
            vec![],
            vec![vec![30.0, 31.0], vec![1.0, 2.0]],
            vec![vec![30.0, 31.0], vec![f64::NAN, 2.0]],
            vec![vec![30.0]],
            vec![vec![30.0, f64::NAN]],
            vec![vec![30.0, f64::INFINITY]],
        ] {
            assert!(model.forecast(&params, 2, 0.05, Some(&future)).is_err());
        }
        assert!(model.forecast(&params, 2, 0.05, None).is_err());
        assert!(model.forecast(&params, 2, 0.05, Some(&[vec![30.0, 31.0]])).is_ok());
    }

    /// `simple_differencing=true` on (y, x) must equal the plain model on (Δy, Δx).
    /// One property that pins every path — loglike, inference, forecast, fit — to the same data.
    #[test]
    fn simple_differencing_matches_manual_differencing() {
        // Δy_t = 0.8·Δx_t + ARMA(1,1) noise (fixture arma11, n=200); y and x are cumulative sums.
        let noise: Vec<f64> = load_fixtures()["arma11"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let cumsum = |d: &[f64]| {
            let mut acc = 0.0;
            std::iter::once(0.0)
                .chain(d.iter().map(|v| {
                    acc += v;
                    acc
                }))
                .collect::<Vec<f64>>()
        };
        let diff = |v: &[f64]| v.windows(2).map(|w| w[1] - w[0]).collect::<Vec<f64>>();
        let x = cumsum(&(0..noise.len()).map(|t| (t as f64 * 0.3).sin()).collect::<Vec<_>>());
        let dx = diff(&x);
        let y = cumsum(&noise.iter().zip(&dx).map(|(e, x)| e + 0.8 * x).collect::<Vec<_>>());
        // Re-derive Δ from the levels so both models see bit-identical data
        // (cumsum then diff is not an exact round trip; the numerical Hessian amplifies 1e-16 by 1/h²).
        let dy = diff(&y);
        let sd = SarimaxConfig {
            order: SarimaxOrder::new(1, 1, 1, 0, 0, 0, 0),
            n_exog: 1,
            enforce_stationarity: true,
            enforce_invertibility: true,
            simple_differencing: true,
            ..SarimaxConfig::default()
        };
        let plain = SarimaxConfig {
            order: SarimaxOrder::new(1, 0, 1, 0, 0, 0, 0),
            simple_differencing: false,
            ..sd.clone()
        };
        let a = SarimaxModel::new(y.clone(), Some(vec![x.clone()]), sd).unwrap();
        let b = SarimaxModel::new(dy, Some(vec![dx]), plain).unwrap();
        let params = [0.8, 0.5, 0.3, 1.0];

        let (la, lb) = (a.loglike(&params).unwrap(), b.loglike(&params).unwrap());
        assert!((la - lb).abs() < 1e-10, "loglike {la} vs {lb}");

        for method in ["opg", "hessian"] {
            let ia = a.inference(&params, method, 0.05).unwrap();
            let ib = b.inference(&params, method, 0.05).unwrap();
            for (sa, sb) in ia.std_err.iter().zip(&ib.std_err) {
                assert!((sa - sb).abs() < 1e-8, "{method} std_err {sa} vs {sb}");
            }
        }

        let x_last = *x.last().unwrap();
        let xf = vec![x_last + 0.2, x_last + 0.1, x_last + 0.5];
        let dxf = diff(&[vec![x_last], xf.clone()].concat());
        let fa = a.forecast(&params, 3, 0.05, Some(&[xf])).unwrap();
        let fb = b.forecast(&params, 3, 0.05, Some(&[dxf])).unwrap();
        let (mut m, mut v) = (*y.last().unwrap(), 0.0);
        for h in 0..3 {
            m += fb.mean[h];
            v += fb.variance[h];
            assert!((fa.mean[h] - m).abs() < 1e-10, "mean[{h}] {} vs {m}", fa.mean[h]);
            assert!((fa.variance[h] - v).abs() < 1e-10, "var[{h}] {} vs {v}", fa.variance[h]);
        }

        let ra = a.fit(None, None, Some(500)).unwrap();
        let rb = b.fit(None, None, Some(500)).unwrap();
        assert!(
            (ra.loglike - rb.loglike).abs() < 1e-6,
            "fit loglike {} vs {}",
            ra.loglike,
            rb.loglike
        );
        for (pa, pb) in ra.params.iter().zip(&rb.params) {
            assert!((pa - pb).abs() < 1e-3, "fit params {pa} vs {pb}");
        }
    }
}
