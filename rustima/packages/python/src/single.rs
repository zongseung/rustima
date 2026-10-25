use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustima_core::{forecast, inference, optimizer, params::SarimaxParams, pipeline};

use crate::inputs::{
    numpy2d_to_cols, prepare_single_request, validate_alpha, validate_finite_cols, validate_steps,
    ModelFlags,
};
use crate::outputs::{
    diagnostics_to_pydict, fit_result_to_pydict, forecast_to_pydict, inference_to_pydict,
    residuals_to_pydict, to_pyerr,
};

/// Compute the SARIMAX concentrated (or full) log-likelihood.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, exog=None, concentrate_scale=false,
                    enforce_stationarity=true, enforce_invertibility=true,
                    trend=None, simple_differencing=false))]
pub(super) fn sarimax_loglike<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<f64> {
    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;
    let (config, exog_cols) = prepare_single_request(
        endog,
        order,
        seasonal,
        exog.as_ref(),
        &ModelFlags {
            enforce_stationarity,
            enforce_invertibility,
            concentrate_scale,
            simple_differencing,
        },
        trend,
    )?;

    // Own all data before releasing GIL
    let endog = endog.to_vec();
    let params_flat = params_flat.to_vec();

    py.detach(move || {
        let sarimax_params = SarimaxParams::from_flat(&params_flat, &config)?;
        pipeline::kalman_eval(&endog, &sarimax_params, &config, exog_cols.as_deref())
            .map(|o| o.loglike)
    })
    .map_err(to_pyerr)
}

/// Fit a SARIMAX model via MLE (L-BFGS-B default, with Nelder-Mead fallback).
///
/// Returns a dict with: params, loglike, scale, aic, bic, n_obs, n_iter, converged, method.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, start_params=None, exog=None,
                    concentrate_scale=false, enforce_stationarity=true,
                    enforce_invertibility=true, method=None, maxiter=None,
                    trend=None, simple_differencing=false))]
pub(super) fn sarimax_fit<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    start_params: Option<PyReadonlyArray1<'py, f64>>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    method: Option<&str>,
    maxiter: Option<u64>,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyDict>> {
    let endog = y.as_slice()?;
    let (config, exog_cols) = prepare_single_request(
        endog,
        order,
        seasonal,
        exog.as_ref(),
        &ModelFlags {
            enforce_stationarity,
            enforce_invertibility,
            concentrate_scale,
            simple_differencing,
        },
        trend,
    )?;

    let sp_owned: Option<Vec<f64>> = start_params
        .as_ref()
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let method_owned = method.map(|s| s.to_string());

    // Own all data before releasing GIL
    let endog = endog.to_vec();

    let result = py
        .detach(move || {
            optimizer::fit(
                &endog,
                &config,
                sp_owned.as_deref(),
                method_owned.as_deref(),
                maxiter,
                exog_cols.as_deref(),
            )
        })
        .map_err(to_pyerr)?;

    let dict = fit_result_to_pydict(py, &result)?;

    Ok(dict.into())
}

/// Compute h-step ahead forecast for a SARIMAX model.
///
/// Returns a dict with: mean, variance, ci_lower, ci_upper.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, steps=10, alpha=0.05,
                    exog=None, future_exog=None, concentrate_scale=false,
                    trend=None, simple_differencing=false,
                    enforce_stationarity=true, enforce_invertibility=true))]
pub(super) fn sarimax_forecast<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    steps: usize,
    alpha: f64,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    future_exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    trend: Option<&str>,
    simple_differencing: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
) -> PyResult<Py<PyDict>> {
    validate_alpha(alpha)?;
    validate_steps(steps)?;

    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;
    let (config, exog_cols) = prepare_single_request(
        endog,
        order,
        seasonal,
        exog.as_ref(),
        &ModelFlags {
            enforce_stationarity,
            enforce_invertibility,
            concentrate_scale,
            simple_differencing,
        },
        trend,
    )?;

    let future_exog_cols = future_exog.as_ref().map(|e| numpy2d_to_cols(e));
    if let Some(ref cols) = future_exog_cols {
        validate_finite_cols(cols, "future_exog")?;
    }

    let n_exog = config.n_exog;
    if let Some(ref fec) = future_exog_cols {
        if fec.len() != n_exog {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "future_exog column count ({}) must match exog column count ({})",
                fec.len(),
                n_exog
            )));
        }
    }

    // Validate future_exog has enough rows for forecast steps
    if let Some(ref fec) = future_exog_cols {
        for (j, col) in fec.iter().enumerate() {
            if col.len() < steps {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "future_exog column {} has {} rows but {} forecast steps requested",
                    j,
                    col.len(),
                    steps
                )));
            }
        }
    }

    // Require future exog for exog models
    if n_exog > 0 && steps > 0 && future_exog.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "model has exogenous variables: future_exog is required for forecasting",
        ));
    }

    // Own all data before releasing GIL
    let endog = endog.to_vec();
    let params_flat = params_flat.to_vec();

    let result = py
        .detach(move || {
            let sarimax_params = SarimaxParams::from_flat(&params_flat, &config)?;
            forecast::forecast_pipeline(
                &endog,
                &config,
                &sarimax_params,
                steps,
                alpha,
                exog_cols.as_deref(),
                future_exog_cols.as_deref(),
            )
        })
        .map_err(to_pyerr)?;

    Ok(forecast_to_pydict(py, &result)?.into())
}

/// Single-pass rolling-origin h-step forecasts with fixed parameters.
///
/// One Kalman filter pass over the full series; forecasts are propagated
/// from the predicted-state snapshot at each origin `start, start+step, ...`.
/// O(T + N*horizon) total. Returns a dict with: origins, mean, variance,
/// ci_lower, ci_upper (row-per-origin nested lists).
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, start, step=1, horizon=1, alpha=0.05,
                    exog=None, concentrate_scale=false, trend=None, simple_differencing=false,
                    enforce_stationarity=true, enforce_invertibility=true))]
#[allow(clippy::too_many_arguments)]
pub(super) fn sarimax_rolling_forecast<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    start: usize,
    step: usize,
    horizon: usize,
    alpha: f64,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    trend: Option<&str>,
    simple_differencing: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
) -> PyResult<Py<PyDict>> {
    validate_alpha(alpha)?;
    if horizon > 0 {
        validate_steps(horizon)?;
    }

    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;
    let (config, exog_cols) = prepare_single_request(
        endog,
        order,
        seasonal,
        exog.as_ref(),
        &ModelFlags {
            enforce_stationarity,
            enforce_invertibility,
            concentrate_scale,
            simple_differencing,
        },
        trend,
    )?;

    // Own all data before releasing GIL
    let endog = endog.to_vec();
    let params_flat = params_flat.to_vec();

    let result = py
        .detach(move || {
            let sarimax_params = SarimaxParams::from_flat(&params_flat, &config)?;
            forecast::rolling_forecast_pipeline(
                &endog,
                &config,
                &sarimax_params,
                start,
                step,
                horizon,
                alpha,
                exog_cols.as_deref(),
            )
        })
        .map_err(to_pyerr)?;

    let dict = PyDict::new(py);
    dict.set_item(
        "origins",
        result
            .origins
            .iter()
            .map(|&o| o as u64)
            .collect::<Vec<u64>>(),
    )?;
    dict.set_item("mean", result.mean)?;
    dict.set_item("variance", result.variance)?;
    dict.set_item("ci_lower", result.ci_lower)?;
    dict.set_item("ci_upper", result.ci_upper)?;
    Ok(dict.into())
}

/// Compute residuals and standardized residuals for a SARIMAX model.
///
/// Returns a dict with: residuals, standardized_residuals, prediction_variances.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, exog=None, concentrate_scale=false, trend=None, simple_differencing=false,
                    enforce_stationarity=true, enforce_invertibility=true))]
pub(super) fn sarimax_residuals<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    trend: Option<&str>,
    simple_differencing: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
) -> PyResult<Py<PyDict>> {
    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;
    let (config, exog_cols) = prepare_single_request(
        endog,
        order,
        seasonal,
        exog.as_ref(),
        &ModelFlags {
            enforce_stationarity,
            enforce_invertibility,
            concentrate_scale,
            simple_differencing,
        },
        trend,
    )?;

    // Own all data before releasing GIL
    let endog = endog.to_vec();
    let params_flat = params_flat.to_vec();

    let result = py
        .detach(move || {
            let sarimax_params = SarimaxParams::from_flat(&params_flat, &config)?;
            forecast::residuals_pipeline(&endog, &config, &sarimax_params, exog_cols.as_deref())
        })
        .map_err(to_pyerr)?;

    Ok(residuals_to_pydict(py, &result)?.into())
}

/// Compute inference statistics (standard errors, z, p-values, CI) for fitted parameters.
///
/// Returns a dict with: method, std_err, z_stat, p_value, ci_lower, ci_upper,
/// cov_params, n_params, status, message.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, method="hessian", alpha=0.05,
                    exog=None, concentrate_scale=false,
                    enforce_stationarity=true, enforce_invertibility=true,
                    trend=None, simple_differencing=false))]
pub(super) fn sarimax_inference<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    method: &str,
    alpha: f64,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyDict>> {
    validate_alpha(alpha)?;

    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;
    let (config, exog_cols) = prepare_single_request(
        endog,
        order,
        seasonal,
        exog.as_ref(),
        &ModelFlags {
            enforce_stationarity,
            enforce_invertibility,
            concentrate_scale,
            simple_differencing,
        },
        trend,
    )?;

    // Reject wrong-length params up front like every other entry point —
    // compute_inference would silently return empty arrays for an empty
    // params vector (DIAGNOSIS_V9 W2).
    SarimaxParams::from_flat(params_flat, &config).map_err(to_pyerr)?;

    // Own all data before releasing GIL
    let endog = endog.to_vec();
    let params_flat = params_flat.to_vec();
    let method_owned = method.to_string();

    let result = py
        .detach(move || {
            inference::compute_inference(
                &endog,
                &config,
                &params_flat,
                &method_owned,
                alpha,
                exog_cols.as_deref(),
            )
        })
        .map_err(to_pyerr)?;

    Ok(inference_to_pydict(py, &result)?.into())
}

/// Compute residual diagnostic tests (Ljung-Box, Jarque-Bera, heteroskedasticity).
///
/// Returns a dict with: ljung_box_stat, ljung_box_pvalue, ljung_box_df,
/// jarque_bera_stat, jarque_bera_pvalue, het_stat, het_pvalue.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, exog=None, concentrate_scale=false, trend=None, simple_differencing=false,
                    enforce_stationarity=true, enforce_invertibility=true))]
pub(super) fn sarimax_diagnostics<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    trend: Option<&str>,
    simple_differencing: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
) -> PyResult<Py<PyDict>> {
    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;
    let (config, exog_cols) = prepare_single_request(
        endog,
        order,
        seasonal,
        exog.as_ref(),
        &ModelFlags {
            enforce_stationarity,
            enforce_invertibility,
            concentrate_scale,
            simple_differencing,
        },
        trend,
    )?;

    // Own all data before releasing GIL
    let endog = endog.to_vec();
    let params_flat = params_flat.to_vec();

    let result = py
        .detach(
            move || -> std::result::Result<_, rustima_core::error::SarimaxError> {
                let sarimax_params = SarimaxParams::from_flat(&params_flat, &config)?;
                let resid_result = forecast::residuals_pipeline(
                    &endog,
                    &config,
                    &sarimax_params,
                    exog_cols.as_deref(),
                )?;
                let n_params = params_flat.len();
                Ok(inference::compute_diagnostics(
                    &resid_result.standardized_residuals,
                    n_params,
                ))
            },
        )
        .map_err(to_pyerr)?;

    Ok(diagnostics_to_pydict(py, &result)?.into())
}
