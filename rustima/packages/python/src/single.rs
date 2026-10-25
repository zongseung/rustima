use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::inputs::{numpy2d_to_cols, prepare_model, ModelFlags};
use crate::outputs::{
    diagnostics_to_pydict, fit_result_to_pydict, forecast_to_pydict, inference_to_pydict,
    residuals_to_pydict, to_pyerr,
};

/// Validate a specification at construction time and report its shape:
/// `n_obs` (observations the filter sees) and `param_names`.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, exog=None, concentrate_scale=false,
                    enforce_stationarity=true, enforce_invertibility=true,
                    trend=None, simple_differencing=false))]
pub(super) fn _model_info<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyDict>> {
    let model = prepare_model(
        y.as_slice()?,
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
    let dict = PyDict::new(py);
    dict.set_item("n_obs", model.n_obs())?;
    dict.set_item("param_names", model.config().param_names())?;
    Ok(dict.into())
}

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
    let model = prepare_model(
        y.as_slice()?,
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
    let params = params.as_slice()?.to_vec();
    py.detach(move || model.loglike(&params)).map_err(to_pyerr)
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
    let model = prepare_model(
        y.as_slice()?,
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
    let start: Option<Vec<f64>> = start_params
        .as_ref()
        .map(|a| a.as_slice().map(<[_]>::to_vec))
        .transpose()?;
    let method = method.map(str::to_owned);
    let result = py
        .detach(move || model.fit(start.as_deref(), method.as_deref(), maxiter))
        .map_err(to_pyerr)?;
    Ok(fit_result_to_pydict(py, &result)?.into())
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
    let model = prepare_model(
        y.as_slice()?,
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
    let params = params.as_slice()?.to_vec();
    let future = future_exog.as_ref().map(numpy2d_to_cols);
    let result = py
        .detach(move || model.forecast(&params, steps, alpha, future.as_deref()))
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
    let model = prepare_model(
        y.as_slice()?,
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
    let params = params.as_slice()?.to_vec();
    let result = py
        .detach(move || model.rolling_forecast(&params, start, step, horizon, alpha))
        .map_err(to_pyerr)?;
    let dict = PyDict::new(py);
    dict.set_item(
        "origins",
        result.origins.iter().map(|&o| o as u64).collect::<Vec<u64>>(),
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
    let model = prepare_model(
        y.as_slice()?,
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
    let params = params.as_slice()?.to_vec();
    let result = py.detach(move || model.residuals(&params)).map_err(to_pyerr)?;
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
    let model = prepare_model(
        y.as_slice()?,
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
    let params = params.as_slice()?.to_vec();
    let method = method.to_owned();
    let result = py
        .detach(move || model.inference(&params, &method, alpha))
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
    let model = prepare_model(
        y.as_slice()?,
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
    let params = params.as_slice()?.to_vec();
    let result = py.detach(move || model.diagnostics(&params)).map_err(to_pyerr)?;
    Ok(diagnostics_to_pydict(py, &result)?.into())
}
