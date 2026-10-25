use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustima_core::{batch, optimizer, types::SarimaxConfig};

use crate::inputs::{exog_cols, parse_batch_request, parse_cols_list, parse_config, ModelFlags};
use crate::outputs::{fit_result_to_pydict, forecast_to_pydict, to_pyerr};

/// Compute log-likelihood for multiple time series in parallel (Rayon).
///
/// All series share the same model config and parameters.
/// Returns a list of dicts, each with "loglike" key or "error" key.
#[pyfunction]
#[pyo3(signature = (series_list, order, seasonal, params,
                    exog_list=None, concentrate_scale=false,
                    enforce_stationarity=false, enforce_invertibility=false,
                    trend=None, simple_differencing=false))]
pub(super) fn sarimax_batch_loglike<'py>(
    py: Python<'py>,
    series_list: Vec<PyReadonlyArray1<'py, f64>>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    exog_list: Option<Vec<PyReadonlyArray2<'py, f64>>>,
    concentrate_scale: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyList>> {
    let (series, exog_vecs, config) = parse_batch_request(
        &series_list,
        order,
        seasonal,
        &exog_list,
        &ModelFlags {
            enforce_stationarity,
            enforce_invertibility,
            concentrate_scale,
            simple_differencing,
        },
        trend,
    )?;
    let params_flat = params.as_slice()?.to_vec();

    // Release GIL for Rayon parallel computation
    let results =
        py.detach(|| batch::batch_loglike(&series, &config, &params_flat, exog_vecs.as_deref()));

    let mut py_results: Vec<Py<PyDict>> = Vec::with_capacity(results.len());
    for r in results {
        let dict = PyDict::new(py);
        match r {
            Ok(loglike) => {
                dict.set_item("loglike", loglike)?;
            }
            Err(e) => {
                dict.set_item("error", e.to_string())?;
            }
        }
        py_results.push(dict.into());
    }

    let list = PyList::new(py, &py_results)?;
    Ok(list.into())
}

/// Fit SARIMAX models to multiple time series in parallel (Rayon).
///
/// Returns a list of dicts (one per series), each with the same keys as sarimax_fit.
/// Failed series return a dict with "error" key instead.
#[pyfunction]
#[pyo3(signature = (series_list, order, seasonal,
                    enforce_stationarity=true, enforce_invertibility=true,
                    concentrate_scale=false, method=None, maxiter=None,
                    exog_list=None, trend=None, simple_differencing=false))]
pub(super) fn sarimax_batch_fit<'py>(
    py: Python<'py>,
    series_list: Vec<PyReadonlyArray1<'py, f64>>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    method: Option<&str>,
    maxiter: Option<u64>,
    exog_list: Option<Vec<PyReadonlyArray2<'py, f64>>>,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyList>> {
    method.map(optimizer::validate_method).transpose().map_err(to_pyerr)?;
    let (series, exog_vecs, config) = parse_batch_request(
        &series_list,
        order,
        seasonal,
        &exog_list,
        &ModelFlags {
            enforce_stationarity,
            enforce_invertibility,
            concentrate_scale,
            simple_differencing,
        },
        trend,
    )?;

    // Release GIL for Rayon parallel computation
    let method_owned = method.map(|s| s.to_string());
    let results = py.detach(|| {
        batch::batch_fit(
            &series,
            &config,
            method_owned.as_deref(),
            maxiter,
            exog_vecs.as_deref(),
        )
    });

    let mut py_results: Vec<Py<PyDict>> = Vec::with_capacity(results.len());
    for r in results {
        match r {
            Ok(result) => {
                let dict = fit_result_to_pydict(py, &result)?;
                py_results.push(dict.into());
            }
            Err(e) => {
                let dict = PyDict::new(py);
                dict.set_item("error", e.to_string())?;
                dict.set_item("converged", false)?;
                py_results.push(dict.into());
            }
        }
    }

    let list = PyList::new(py, &py_results)?;
    Ok(list.into())
}

/// Fit a single time series with multiple (order, seasonal_order) combinations
/// in parallel using Rayon.
///
/// Returns a list of dicts (one per combination), same structure as sarimax_fit.
/// Failed combinations have an "error" key instead of fit results.
#[pyfunction]
#[pyo3(signature = (y, order_list, seasonal_list,
                    enforce_stationarity=true, enforce_invertibility=true,
                    concentrate_scale=false, method=None, maxiter=None,
                    exog=None, trend=None, simple_differencing=false))]
pub(super) fn sarimax_grid_search<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order_list: Vec<(usize, usize, usize)>,
    seasonal_list: Vec<(usize, usize, usize, usize)>,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    method: Option<&str>,
    maxiter: Option<u64>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyList>> {
    method.map(optimizer::validate_method).transpose().map_err(to_pyerr)?;
    if order_list.len() != seasonal_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "order_list length ({}) must match seasonal_list length ({})",
            order_list.len(),
            seasonal_list.len()
        )));
    }

    let endog = y.as_slice()?;
    let exog_cols = exog_cols(endog.len(), exog.as_ref())?;
    let n_exog = exog_cols.as_ref().map_or(0, Vec::len);
    let flags = ModelFlags {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        simple_differencing,
    };
    // Each candidate is validated by the core when it is bound, so a bad order is one error entry.
    let configs = order_list
        .iter()
        .zip(&seasonal_list)
        .map(|(o, s)| parse_config(*o, *s, n_exog, &flags, trend))
        .collect::<PyResult<Vec<SarimaxConfig>>>()?;

    // Release GIL for Rayon parallel computation
    let endog_owned = endog.to_vec();
    let method_owned = method.map(|s| s.to_string());
    let results = py.detach(|| {
        batch::grid_search_fit(
            &endog_owned,
            &configs,
            method_owned.as_deref(),
            maxiter,
            exog_cols.as_deref(),
        )
    });

    let mut py_results: Vec<Py<PyDict>> = Vec::with_capacity(results.len());
    for (idx, r) in results.into_iter().enumerate() {
        match r {
            Ok(result) => {
                let dict = fit_result_to_pydict(py, &result)?;
                dict.set_item("order", order_list[idx])?;
                dict.set_item("seasonal_order", seasonal_list[idx])?;
                py_results.push(dict.into());
            }
            Err(e) => {
                let dict = PyDict::new(py);
                dict.set_item("order", order_list[idx])?;
                dict.set_item("seasonal_order", seasonal_list[idx])?;
                dict.set_item("error", e.to_string())?;
                dict.set_item("converged", false)?;
                py_results.push(dict.into());
            }
        }
    }

    let list = PyList::new(py, &py_results)?;
    Ok(list.into())
}

/// Forecast multiple time series in parallel (Rayon).
///
/// Each series uses its own parameter vector from params_list.
/// Returns a list of dicts (one per series).
#[pyfunction]
#[pyo3(signature = (series_list, order, seasonal, params_list,
                    steps=10, alpha=0.05, concentrate_scale=false,
                    exog_list=None, exog_forecast_list=None, trend=None,
                    simple_differencing=false,
                    enforce_stationarity=true, enforce_invertibility=true))]
pub(super) fn sarimax_batch_forecast<'py>(
    py: Python<'py>,
    series_list: Vec<PyReadonlyArray1<'py, f64>>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params_list: Vec<PyReadonlyArray1<'py, f64>>,
    steps: usize,
    alpha: f64,
    concentrate_scale: bool,
    exog_list: Option<Vec<PyReadonlyArray2<'py, f64>>>,
    exog_forecast_list: Option<Vec<PyReadonlyArray2<'py, f64>>>,
    trend: Option<&str>,
    simple_differencing: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
) -> PyResult<Py<PyList>> {
    if series_list.len() != params_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "series_list and params_list must have same length: {} vs {}",
            series_list.len(),
            params_list.len()
        )));
    }
    // Call-level arguments: reject once instead of once per series.
    rustima_core::validation::validate_alpha(alpha).map_err(to_pyerr)?;
    rustima_core::validation::validate_steps(steps).map_err(to_pyerr)?;
    let (series, exog_vecs, config) = parse_batch_request(
        &series_list,
        order,
        seasonal,
        &exog_list,
        &ModelFlags {
            enforce_stationarity,
            enforce_invertibility,
            concentrate_scale,
            simple_differencing,
        },
        trend,
    )?;
    let future_exog_vecs =
        parse_cols_list(&exog_forecast_list, series_list.len(), "exog_forecast_list")?;

    let params_vecs: Vec<Vec<f64>> = params_list
        .iter()
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    // Release GIL for Rayon parallel computation
    let results = py.detach(|| {
        batch::batch_forecast(
            &series,
            &config,
            &params_vecs,
            steps,
            alpha,
            exog_vecs.as_deref(),
            future_exog_vecs.as_deref(),
        )
    });

    let mut py_results: Vec<Py<PyDict>> = Vec::with_capacity(results.len());
    for r in results {
        match r {
            Ok(result) => {
                py_results.push(forecast_to_pydict(py, &result)?.into());
            }
            Err(e) => {
                let dict = PyDict::new(py);
                dict.set_item("error", e.to_string())?;
                py_results.push(dict.into());
            }
        }
    }

    let list = PyList::new(py, &py_results)?;
    Ok(list.into())
}
