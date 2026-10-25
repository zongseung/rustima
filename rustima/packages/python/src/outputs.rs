use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Convert a FitResult to a PyDict with standard keys.
///
/// Warnings from the Rust fit path are emitted via Python's `warnings.warn()`
/// and also stored in the `"warnings"` key (V8.5 P2-1).
pub(super) fn fit_result_to_pydict<'py>(
    py: Python<'py>,
    result: &rustima_core::types::FitResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("params", &result.params)?;
    dict.set_item("loglike", result.loglike)?;
    dict.set_item("scale", result.scale)?;
    dict.set_item("aic", result.aic)?;
    dict.set_item("bic", result.bic)?;
    dict.set_item("n_obs", result.n_obs)?;
    dict.set_item("n_params", result.n_params)?;
    dict.set_item("n_iter", result.n_iter)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("method", &result.method)?;
    dict.set_item("warnings", &result.warnings)?;

    // Emit warnings via Python warnings module
    for msg in &result.warnings {
        let warn_mod = py.import("warnings")?;
        warn_mod.call_method1("warn", (msg.as_str(),))?;
    }

    Ok(dict)
}

/// Convert a ForecastResult to a PyDict.
pub(super) fn forecast_to_pydict<'py>(
    py: Python<'py>,
    result: &rustima_core::forecast::ForecastResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("mean", &result.mean)?;
    dict.set_item("variance", &result.variance)?;
    dict.set_item("ci_lower", &result.ci_lower)?;
    dict.set_item("ci_upper", &result.ci_upper)?;
    Ok(dict)
}

/// Convert a ResidualOutput to a PyDict.
pub(super) fn residuals_to_pydict<'py>(
    py: Python<'py>,
    result: &rustima_core::forecast::ResidualOutput,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("residuals", &result.residuals)?;
    dict.set_item("standardized_residuals", &result.standardized_residuals)?;
    dict.set_item("prediction_variances", &result.prediction_variances)?;
    Ok(dict)
}

/// Convert an InferenceResult to a PyDict.
pub(super) fn inference_to_pydict<'py>(
    py: Python<'py>,
    result: &rustima_core::inference::InferenceResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("method", &result.method)?;
    dict.set_item("std_err", &result.std_err)?;
    dict.set_item("z_stat", &result.z_stat)?;
    dict.set_item("p_value", &result.p_value)?;
    dict.set_item("ci_lower", &result.ci_lower)?;
    dict.set_item("ci_upper", &result.ci_upper)?;
    dict.set_item("cov_params", &result.cov_params)?;
    dict.set_item("n_params", result.n_params)?;
    dict.set_item("status", &result.status)?;
    dict.set_item("message", &result.message)?;
    Ok(dict)
}

/// Convert a DiagnosticsResult to a PyDict.
pub(super) fn diagnostics_to_pydict<'py>(
    py: Python<'py>,
    result: &rustima_core::inference::DiagnosticsResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("ljung_box_stat", result.ljung_box_stat)?;
    dict.set_item("ljung_box_pvalue", result.ljung_box_pvalue)?;
    dict.set_item("ljung_box_df", result.ljung_box_df)?;
    dict.set_item("jarque_bera_stat", result.jarque_bera_stat)?;
    dict.set_item("jarque_bera_pvalue", result.jarque_bera_pvalue)?;
    dict.set_item("het_stat", result.het_stat)?;
    dict.set_item("het_pvalue", result.het_pvalue)?;
    Ok(dict)
}

/// Map internal SarimaxError to appropriate Python exception type.
pub(super) fn to_pyerr(e: rustima_core::error::SarimaxError) -> PyErr {
    use rustima_core::error::SarimaxError;
    match &e {
        SarimaxError::OptimizationFailed(_) | SarimaxError::CholeskyFailed => {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        }
        _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
    }
}
