mod auto;
mod batch;
mod inputs;
mod outputs;
mod single;

use pyo3::prelude::*;

/// Smoke-test function: returns the version string.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python module definition.
#[pymodule]
fn rustima(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(auto::_auto_arima, m)?)?;
    m.add_function(wrap_pyfunction!(auto::_difference_orders, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(single::_model_info, m)?)?;
    m.add_function(wrap_pyfunction!(single::sarimax_loglike, m)?)?;
    m.add_function(wrap_pyfunction!(single::sarimax_fit, m)?)?;
    m.add_function(wrap_pyfunction!(single::sarimax_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(single::sarimax_rolling_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(single::sarimax_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(batch::sarimax_batch_loglike, m)?)?;
    m.add_function(wrap_pyfunction!(batch::sarimax_batch_fit, m)?)?;
    m.add_function(wrap_pyfunction!(batch::sarimax_grid_search, m)?)?;
    m.add_function(wrap_pyfunction!(batch::sarimax_batch_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(single::sarimax_inference, m)?)?;
    m.add_function(wrap_pyfunction!(single::sarimax_diagnostics, m)?)?;
    Ok(())
}
