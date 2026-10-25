use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustima_core::auto::{self, Criterion, SearchOptions};

use crate::inputs::{prepare_single_request, ModelFlags};
use crate::outputs::{fit_result_dict, to_pyerr};

#[pyfunction]
pub(super) fn _difference_orders(
    y: PyReadonlyArray1<'_, f64>,
    max_d: usize,
    max_dd: usize,
    period: usize,
) -> PyResult<(usize, usize)> {
    auto::difference_orders(y.as_slice()?, max_d, max_dd, period).map_err(to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (y, max_order=(5, 2, 5), max_seasonal=(2, 1, 2, 0),
    d=None, seasonal_d=None, exog=None, trend=None, criterion="aic", stepwise=true,
    enforce_stationarity=true, enforce_invertibility=true, simple_differencing=false,
    method=None, maxiter=None))]
pub(super) fn _auto_arima<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    max_order: (usize, usize, usize),
    max_seasonal: (usize, usize, usize, usize),
    d: Option<usize>,
    seasonal_d: Option<usize>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    trend: Option<&str>,
    criterion: &str,
    stepwise: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    simple_differencing: bool,
    method: Option<&str>,
    maxiter: Option<u64>,
) -> PyResult<Py<PyDict>> {
    let y = y.as_slice()?;
    let (config, x) = prepare_single_request(
        y,
        (0, 0, 0),
        (0, 0, 0, 0),
        exog.as_ref(),
        &ModelFlags {
            enforce_stationarity,
            enforce_invertibility,
            simple_differencing,
            concentrate_scale: false,
        },
        trend,
    )?;
    let options = SearchOptions {
        max_p: max_order.0,
        max_d: max_order.1,
        max_q: max_order.2,
        max_pp: max_seasonal.0,
        max_dd: max_seasonal.1,
        max_qq: max_seasonal.2,
        period: max_seasonal.3,
        d,
        dd: seasonal_d,
        stepwise,
        criterion: Criterion::parse(criterion).map_err(to_pyerr)?,
    };
    let y = y.to_vec();
    let method = method.map(str::to_owned);
    let result = py
        .detach(move || {
            auto::search(
                &y,
                &config,
                &options,
                method.as_deref(),
                maxiter,
                x.as_deref(),
            )
        })
        .map_err(to_pyerr)?;
    let out = PyDict::new(py);
    if let Some(best) = result.best {
        let fit = fit_result_dict(py, &best.fit)?;
        let o = best.order;
        fit.set_item("order", (o.p, o.d, o.q))?;
        fit.set_item("seasonal_order", (o.pp, o.dd, o.qq, o.s))?;
        out.set_item("fit", fit)?;
    } else {
        out.set_item("fit", py.None())?;
    }
    let history = PyList::empty(py);
    for item in result.history {
        let entry = PyDict::new(py);
        let o = item.order;
        entry.set_item("order", (o.p, o.d, o.q))?;
        entry.set_item("seasonal_order", (o.pp, o.dd, o.qq, o.s))?;
        entry.set_item(criterion, item.ic)?;
        entry.set_item("converged", item.converged)?;
        if let Some(error) = item.error {
            entry.set_item("error_type", "RuntimeError")?;
            entry.set_item("error_message", error)?;
        }
        for warning in item.warnings {
            py.import("warnings")?.call_method1("warn", (warning,))?;
        }
        history.append(entry)?;
    }
    out.set_item("history", history)?;
    out.set_item("d", result.d)?;
    out.set_item("D", result.dd)?;
    Ok(out.into())
}
