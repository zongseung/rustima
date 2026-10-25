use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustima_core::model::SarimaxModel;
use rustima_core::types::{SarimaxConfig, SarimaxOrder, Trend};

use crate::outputs::to_pyerr;

/// Convert a 2D numpy array (n_obs × n_exog, row-major) to column-major Vec<Vec<f64>>.
pub(super) fn numpy2d_to_cols(arr: &PyReadonlyArray2<f64>) -> Vec<Vec<f64>> {
    let shape = arr.shape();
    let (n_obs, n_exog) = (shape[0], shape[1]);
    let data = arr.as_array();
    (0..n_exog)
        .map(|j| (0..n_obs).map(|t| data[[t, j]]).collect())
        .collect()
}

/// Model flags shared by every PyO3 entry point. A struct rather than
/// adjacent positional bools — transposing two bools compiles silently,
/// and the `false, false` literal smuggling this enabled caused the
/// fit-vs-residuals init mismatch (DIAGNOSIS_V9 S3/S13). Field-init
/// shorthand at call sites makes transposition impossible.
pub(super) struct ModelFlags {
    pub(super) enforce_stationarity: bool,
    pub(super) enforce_invertibility: bool,
    pub(super) concentrate_scale: bool,
    pub(super) simple_differencing: bool,
}

/// Config from Python-facing tuples. No validation here: `SarimaxModel::new` owns the rules.
pub(super) fn parse_config(
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    n_exog: usize,
    flags: &ModelFlags,
    trend: Option<&str>,
) -> PyResult<SarimaxConfig> {
    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;
    Ok(SarimaxConfig {
        order: SarimaxOrder::new(p, d, q, pp, dd, qq, s),
        n_exog,
        trend: Trend::from_str(trend.unwrap_or("n")).map_err(PyValueError::new_err)?,
        enforce_stationarity: flags.enforce_stationarity,
        enforce_invertibility: flags.enforce_invertibility,
        concentrate_scale: flags.concentrate_scale,
        simple_differencing: flags.simple_differencing,
    })
}

/// Column-major exog. The row count is checked here because a 0-column numpy
/// array has no rows the core could see.
pub(super) fn exog_cols(
    endog_len: usize,
    exog: Option<&PyReadonlyArray2<f64>>,
) -> PyResult<Option<Vec<Vec<f64>>>> {
    let Some(e) = exog else {
        return Ok(None);
    };
    let rows = e.shape()[0];
    if rows != endog_len {
        return Err(PyValueError::new_err(format!(
            "exog has {rows} rows but endog has {endog_len} observations; they must match"
        )));
    }
    Ok(Some(numpy2d_to_cols(e)))
}

/// Bind one series to its spec; every single-series entry point starts here.
pub(super) fn prepare_model(
    endog: &[f64],
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    exog: Option<&PyReadonlyArray2<f64>>,
    flags: &ModelFlags,
    trend: Option<&str>,
) -> PyResult<SarimaxModel> {
    let exog_cols = exog_cols(endog.len(), exog)?;
    let n_exog = exog_cols.as_ref().map_or(0, Vec::len);
    let config = parse_config(order, seasonal, n_exog, flags, trend)?;
    SarimaxModel::new(endog.to_vec(), exog_cols, config).map_err(to_pyerr)
}

/// Per-series exog columns for batch calls; the list must have one entry per series.
pub(super) fn parse_cols_list(
    list: &Option<Vec<PyReadonlyArray2<'_, f64>>>,
    n_series: usize,
    name: &str,
) -> PyResult<Option<Vec<Vec<Vec<f64>>>>> {
    let Some(el) = list else {
        return Ok(None);
    };
    if el.len() != n_series {
        return Err(PyValueError::new_err(format!(
            "{name} length ({}) must match series_list length ({n_series})",
            el.len()
        )));
    }
    Ok(Some(el.iter().map(numpy2d_to_cols).collect()))
}

/// Series + per-series exog + shared config (n_exog taken from the first exog).
/// Per-series validation happens in the core except for row counts lost by conversion.
pub(super) fn parse_batch_request(
    series_list: &[PyReadonlyArray1<'_, f64>],
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    exog_list: &Option<Vec<PyReadonlyArray2<'_, f64>>>,
    flags: &ModelFlags,
    trend: Option<&str>,
) -> PyResult<(Vec<Vec<f64>>, Option<Vec<Vec<Vec<f64>>>>, SarimaxConfig)> {
    let series = series_list
        .iter()
        .map(|s| Ok(s.as_slice()?.to_vec()))
        .collect::<PyResult<Vec<_>>>()?;
    if let Some(exogs) = exog_list {
        for (i, (exog, y)) in exogs.iter().zip(&series).enumerate() {
            // A zero-column array loses its row count when converted to columns.
            if exog.shape()[1] == 0 && exog.shape()[0] != y.len() {
                return Err(PyValueError::new_err(format!(
                    "exog_list[{i}] has {} rows but series[{i}] has {} observations; they must match",
                    exog.shape()[0], y.len()
                )));
            }
        }
    }
    let exog_vecs = parse_cols_list(exog_list, series.len(), "exog_list")?;
    let n_exog = exog_vecs.as_ref().and_then(|el| el.first()).map_or(0, Vec::len);
    let config = parse_config(order, seasonal, n_exog, flags, trend)?;
    Ok((series, exog_vecs, config))
}
