use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
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

/// Parse optional exog numpy array → (column-major vecs, n_exog, n_rows).
///
/// Returns `n_rows` from the original numpy shape so that row-count validation
/// works even for 0-column exog arrays (shape `(m, 0)`).
fn parse_exog(exog: Option<&PyReadonlyArray2<f64>>) -> (Option<Vec<Vec<f64>>>, usize, usize) {
    match exog {
        Some(e) => {
            let n_rows = e.shape()[0];
            let cols = numpy2d_to_cols(e);
            let n = cols.len();
            (Some(cols), n, n_rows)
        }
        None => (None, 0, 0),
    }
}

pub(super) fn validate_finite_cols(cols: &[Vec<f64>], input_name: &str) -> PyResult<()> {
    for (j, col) in cols.iter().enumerate() {
        rustima_core::validation::validate_finite(col, input_name)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e} (column {j})")))?;
    }
    Ok(())
}

/// Validate that a 1D endog slice contains no NaN/Inf values.
fn validate_endog_finite(endog: &[f64]) -> PyResult<()> {
    rustima_core::validation::validate_endog(endog).map_err(to_pyerr)
}

/// Validate that each series in a batch contains no NaN/Inf values.
fn validate_batch_finite(series_list: &[PyReadonlyArray1<'_, f64>]) -> PyResult<Vec<Vec<f64>>> {
    let mut result = Vec::with_capacity(series_list.len());
    for (i, s) in series_list.iter().enumerate() {
        let sl = s.as_slice()?;
        rustima_core::validation::validate_finite(sl, "input time series").map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("series at index {i}: {e}"))
        })?;
        result.push(sl.to_vec());
    }
    Ok(result)
}

/// Parse and validate a batch exog list: matching lengths, finite values, consistent column counts,
/// and per-series row count matching endog length.
///
/// Converts numpy arrays to column-major Vecs and validates:
/// - `exog_list` length matches `series_len`
/// - All values are finite
/// - All series have the same number of exog columns
/// - Each exog's row count matches the corresponding series length
fn parse_and_validate_batch_exog(
    exog_list: &Option<Vec<PyReadonlyArray2<'_, f64>>>,
    series_len: usize,
    series_lengths: &[usize],
) -> PyResult<(Option<Vec<Vec<Vec<f64>>>>, usize)> {
    if let Some(ref el) = exog_list {
        if el.len() != series_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "exog_list length ({}) must match series_list length ({})",
                el.len(),
                series_len
            )));
        }
        let exog_vecs: Vec<Vec<Vec<f64>>> = el.iter().map(|e| numpy2d_to_cols(e)).collect();
        let n_exog = exog_vecs.first().map_or(0, |c| c.len());
        for (i, cols) in exog_vecs.iter().enumerate() {
            validate_finite_cols(cols, &format!("exog_list[{}]", i))?;
            if cols.len() != n_exog {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "exog at index {} has {} columns but expected {} (from first series)",
                    i,
                    cols.len(),
                    n_exog
                )));
            }
            // Per-series row count validation (shape-based via numpy shape[0])
            let exog_nrows = el[i].shape()[0];
            if exog_nrows != series_lengths[i] {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "exog_list[{}] has {} rows but series[{}] has {} observations; they must match",
                    i, exog_nrows, i, series_lengths[i]
                )));
            }
        }
        Ok((Some(exog_vecs), n_exog))
    } else {
        Ok((None, 0))
    }
}

/// Parse and validate a batch forecast exog list (future_exog for batch forecasting).
///
/// Validates:
/// - Length matches `series_len`
/// - All values are finite
/// - Column counts match `n_exog`
/// - Each column has at least `steps` rows
pub(super) fn parse_and_validate_batch_forecast_exog(
    exog_forecast_list: &Option<Vec<PyReadonlyArray2<'_, f64>>>,
    series_len: usize,
    n_exog: usize,
    steps: usize,
) -> PyResult<Option<Vec<Vec<Vec<f64>>>>> {
    match exog_forecast_list {
        Some(ref el) => {
            if el.len() != series_len {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "exog_forecast_list length ({}) must match series_list length ({})",
                    el.len(),
                    series_len
                )));
            }
            let vecs: Vec<Vec<Vec<f64>>> = el.iter().map(|e| numpy2d_to_cols(e)).collect();
            for (i, cols) in vecs.iter().enumerate() {
                validate_finite_cols(cols, &format!("exog_forecast_list[{}]", i))?;
                if cols.len() != n_exog {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "exog_forecast at index {} has {} columns but expected {}",
                        i,
                        cols.len(),
                        n_exog
                    )));
                }
                for (j, col) in cols.iter().enumerate() {
                    if col.len() < steps {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "exog_forecast_list[{}] column {} has {} rows but {} forecast steps requested",
                            i, j, col.len(), steps
                        )));
                    }
                }
            }
            Ok(Some(vecs))
        }
        None => Ok(None),
    }
}

/// Validate model order bounds and check for arithmetic overflow.
///
/// Consolidates all bounds checking (p, q, d, P, D, Q, s, n_exog) and
/// overflow checking (checked_add/checked_mul chains) into one place.
pub(super) fn validate_order(
    p: usize,
    d: usize,
    q: usize,
    pp: usize,
    dd: usize,
    qq: usize,
    s: usize,
    n_exog: usize,
) -> PyResult<()> {
    rustima_core::validation::validate_order(p, d, q, pp, dd, qq, s, n_exog).map_err(to_pyerr)
}

/// Validate single-series exog columns: finite values and consistent length.
fn validate_exog(exog_cols: &Option<Vec<Vec<f64>>>) -> PyResult<()> {
    if let Some(ref cols) = exog_cols {
        validate_finite_cols(cols, "exog")?;
    }
    Ok(())
}

/// Validate batch exog list: matching lengths, finite values, consistent column counts,
/// and per-series row count matching endog length.
fn validate_batch_exog(
    exog_list: &Option<Vec<PyReadonlyArray2<'_, f64>>>,
    series_len: usize,
    series_lengths: &[usize],
) -> PyResult<(Option<Vec<Vec<Vec<f64>>>>, usize)> {
    parse_and_validate_batch_exog(exog_list, series_len, series_lengths)
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

/// Validate endog + exog for a single-series call: finiteness, exog shape,
/// and row-count agreement. Returns `(exog_cols, n_exog)`.
///
/// Shared by `prepare_single_request` and `sarimax_grid_search` so a new
/// boundary check lands on every entry point at once (DIAGNOSIS_V9 S12).
pub(super) fn validate_single_inputs(
    endog: &[f64],
    exog: Option<&PyReadonlyArray2<f64>>,
) -> PyResult<(Option<Vec<Vec<f64>>>, usize)> {
    validate_endog_finite(endog)?;
    let (exog_cols, n_exog, exog_nrows) = parse_exog(exog);
    validate_exog(&exog_cols)?;
    // Early exog row-count validation at PyO3 boundary (shape-based, works for 0-col exog too)
    if exog_cols.is_some() && exog_nrows != endog.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "exog has {} rows but endog has {} observations; they must match",
            exog_nrows,
            endog.len()
        )));
    }
    Ok((exog_cols, n_exog))
}

/// Build SarimaxConfig from Python-facing tuples and flags.
///
/// Assumes `validate_order()` has already been called to verify bounds
/// and overflow safety.
pub(super) fn build_config(
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    n_exog: usize,
    flags: &ModelFlags,
    trend: Trend,
) -> SarimaxConfig {
    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;

    SarimaxConfig {
        order: SarimaxOrder::new(p, d, q, pp, dd, qq, s),
        n_exog,
        trend,
        enforce_stationarity: flags.enforce_stationarity,
        enforce_invertibility: flags.enforce_invertibility,
        concentrate_scale: flags.concentrate_scale,
        simple_differencing: flags.simple_differencing,
    }
}

/// Parse optional trend string from Python into Trend enum.
pub(super) fn parse_trend(trend: Option<&str>) -> PyResult<Trend> {
    Trend::from_str(trend.unwrap_or("n")).map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Common validation and config-building for single-series PyO3 functions.
///
/// Validates endog (finite), parses and validates exog, checks order bounds,
/// and builds `SarimaxConfig`. Returns `(config, exog_cols)`.
pub(super) fn prepare_single_request(
    endog: &[f64],
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    exog: Option<&PyReadonlyArray2<f64>>,
    flags: &ModelFlags,
    trend: Option<&str>,
) -> PyResult<(SarimaxConfig, Option<Vec<Vec<f64>>>)> {
    let (exog_cols, n_exog) = validate_single_inputs(endog, exog)?;
    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;
    validate_order(p, d, q, pp, dd, qq, s, n_exog)?;
    let config = build_config(order, seasonal, n_exog, flags, parse_trend(trend)?);
    rustima_core::validation::validate_request(endog, &config, exog_cols.as_deref())
        .map_err(to_pyerr)?;
    Ok((config, exog_cols))
}

/// Common validation and config-building for batch (multi-series) PyO3 functions.
///
/// Validates each series (finite), parses and validates the per-series exog
/// list, checks order bounds, and builds `SarimaxConfig`.
/// Returns `(series, exog_vecs, config)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_batch_request(
    series_list: &[PyReadonlyArray1<'_, f64>],
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    exog_list: &Option<Vec<PyReadonlyArray2<'_, f64>>>,
    flags: &ModelFlags,
    trend: Option<&str>,
) -> PyResult<(Vec<Vec<f64>>, Option<Vec<Vec<Vec<f64>>>>, SarimaxConfig)> {
    let series = validate_batch_finite(series_list)?;
    let series_lengths: Vec<usize> = series.iter().map(|s| s.len()).collect();
    let (exog_vecs, n_exog) = validate_batch_exog(exog_list, series_list.len(), &series_lengths)?;
    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;
    validate_order(p, d, q, pp, dd, qq, s, n_exog)?;
    let config = build_config(order, seasonal, n_exog, flags, parse_trend(trend)?);
    Ok((series, exog_vecs, config))
}

/// Validate that alpha is in the open interval (0, 1).
pub(super) fn validate_alpha(alpha: f64) -> PyResult<()> {
    rustima_core::validation::validate_alpha(alpha).map_err(to_pyerr)
}

/// Validate that forecast steps do not exceed the maximum.
pub(super) fn validate_steps(steps: usize) -> PyResult<()> {
    rustima_core::validation::validate_steps(steps).map_err(to_pyerr)
}
