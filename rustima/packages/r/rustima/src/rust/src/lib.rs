use extendr_api::prelude::*;
use extendr_api::Result;
use rustima_core::{
    batch, forecast, optimizer,
    params::SarimaxParams,
    pipeline,
    types::{FitResult, SarimaxConfig, SarimaxOrder, Trend},
    validation,
};

fn error(e: impl std::fmt::Display) -> Error {
    Error::Other(e.to_string())
}

fn columns(x: List) -> Result<Option<Vec<Vec<f64>>>> {
    if x.is_empty() {
        return Ok(None);
    }
    x.values()
        .map(|v| {
            v.as_real_vector()
                .ok_or_else(|| error("xreg columns must be numeric"))
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn config(spec: &List, n_exog: usize) -> Result<SarimaxConfig> {
    let ints = |key: &str| -> Result<Vec<usize>> {
        spec.dollar(key)?
            .as_integer_vector()
            .ok_or_else(|| error("orders must be integer vectors"))?
            .into_iter()
            .map(|v| usize::try_from(v).map_err(error))
            .collect()
    };
    let o = ints("order")?;
    let s = ints("seasonal")?;
    if o.len() != 3 || s.len() != 4 {
        return Err(error("expected order length 3 and seasonal length 4"));
    }
    let flag = |key: &str| -> Result<bool> {
        spec.dollar(key)?
            .as_bool()
            .ok_or_else(|| error(format!("{key} must be TRUE or FALSE")))
    };
    let trend_obj = spec.dollar("trend")?;
    let trend = Trend::from_str(trend_obj.as_str().ok_or_else(|| error("invalid trend"))?)
        .map_err(error)?;
    Ok(SarimaxConfig {
        order: SarimaxOrder::new(o[0], o[1], o[2], s[0], s[1], s[2], s[3]),
        n_exog,
        trend,
        enforce_stationarity: flag("enforce_stationarity")?,
        enforce_invertibility: flag("enforce_invertibility")?,
        concentrate_scale: flag("concentrate_scale")?,
        simple_differencing: flag("simple_differencing")?,
    })
}

fn fit_list(f: FitResult) -> List {
    list!(
        params = f.params,
        loglike = f.loglike,
        scale = f.scale,
        aic = f.aic,
        bic = f.bic,
        n_obs = f.n_obs as f64,
        n_params = f.n_params as f64,
        n_iter = f.n_iter as f64,
        converged = f.converged,
        method = f.method,
        warnings = f.warnings
    )
}

fn check_params(params: &[f64], c: &SarimaxConfig) -> Result<SarimaxParams> {
    if params.iter().any(|x| !x.is_finite()) {
        return Err(error("params must be finite"));
    }
    SarimaxParams::from_flat(params, c).map_err(error)
}

fn pool(n: i32) -> Result<rayon::ThreadPool> {
    if !(1..=128).contains(&n) {
        return Err(error("nthreads must be between 1 and 128"));
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(n as usize)
        .build()
        .map_err(error)
}

#[extendr]
fn engine_info() -> List {
    list!(
        core_version = rustima_core::VERSION,
        basin_version = "3.0.0",
        backend = "basin-lbfgsb"
    )
}

#[extendr]
fn engine_fit(
    y: Vec<f64>,
    spec: List,
    xreg: List,
    start: Nullable<Vec<f64>>,
    optimizer: &str,
    maxiter: i32,
) -> Result<List> {
    if maxiter < 0 {
        return Err(error("maxiter must be nonnegative"));
    }
    let x = columns(xreg)?;
    let c = config(&spec, x.as_ref().map_or(0, Vec::len))?;
    validation::validate_request(&y, &c, x.as_deref()).map_err(error)?;
    let start = match start {
        Nullable::Null => None,
        Nullable::NotNull(v) => Some(v),
    };
    if start
        .as_ref()
        .is_some_and(|v| v.iter().any(|x| !x.is_finite()))
    {
        return Err(error("start_params must be finite"));
    }
    optimizer::fit(
        &y,
        &c,
        start.as_deref(),
        Some(optimizer),
        Some(maxiter as u64),
        x.as_deref(),
    )
    .map(fit_list)
    .map_err(error)
}

#[extendr]
fn engine_loglike(y: Vec<f64>, spec: List, xreg: List, params: Vec<f64>) -> Result<f64> {
    let x = columns(xreg)?;
    let c = config(&spec, x.as_ref().map_or(0, Vec::len))?;
    validation::validate_request(&y, &c, x.as_deref()).map_err(error)?;
    let p = check_params(&params, &c)?;
    pipeline::kalman_eval(&y, &p, &c, x.as_deref())
        .map(|v| v.loglike)
        .map_err(error)
}

#[extendr]
fn engine_forecast(
    y: Vec<f64>,
    spec: List,
    xreg: List,
    params: Vec<f64>,
    n_ahead: i32,
    alpha: f64,
    future: List,
) -> Result<List> {
    if !(0..=10_000).contains(&n_ahead) {
        return Err(error("n.ahead must be between 0 and 10000"));
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(error("alpha must be in (0, 1)"));
    }
    let x = columns(xreg)?;
    let c = config(&spec, x.as_ref().map_or(0, Vec::len))?;
    validation::validate_request(&y, &c, x.as_deref()).map_err(error)?;
    let p = check_params(&params, &c)?;
    let future = columns(future)?;
    let f = forecast::forecast_pipeline(
        &y,
        &c,
        &p,
        n_ahead as usize,
        alpha,
        x.as_deref(),
        future.as_deref(),
    )
    .map_err(error)?;
    Ok(list!(
        pred = f.mean,
        se = f.variance.iter().map(|v| v.sqrt()).collect::<Vec<_>>(),
        lower = f.ci_lower,
        upper = f.ci_upper
    ))
}

#[extendr]
fn engine_batch(
    series: List,
    spec: List,
    xreg: List,
    optimizer: &str,
    maxiter: i32,
    nthreads: i32,
) -> Result<List> {
    if maxiter < 0 {
        return Err(error("maxiter must be nonnegative"));
    }
    optimizer::validate_method(optimizer).map_err(error)?;
    if series.len() != xreg.len() {
        return Err(error("xreg list length must match series"));
    }
    let y: Vec<Vec<f64>> = series
        .values()
        .map(|v| {
            v.as_real_vector()
                .ok_or_else(|| error("series must be numeric"))
        })
        .collect::<Result<_>>()?;
    let x: Vec<Vec<Vec<f64>>> = xreg
        .values()
        .map(|v| {
            let cols = List::try_from(v)?;
            columns(cols).map(Option::unwrap_or_default)
        })
        .collect::<Result<_>>()?;
    let c = config(&spec, x.first().map_or(0, Vec::len))?;
    for (y, x) in y.iter().zip(&x) {
        validation::validate_request(y, &c, Some(x)).map_err(error)?;
    }
    let results = pool(nthreads)?
        .install(|| batch::batch_fit(&y, &c, Some(optimizer), Some(maxiter as u64), Some(&x)));
    Ok(List::from_values(results.into_iter().map(|r| match r {
        Ok(f) => fit_list(f),
        Err(e) => list!(error = e.to_string()),
    })))
}

#[extendr]
fn engine_grid(
    y: Vec<f64>,
    specs: List,
    xreg: List,
    optimizer: &str,
    maxiter: i32,
    nthreads: i32,
) -> Result<List> {
    if maxiter < 0 {
        return Err(error("maxiter must be nonnegative"));
    }
    optimizer::validate_method(optimizer).map_err(error)?;
    let x = columns(xreg)?;
    let configs: Vec<_> = specs
        .values()
        .map(|v| {
            let c = config(&List::try_from(v)?, x.as_ref().map_or(0, Vec::len))?;
            validation::validate_request(&y, &c, x.as_deref()).map_err(error)?;
            Ok(c)
        })
        .collect::<Result<_>>()?;
    let results = pool(nthreads)?.install(|| {
        batch::grid_search_fit(
            &y,
            &configs,
            Some(optimizer),
            Some(maxiter as u64),
            x.as_deref(),
        )
    });
    Ok(List::from_values(results.into_iter().map(|r| match r {
        Ok(f) => fit_list(f),
        Err(e) => list!(error = e.to_string()),
    })))
}

extendr_module! {
    mod rustima;
    fn engine_info;
    fn engine_fit;
    fn engine_loglike;
    fn engine_forecast;
    fn engine_batch;
    fn engine_grid;
}
