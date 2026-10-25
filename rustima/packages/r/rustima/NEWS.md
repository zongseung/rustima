# rustima 0.1.0 (release candidate)

- `coef()` returns a named vector (`ar.L1`, `ma.S.L12`, `x1`, `sigma2`, ...).
- One invalid series in `sarimax_batch()` / `sarimax_grid_search()` is reported as that
  entry's `error` instead of failing the whole call.
- Validation rules are shared with the Rust core; `trend = "tc"` is accepted as `"ct"`.
- `simple_differencing = TRUE` with `xreg`: forecasts difference `newxreg` like the fitted
  regressors (previously used in levels).
- Add `auto_arima()` with Rust ADF differencing, stepwise/grid search, and prediction methods.
- Share forecast probability, horizon and finite-parameter validation with Python.
- Bundle core test fixtures inside the core crate for standalone testing.
- L-BFGS, BFGS and Nelder-Mead now also use Basin 3.0; remove argmin dependencies.
- Initial R interface to the shared Rust SARIMAX engine through extendr.
- Fixed-order fitting, likelihood evaluation, S3 forecasts and model summaries.
- Independent batch and explicit order-grid fits with a local Rayon thread pool.
- Basin 3.0 replaces the vendored C L-BFGS-B solver.
- Source-package builder bundles the Rust core and dependencies for offline compilation.
