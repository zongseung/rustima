# rustima 0.1.0 (release candidate)

- Add `auto_arima()` with Rust ADF differencing, stepwise/grid search, and prediction methods.
- Share forecast probability, horizon and finite-parameter validation with Python.
- Bundle core test fixtures inside the core crate for standalone testing.
- L-BFGS, BFGS and Nelder-Mead now also use Basin 3.0; remove argmin dependencies.
- Initial R interface to the shared Rust SARIMAX engine through extendr.
- Fixed-order fitting, likelihood evaluation, S3 forecasts and model summaries.
- Independent batch and explicit order-grid fits with a local Rayon thread pool.
- Basin 3.0 replaces the vendored C L-BFGS-B solver.
- Source-package builder bundles the Rust core and dependencies for offline compilation.
