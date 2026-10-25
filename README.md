# rustima

[![CI](https://github.com/zongseung/rustima/actions/workflows/ci.yml/badge.svg)](https://github.com/zongseung/rustima/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![R](https://img.shields.io/badge/R-4.2%2B%20preview-blue)
![Rust](https://img.shields.io/badge/Rust-1.87%2B-orange)
[![License](https://img.shields.io/badge/License-GPLv3%2B-blue)](rustima/LICENSE)

[한국어](README_ko.md) · [Development guide](rustima/README.md) · [Changelog](rustima/CHANGELOG.md)

A Rust SARIMAX engine with Python and R interfaces. It evaluates state-space
Kalman likelihoods, fits seasonal models with optional regression terms, and
forecasts with prediction intervals. Independent series and explicit order
candidates can be fitted in parallel with Rayon.

Python `0.4.0` and the R `0.1.1` preview share one Rust core.
L-BFGS-B (default), L-BFGS, BFGS and Nelder-Mead use **Basin 3.0**. The previous
vendored C optimizer is removed; `lbfgsb` and `basin-lbfgsb` select the same
backend. Trust-region and profile-trust-region remain rustima implementations.
L-BFGS-B's `n_iter` counts objective/gradient evaluations, including line search;
the other Basin methods count completed solver iterations.

## Python

Install the published Python version with `pip install rustima`. That release
can differ from the development version shown here. To build this checkout:

```sh
cd rustima
python -m pip install maturin
# Activate a virtual environment before maturin develop.
maturin develop --release --manifest-path packages/python/Cargo.toml
```

```python
import numpy as np
from rustima import SARIMAXModel, auto_arima

y = np.random.default_rng(10).normal(size=200)
result = SARIMAXModel(y, order=(1, 0, 1)).fit()
print(result.params, result.aic)
print(result.forecast(5))

selected = auto_arima(y, s=0)
print(selected.order)
```

Python also supports exogenous regressors, inference, diagnostics, rolling
forecasts, batch fitting and stepwise/grid Auto ARIMA. See the
[Python package](rustima/packages/python/README.md) and its tests for examples.
Auto ARIMA now selects differences in Rust using ADF and the seasonal variance
ratio, independently of whether statsmodels is installed. Search traces print
after the Rust call returns.

## R preview

The R package is being prepared for distribution and has not been published to
CRAN. Source installation requires R >= 4.2, Rust >= 1.87, Cargo and the normal
R build tools for your platform, including Rtools on Windows.

```sh
cd rustima
python3 tools/build_r_package.py
R CMD INSTALL dist/r/rustima_0.1.1.tar.gz
```

The builder bundles the Rust core and locked dependencies, so Cargo compilation
during installation works offline. Python is only needed to assemble the archive;
the installed R package calls Rust directly through extendr.

```r
library(rustima)
set.seed(10)
y <- arima.sim(list(ar = 0.5), n = 120)
fit <- sarimax(y, order = c(1, 0, 0))
coef(fit)
AIC(fit)
predict(fit, n.ahead = 5)$pred

fits <- sarimax_batch(list(y, y), c(1, 0, 0), nthreads = 2)
grid <- sarimax_grid_search(y, list(c(1, 0, 0), c(2, 0, 0)), nthreads = 2)
selected <- auto_arima(y, max_p = 2, max_q = 2, nthreads = 2)
predict(selected, n.ahead = 5)$pred
rustima_backend_info()
```

R exposes fixed-order fitting, likelihood evaluation, forecasts, batch fits,
explicit paired order candidates, and stepwise/grid Auto ARIMA. Python and R
share Rust differencing selection and candidate search. Check each selected fit's
`converged` and `warnings` fields. Missing and nonfinite observations are rejected.

## Repository and releases

```text
rustima/
  crates/rustima-core/       shared statistical engine
    tests/fixtures/         self-contained core test references
  packages/python/          Python binding and package
  packages/r/rustima/        R binding and package
  tests/fixtures/            numerical reference data
  tools/build_r_package.py  standalone R source-package builder
```

Python and R have independent package versions and release tag prefixes:
`py-v…` and `r-v…`. Release candidates use `-rc.N` in Git tags.
See the [development guide](rustima/README.md) for builds and checks.
R release automation produces checked source artifacts; CRAN submission is a
separate step. No GPU backend is included.

Licensed under [GPL-3.0-or-later](rustima/LICENSE).
