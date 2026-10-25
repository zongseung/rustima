# rustima development workspace

Shared Rust SARIMAX engine with separate Python and R packages.
See the [project overview](../README.md) or [Korean guide](../README_ko.md).

| Directory | Responsibility |
|---|---|
| `crates/rustima-core/` | Kalman likelihood, optimization, forecasts, inference, Rayon jobs |
| `packages/python/` | PyO3 extension, Python API, Python packaging and tests |
| `packages/r/rustima/` | extendr adapter, R API, documentation and tests |
| `tests/fixtures/` | Shared numerical reference data |
| `tools/build_r_package.py` | Assemble a standalone R source package |
| `benchmarks/` | Performance experiments |

Basin 3.0 provides L-BFGS-B, L-BFGS, BFGS and Nelder-Mead. The vendored C solver and
its build script have been removed. Existing `lbfgsb*` names and `basin-lbfgsb*`
aliases both select Basin. Trust-region and profile-trust-region remain shared
Rust implementations.

`lbfgs`, `bfgs` and `nm` keep their existing API names. Their solver iterations,
BFGS diagonal preconditioning and initial simplex construction are preserved;
`argmin` and `argmin-math` are no longer dependencies.

From this directory, with Rust >= 1.87:

```sh
cargo test -p rustima-core --all-targets --locked

# Python >= 3.10; activate a virtual environment first.
python -m pip install maturin numpy polars pytest
maturin develop --release --manifest-path packages/python/Cargo.toml
python -m pytest packages/python/tests -o pythonpath=

# R >= 4.2, Cargo/rustc and the platform's R build tools must be installed.
python3 tools/build_r_package.py
R CMD INSTALL dist/r/rustima_0.1.1.tar.gz
R CMD check --no-manual --as-cran dist/r/rustima_0.1.1.tar.gz
```

The R builder copies the shared core into a staging directory and vendors the
locked Rust dependencies. The resulting archive compiles with Cargo's `--offline`
and `--locked` options. Installing the raw `packages/r/rustima` directory is not
supported; build the archive first. Python and R use the same core source; no
second hand-maintained engine is kept under the R package.

Release candidate tags follow `py-v<VERSION>-rc.<N>` and `r-v<VERSION>-rc.<N>`.
Python tags trigger wheel/sdist builds; R tags trigger source builds and R checks.
R tags never invoke the PyPI publication workflow. A CRAN submission remains a
separate maintainer action after platform checks pass.
