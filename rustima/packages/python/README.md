# rustima for Python

Python bindings for the shared Rust SARIMAX engine. Stable version:
`0.4.0`. L-BFGS-B, L-BFGS, BFGS and Nelder-Mead use Basin 3.0;
`basin-lbfgsb` remains an alias. Custom trust-region methods remain available.

From the project workspace:

```sh
uv venv .venv
uv pip install maturin numpy polars pytest
.venv/bin/maturin develop --release --manifest-path packages/python/Cargo.toml
.venv/bin/python -m pytest packages/python/tests -o pythonpath=
```

```python
from rustima import SARIMAXModel

model = SARIMAXModel(y, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
result = model.fit(method="lbfgsb", maxiter=500)
prediction = result.forecast(24)
```

The Python package does not require R. Rust source builds require Rust 1.87
or newer. Existing PyPI releases remain available with `pip install rustima`;
the development version above is not yet published.
