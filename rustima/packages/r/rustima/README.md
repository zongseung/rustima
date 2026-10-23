# rustima R preview

Direct extendr binding to `crates/rustima-core`. No Python runtime is required.
This preview has not been submitted to CRAN.

From the `rustima/` workspace directory:

```sh
python3 tools/build_r_package.py
R CMD INSTALL dist/r/rustima_0.1.0.tar.gz
R CMD check --no-manual --as-cran dist/r/rustima_0.1.0.tar.gz
```

The builder embeds the shared Rust source and vendors locked dependencies.
Install the generated archive rather than this directory. Compilation requires
R >= 4.2, Rust >= 1.87, Cargo and R's platform build tools (Rtools on Windows).

```r
library(rustima)
fit <- sarimax(AirPassengers, c(1, 1, 0), seasonal = c(0, 1, 1, 12))
predict(fit, 12)$pred
```

See `?sarimax`, `?sarimax_batch`, and `?rustima_backend_info` after installation.
The initial API includes explicit grid candidates, not stepwise Auto ARIMA.
