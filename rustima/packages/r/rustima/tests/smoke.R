library(rustima)

fails <- function(expr) inherits(tryCatch(force(expr), error = identity), "error")
set.seed(324)
y <- stats::arima.sim(list(ar = 0.5), n = 120)
fit <- sarimax(y, c(1, 0, 0))
alias <- sarimax(y, c(1, 0, 0), optimizer = "basin-lbfgsb")
stopifnot(fit$converged, identical(fit$params, alias$params),
          grepl("basin-lbfgsb", fit$method), length(coef(fit)) == 2L,
          abs(as.numeric(logLik(fit)) - fit$loglike) < 1e-10,
          abs(AIC(fit) - fit$aic) < 1e-10, abs(BIC(fit) - fit$bic) < 1e-10,
          nobs(fit) == length(y), rustima_backend_info()$basin_version == "3.0.0")
ll <- sarimax_loglike(y, fit$params, c(1, 0, 0))
stopifnot(abs(ll - fit$loglike) < 1e-8)
reference <- stats::arima(y, c(1, 0, 0), include.mean = FALSE, method = "ML")
stopifnot(abs(as.numeric(logLik(reference)) - fit$loglike) < 1e-4,
          abs(coef(reference)[1] - fit$params[1]) < 1e-3)
for (method in c("lbfgs", "bfgs", "nm")) {
    alternative <- sarimax(y, c(1, 0, 0), optimizer = method)
    stopifnot(alternative$converged, abs(alternative$loglike - fit$loglike) < 1e-4)
    parallel <- sarimax_batch(list(y, y), c(1, 0, 0), optimizer = method, nthreads = 2)
    stopifnot(all(vapply(parallel, function(f) f$converged &&
              identical(f$params, alternative$params), logical(1))))
}
fc <- predict(fit, 8L)
stopifnot(length(fc$pred) == 8L, all(is.finite(fc$pred)), all(fc$se > 0),
          all(fc$lower < fc$pred), all(fc$upper > fc$pred),
          length(predict(fit, 0L)$pred) == 0L)
tmp <- tempfile(fileext = ".rds")
saveRDS(fit, tmp)
stopifnot(identical(predict(readRDS(tmp), 8L), fc))
unlink(tmp)
dated <- sarimax(ts(y, start = c(2020, 1), frequency = 12), c(1, 0, 0))
stopifnot(is.ts(predict(dated, 3L)$pred),
          abs(start(predict(dated, 3L)$pred)[1] - 2030) < 1e-8)

x <- cbind(a = rnorm(length(y)), b = rnorm(length(y)))
yx <- 1.5 * x[, 1] - 0.3 * x[, 2] + as.numeric(y)
reg <- sarimax(yx, c(1, 0, 0), xreg = x)
ptr <- sarimax(yx, c(1, 0, 0), xreg = x, optimizer = "profile-trust-region")
stopifnot(reg$converged, ptr$converged, abs(reg$loglike - ptr$loglike) < 1e-3,
          length(predict(reg, 2L, newxreg = x[1:2, , drop = FALSE])$pred) == 2L,
          fails(predict(reg, 2L)), fails(predict(reg, 2L, newxreg = x[1:3, , drop = FALSE])))

for (n in c(1L, 2L)) {
    batch <- sarimax_batch(list(y, y), c(1, 0, 0), nthreads = n)
    stopifnot(length(batch) == 2L, all(vapply(batch, inherits, logical(1), "rustima_fit")),
              identical(batch[[1]]$params, fit$params), identical(batch[[2]]$params, fit$params))
    grid <- sarimax_grid_search(y, list(c(1, 0, 0), c(2, 0, 0)), nthreads = n)
    stopifnot(length(grid) == 2L, identical(grid[[1]]$params, fit$params), grid[[2]]$converged)
}
stopifnot(length(sarimax_batch(list(), c(1, 0, 0))) == 0L,
          length(sarimax_grid_search(y, list())) == 0L,
          fails(sarimax(c(1, NA), c(1, 0, 0))),
          fails(sarimax(matrix(y, ncol = 2), c(1, 0, 0))),
          fails(sarimax(y, c(1.5, 0, 0))), fails(sarimax(y, c(21, 0, 0))),
          fails(sarimax(y, c(1, 0, 0), seasonal = c(1, 0, 0, 0))),
          fails(sarimax(y, c(1, 0, 0), xreg = matrix(numeric(), 3, 0))),
          fails(sarimax(y, c(1, 0, 0), start_params = c(NA, 1))),
          fails(sarimax(y, c(1, 0, 0), enforce_stationarity = NA)),
          fails(sarimax(y, c(1, 0, 0), optimizer = "unknown")),
          fails(sarimax_batch(list(y), c(1, 0, 0), nthreads = 0)),
          fails(predict(fit, 10001L)), fails(predict(fit, level = NaN)))

sd_reg <- sarimax(cumsum(y), c(1, 1, 0), xreg = x[, 1, drop = FALSE],
                  simple_differencing = TRUE, start_params = c(0.3, 0.4, 1), maxiter = 0L)
stopifnot(fails(predict(sd_reg, 2L, newxreg = x[1:2, , drop = FALSE])),
          length(predict(sd_reg, 2L, newxreg = x[1:2, 1, drop = FALSE])$pred) == 2L)
