library(rustima)
fails <- function(expr) inherits(tryCatch(force(expr), error = identity), "error")
set.seed(324)
y <- as.numeric(stats::arima.sim(list(ar = 0.5), n = 80))
for (stepwise in c(TRUE, FALSE)) {
    a <- auto_arima(y, max_p = 1, max_q = 1, d = 0, D = 0, stepwise = stepwise)
    b <- auto_arima(y, max_p = 1, max_q = 1, d = 0, D = 0, stepwise = stepwise, nthreads = 2)
    stopifnot(inherits(a, "rustima_auto"), inherits(a$result, "rustima_fit"),
              identical(coef(a), coef(b)), identical(a$history, b$history),
              identical(predict(a, 3)$pred, predict(a$result, 3)$pred),
              identical(logLik(a), logLik(a$result)), nobs(a) == length(y))
    file <- tempfile(fileext = ".rds")
    saveRDS(a, file)
    stopifnot(identical(predict(readRDS(file), 3), predict(a, 3)))
    unlink(file)
}
x <- matrix(rnorm(length(y)), ncol = 1)
selected <- auto_arima(y, xreg = x, max_p = 1, max_q = 0, d = 0, D = 0)
stopifnot(selected$result$method == "profile-trust-region",
          length(predict(selected, 2, newxreg = x[1:2, , drop = FALSE])$pred) == 2,
          fails(predict(selected, level = NaN)), fails(predict(selected, 10001)))
detected <- auto_arima(y, max_p = 1, max_q = 1)
stopifnot(detected$d %in% 0:2, detected$D == 0)
failed <- auto_arima(1, max_p = 1, max_q = 1, d = 3, D = 0,
                     simple_differencing = TRUE, stepwise = FALSE)
stopifnot(is.null(failed$result), length(failed$history) == 4,
          all(vapply(failed$history, function(h) !is.na(h$error), logical(1))),
          fails(predict(failed)))
stopifnot(fails(auto_arima(y, max_p = -1)), fails(auto_arima(y, max_p = 21)),
          fails(auto_arima(y, max_p = c(1, 0), max_d = NULL)),
          fails(auto_arima(y, s = 1)), fails(auto_arima(y, D = 1, s = 0)),
          fails(auto_arima(y, nthreads = 0)), fails(auto_arima(y, criterion = "invalid")),
          fails(auto_arima(y, stepwise = NA)), fails(auto_arima(y, optimizer = "invalid")))
