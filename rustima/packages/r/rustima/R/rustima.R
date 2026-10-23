.count <- function(x, name, max = .Machine$integer.max) {
    if (!is.numeric(x) || length(x) != 1L || is.na(x) || !is.finite(x) ||
        x < 0 || x > max || x != floor(x))
        stop(name, " must be a nonnegative integer", call. = FALSE)
    as.integer(x)
}

.orders <- function(x, n, name) {
    if (!is.numeric(x) || length(x) != n)
        stop(name, " must have length ", n, call. = FALSE)
    vapply(seq_len(n), function(i) .count(x[i], name), integer(1))
}

.spec <- function(order, seasonal, trend, enforce_stationarity,
                  enforce_invertibility, concentrate_scale, simple_differencing) {
    if (is.null(seasonal)) seasonal <- c(0L, 0L, 0L, 0L)
    if (is.list(seasonal)) {
        if (is.null(seasonal$order) || is.null(seasonal$period))
            stop("seasonal needs order and period", call. = FALSE)
        seasonal <- c(seasonal$order, seasonal$period)
    }
    flags <- list(enforce_stationarity = enforce_stationarity,
                  enforce_invertibility = enforce_invertibility,
                  concentrate_scale = concentrate_scale,
                  simple_differencing = simple_differencing)
    if (!all(vapply(flags, function(x) is.logical(x) && length(x) == 1L &&
                    !is.na(x), logical(1))))
        stop("model flags must be TRUE or FALSE", call. = FALSE)
    if (!is.character(trend) || length(trend) != 1L || is.na(trend) ||
        !trend %in% c("n", "c", "t", "ct"))
        stop("trend must be n, c, t or ct", call. = FALSE)
    c(list(order = .orders(order, 3L, "order"),
           seasonal = .orders(seasonal, 4L, "seasonal"), trend = trend), flags)
}

.numeric <- function(x, name, empty = FALSE) {
    if (!is.numeric(x) || is.complex(x) || !is.null(dim(x)) || (!empty && !length(x)) ||
        anyNA(x) || any(!is.finite(x)))
        stop(name, " must be finite numeric data", call. = FALSE)
    as.double(x)
}

.columns <- function(x, n) {
    if (is.null(x)) return(list())
    if (!is.matrix(x) || !is.numeric(x) || is.complex(x) || nrow(x) != n ||
        anyNA(x) || any(!is.finite(x)))
        stop("xreg must be a finite numeric matrix with one row per observation", call. = FALSE)
    lapply(seq_len(ncol(x)), function(j) as.double(x[, j]))
}

.fit_object <- function(result, y, spec, xreg) {
    if (!is.null(result$error)) return(result)
    result$y <- y
    result$spec <- spec
    result$xreg <- xreg
    result$tsp <- attr(y, "tsp")
    class(result) <- "rustima_fit"
    result
}

sarimax <- function(y, order, seasonal = NULL, xreg = NULL, optimizer = "lbfgsb",
                    maxiter = 500L, start_params = NULL, trend = "n",
                    enforce_stationarity = TRUE, enforce_invertibility = TRUE,
                    concentrate_scale = FALSE, simple_differencing = FALSE) {
    spec <- .spec(order, seasonal, trend, enforce_stationarity,
                  enforce_invertibility, concentrate_scale, simple_differencing)
    if (!is.null(start_params)) start_params <- .numeric(start_params, "start_params", TRUE)
    result <- .Call(wrap__engine_fit, .numeric(y, "y"), spec, .columns(xreg, length(y)),
                    start_params, optimizer, .count(maxiter, "maxiter"))
    .fit_object(result, y, spec, xreg)
}

sarimax_loglike <- function(y, params, order, seasonal = NULL, xreg = NULL,
                            trend = "n", enforce_stationarity = TRUE,
                            enforce_invertibility = TRUE, concentrate_scale = FALSE,
                            simple_differencing = FALSE) {
    spec <- .spec(order, seasonal, trend, enforce_stationarity,
                  enforce_invertibility, concentrate_scale, simple_differencing)
    .Call(wrap__engine_loglike, .numeric(y, "y"), spec, .columns(xreg, length(y)),
          .numeric(params, "params", TRUE))
}

coef.rustima_fit <- function(object, ...) object$params

logLik.rustima_fit <- function(object, ...) {
    structure(object$loglike, class = "logLik", df = as.integer(object$n_params),
              nobs = as.integer(object$n_obs))
}

nobs.rustima_fit <- function(object, ...) as.integer(object$n_obs)

print.rustima_fit <- function(x, ...) {
    cat("rustima SARIMAX (", x$method, ")\n", sep = "")
    cat("logLik:", x$loglike, " AIC:", x$aic, " converged:", x$converged, "\n")
    invisible(x)
}

predict.rustima_fit <- function(object, n.ahead = 10L, newxreg = NULL,
                                level = 0.95, ...) {
    n.ahead <- .count(n.ahead, "n.ahead", 10000L)
    if (!is.numeric(level) || length(level) != 1L || is.na(level) ||
        !is.finite(level) || level <= 0 || level >= 1)
        stop("level must be in (0, 1)", call. = FALSE)
    result <- .Call(wrap__engine_forecast, .numeric(object$y, "y"), object$spec,
                    .columns(object$xreg, length(object$y)), object$params,
                    n.ahead, 1 - level, .columns(newxreg, n.ahead))
    if (!is.null(object$tsp) && n.ahead > 0L) {
        start <- object$tsp[2L] + 1 / object$tsp[3L]
        result <- lapply(result, stats::ts, start = start, frequency = object$tsp[3L])
    }
    result
}

sarimax_batch <- function(series, order, seasonal = NULL, xreg = NULL,
                          optimizer = "lbfgsb", maxiter = 500L, nthreads = 1L,
                          trend = "n", enforce_stationarity = TRUE,
                          enforce_invertibility = TRUE, concentrate_scale = FALSE,
                          simple_differencing = FALSE) {
    if (!is.list(series)) stop("series must be a list", call. = FALSE)
    spec <- .spec(order, seasonal, trend, enforce_stationarity,
                  enforce_invertibility, concentrate_scale, simple_differencing)
    if (is.null(xreg)) xreg <- rep(list(NULL), length(series))
    if (!is.list(xreg) || length(xreg) != length(series))
        stop("xreg must have one matrix per series", call. = FALSE)
    cols <- Map(function(y, x) .columns(x, length(y)), series, xreg)
    nthreads <- .count(nthreads, "nthreads", 128L)
    if (!nthreads) stop("nthreads must be positive", call. = FALSE)
    results <- .Call(wrap__engine_batch, lapply(series, .numeric, name = "series"),
                     spec, cols, optimizer, .count(maxiter, "maxiter"), nthreads)
    Map(function(r, y, x) .fit_object(r, y, spec, x), results, series, xreg)
}

sarimax_grid_search <- function(y, orders, seasonal_orders = NULL, xreg = NULL,
                                optimizer = "lbfgsb", maxiter = 500L, nthreads = 1L,
                                trend = "n", enforce_stationarity = TRUE,
                                enforce_invertibility = TRUE, concentrate_scale = FALSE,
                                simple_differencing = FALSE) {
    if (!is.list(orders)) stop("orders must be a list of candidate orders", call. = FALSE)
    if (is.null(seasonal_orders)) seasonal_orders <- rep(list(NULL), length(orders))
    if (!is.list(seasonal_orders) || length(orders) != length(seasonal_orders))
        stop("seasonal_orders must have one entry per order", call. = FALSE)
    specs <- Map(function(o, s) .spec(o, s, trend, enforce_stationarity,
                    enforce_invertibility, concentrate_scale, simple_differencing),
                 orders, seasonal_orders)
    nthreads <- .count(nthreads, "nthreads", 128L)
    if (!nthreads) stop("nthreads must be positive", call. = FALSE)
    results <- .Call(wrap__engine_grid, .numeric(y, "y"), specs, .columns(xreg, length(y)),
                     optimizer, .count(maxiter, "maxiter"), nthreads)
    Map(function(r, spec) .fit_object(r, y, spec, xreg), results, specs)
}

rustima_backend_info <- function() .Call(wrap__engine_info)
