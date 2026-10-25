auto_arima <- function(y, xreg = NULL, max_p = 5L, max_q = 5L, max_d = 2L,
                       max_P = 2L, max_Q = 2L, max_D = 1L, s = 0L,
                       d = NULL, D = NULL, trend = "n", criterion = "aic",
                       stepwise = TRUE, enforce_stationarity = TRUE,
                       enforce_invertibility = TRUE, optimizer = NULL,
                       maxiter = 500L, simple_differencing = FALSE,
                       nthreads = 1L, trace = FALSE) {
    spec <- .spec(c(0L, 0L, 0L), NULL, trend, enforce_stationarity,
                  enforce_invertibility, FALSE, simple_differencing)
    for (flag in list(stepwise, trace)) {
        if (!is.logical(flag) || length(flag) != 1L || is.na(flag))
            stop("stepwise and trace must be TRUE or FALSE", call. = FALSE)
    }
    criterion <- match.arg(criterion, c("aic", "bic", "hqic"))
    if (is.null(optimizer)) optimizer <- ""
    if (!is.character(optimizer) || length(optimizer) != 1L || is.na(optimizer))
        stop("optimizer must be a string or NULL", call. = FALSE)
    nthreads <- .count(nthreads, "nthreads", 128L)
    if (!nthreads) stop("nthreads must be positive", call. = FALSE)
    optional_diff <- function(x, name) if (is.null(x)) -1L else .count(x, name)
    max_order <- c(.count(max_p, "max_p"), .count(max_d, "max_d"), .count(max_q, "max_q"))
    max_seasonal <- c(.count(max_P, "max_P"), .count(max_D, "max_D"),
                      .count(max_Q, "max_Q"), .count(s, "s"))
    raw <- .Call(wrap__engine_auto, .numeric(y, "y"), spec, .columns(xreg, length(y)),
                 max_order, max_seasonal,
                 optional_diff(d, "d"), optional_diff(D, "D"),
                 criterion, stepwise, optimizer, .count(maxiter, "maxiter"), nthreads)
    history <- lapply(raw$history, function(entry) {
        entry[[criterion]] <- entry$ic
        entry$ic <- NULL
        entry
    })
    if (trace) {
        for (entry in history)
            cat("ARIMA(", paste(entry$order, collapse = ","), ")(",
                paste(entry$seasonal_order[1:3], collapse = ","), ")[",
                entry$seasonal_order[4], "] : ", criterion, "=", entry[[criterion]], "\n", sep = "")
    }
    result <- NULL
    if (!is.null(raw$best)) {
        spec$order <- raw$best$order
        spec$seasonal <- raw$best$seasonal
        result <- .fit_object(raw$best$fit, y, spec, xreg)
    }
    structure(list(result = result, history = history, criterion = criterion,
                   d = raw$d, D = raw$D), class = "rustima_auto")
}

.selected_fit <- function(object) {
    if (is.null(object$result)) stop("auto_arima: no valid model found", call. = FALSE)
    object$result
}

predict.rustima_auto <- function(object, ...) predict(.selected_fit(object), ...)
coef.rustima_auto <- function(object, ...) coef(.selected_fit(object), ...)
logLik.rustima_auto <- function(object, ...) logLik(.selected_fit(object), ...)
nobs.rustima_auto <- function(object, ...) nobs(.selected_fit(object), ...)
print.rustima_auto <- function(x, ...) {
    if (is.null(x$result)) cat("auto_arima: no valid model found\n") else print(x$result)
    cat("Search:", length(x$history), "models; criterion:", x$criterion, "\n")
    invisible(x)
}
