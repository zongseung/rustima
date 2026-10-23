\name{sarimax}
\alias{sarimax}
\alias{sarimax_loglike}
\alias{coef.rustima_fit}
\alias{logLik.rustima_fit}
\alias{nobs.rustima_fit}
\alias{print.rustima_fit}
\alias{predict.rustima_fit}
\title{Fit and Forecast a Seasonal ARIMA Model}
\description{Fits a fixed-order SARIMA model with optional regression terms using
the shared Rust Kalman likelihood and Basin optimizer.}
\usage{
sarimax(y, order, seasonal = NULL, xreg = NULL, optimizer = "lbfgsb",
        maxiter = 500L, start_params = NULL, trend = "n",
        enforce_stationarity = TRUE, enforce_invertibility = TRUE,
        concentrate_scale = FALSE, simple_differencing = FALSE)
sarimax_loglike(y, params, order, seasonal = NULL, xreg = NULL,
        trend = "n", enforce_stationarity = TRUE,
        enforce_invertibility = TRUE, concentrate_scale = FALSE,
        simple_differencing = FALSE)
\method{coef}{rustima_fit}(object, ...)
\method{logLik}{rustima_fit}(object, ...)
\method{nobs}{rustima_fit}(object, ...)
\method{print}{rustima_fit}(x, ...)
\method{predict}{rustima_fit}(object, n.ahead = 10L, newxreg = NULL,
        level = 0.95, ...)
}
\arguments{
\item{y}{A nonempty finite numeric vector or univariate time series. Missing values are not supported.}
\item{order}{Integer vector \code{c(p, d, q)}.}
\item{seasonal}{\code{NULL}, integer vector \code{c(P, D, Q, period)}, or a list
with \code{order} and \code{period}.}
\item{xreg}{Optional finite numeric matrix, with one row per observation.}
\item{optimizer}{Optimizer name. The default \code{"lbfgsb"} and
\code{"basin-lbfgsb"} use Basin 3.0. Other methods include \code{"trust-region"},
\code{"profile-trust-region"}, \code{"lbfgs"}, and \code{"nm"}.}
\item{maxiter}{Nonnegative optimizer work budget. For L-BFGS-B this counts
objective/gradient evaluations, including line search, rather than accepted steps.}
\item{start_params}{Optional constrained parameter vector in the order described below.}
\item{trend}{\code{"n"} (none), \code{"c"} (constant), \code{"t"} (linear), or \code{"ct"} (both).}
\item{enforce_stationarity, enforce_invertibility}{Logical constraint flags.}
\item{concentrate_scale}{Whether to profile the innovation variance out of the likelihood.}
\item{simple_differencing}{Whether to difference the data before filtering.}
\item{params}{Constrained parameter vector: trend coefficients, regression coefficients,
nonseasonal AR, nonseasonal MA, seasonal AR, seasonal MA, then innovation variance
unless \code{concentrate_scale = TRUE}.}
\item{object, x}{An object returned by \code{sarimax}.}
\item{n.ahead}{Forecast horizon, from zero to 10000.}
\item{newxreg}{Future regression matrix, with \code{n.ahead} rows and the original
number of columns. Required for positive horizons when regression terms were fitted.}
\item{level}{Prediction interval coverage between zero and one.}
\item{...}{Reserved for S3 generic compatibility.}
}
\value{\code{sarimax} returns a \code{rustima_fit} list containing parameters,
log likelihood, scale, AIC, BIC, observation and parameter counts, work count,
convergence flag, method, warnings, and the original model inputs.
\code{sarimax_loglike} returns a numeric log likelihood.
\code{predict} returns \code{pred}, \code{se}, \code{lower}, and \code{upper};
time-series inputs preserve their frequency and forecast time index.
The other S3 methods provide coefficients, a \code{logLik} object, the number
of observations, or an invisible copy of the printed object.}
\details{The default evaluates a state-space Kalman likelihood; it does not
substitute a Whittle approximation. CSS is used for initialization in applicable
optimizer paths. Check \code{converged} and \code{warnings} before using a fit.}
\examples{
set.seed(10)
y <- arima.sim(list(ar = 0.5), n = 100)
fit <- sarimax(y, c(1, 0, 0))
coef(fit)
AIC(fit)
predict(fit, n.ahead = 3)$pred
}
