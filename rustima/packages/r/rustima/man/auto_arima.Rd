\name{auto_arima}
\alias{auto_arima}
\title{Select an ARIMA Model Using the Shared Rust Engine}
\usage{
auto_arima(y, xreg = NULL, max_p = 5L, max_q = 5L, max_d = 2L,
  max_P = 2L, max_Q = 2L, max_D = 1L, s = 0L,
  d = NULL, D = NULL, trend = "n", criterion = "aic",
  stepwise = TRUE, enforce_stationarity = TRUE,
  enforce_invertibility = TRUE, optimizer = NULL,
  maxiter = 500L, simple_differencing = FALSE,
  nthreads = 1L, trace = FALSE)
}
\arguments{
\item{y}{Finite numeric time series.}
\item{xreg}{Optional finite numeric matrix, one row per observation.}
\item{max_p, max_q, max_P, max_Q}{Maximum nonseasonal and seasonal AR/MA orders.}
\item{max_d, max_D}{Bounds for automatic differencing selection. Seasonal detection selects zero or one difference.}
\item{s}{Seasonal period; zero disables seasonal terms. Set explicitly for seasonal data.}
\item{d, D}{Fixed differencing orders, or \code{NULL} to select automatically.}
\item{trend}{One of \code{"n"}, \code{"c"}, \code{"t"}, or \code{"ct"}.}
\item{criterion}{Information criterion: \code{"aic"}, \code{"bic"}, or \code{"hqic"}.}
\item{stepwise}{Use stepwise candidate search; otherwise evaluate the complete grid.}
\item{enforce_stationarity, enforce_invertibility}{Constrain AR and MA parameters.}
\item{optimizer}{Optimizer name, or \code{NULL} for L-BFGS-B without regressors and profile trust-region with regressors.}
\item{maxiter}{Per-model optimizer work limit, as in \code{sarimax}.}
\item{simple_differencing}{Difference before state-space construction.}
\item{nthreads}{Number of Rayon workers, from 1 to 128.}
\item{trace}{Print candidate history after the Rust search returns.}
}
\description{
Fits and selects candidate models in Rust, using the same search engine as
the Python interface. No Python runtime or additional R forecasting package is needed.
}
\details{
Nonseasonal differencing uses a constant-regression augmented Dickey-Fuller test
at the five-percent level, with AIC lag selection up to \code{min(10, floor(n/3))}.
Series shorter than ten observations and constant series stop differencing.
Seasonal differencing uses a variance ratio below 0.64. Both selectors operate
on the supplied series before exogenous regression, matching the original Python policy.
The stepwise walk follows the best information criterion. Final selection
prefers a converged model unless a nonconverged candidate improves the criterion
by at least five units. This is rustima's policy, not an implementation of every
option of \code{forecast::auto.arima}.
}
\value{
A \code{rustima_auto} list with \code{result} (a \code{rustima_fit}, or \code{NULL}
if every candidate failed), \code{history}, \code{criterion}, and selected
differencing orders \code{d} and \code{D}. Methods for \code{predict}, \code{coef},
\code{logLik} and \code{nobs} use the selected fit. Its orders are stored in
\code{result$spec}. Individual candidate failures remain in the history.
}
\examples{
fit <- auto_arima(as.numeric(LakeHuron), max_p = 1, max_q = 1, d = 0, D = 0)
predict(fit, n.ahead = 3)$pred
}
