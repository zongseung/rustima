\name{sarimax_batch}
\alias{sarimax_batch}
\alias{sarimax_grid_search}
\title{Fit Independent Series or Explicit Order Candidates}
\description{Runs independent fixed-order fits on a local Rayon thread pool.
R objects are converted before workers start and results are converted after joining.}
\usage{
sarimax_batch(series, order, seasonal = NULL, xreg = NULL,
        optimizer = "lbfgsb", maxiter = 500L, nthreads = 1L,
        trend = "n", enforce_stationarity = TRUE,
        enforce_invertibility = TRUE, concentrate_scale = FALSE,
        simple_differencing = FALSE)
sarimax_grid_search(y, orders, seasonal_orders = NULL, xreg = NULL,
        optimizer = "lbfgsb", maxiter = 500L, nthreads = 1L,
        trend = "n", enforce_stationarity = TRUE,
        enforce_invertibility = TRUE, concentrate_scale = FALSE,
        simple_differencing = FALSE)
}
\arguments{
\item{series}{List of finite numeric series.}
\item{y}{A finite numeric series.}
\item{order, seasonal, optimizer, maxiter, trend, enforce_stationarity,
enforce_invertibility, concentrate_scale, simple_differencing}{See \code{\link{sarimax}}.}
\item{xreg}{For batch fitting, a list of regression matrices, one per series.
For grid search, one regression matrix shared by all candidates. May be \code{NULL}.}
\item{orders}{List of \code{c(p, d, q)} vectors.}
\item{seasonal_orders}{List of seasonal specifications, one per order, or \code{NULL}.}
\item{nthreads}{Number of Rust workers, from one to 128. Defaults to one.}
}
\value{A list in input order. Each element is a \code{rustima_fit}, or a list
with an \code{error} message if that fit failed. Invalid inputs raise an error
before workers start. Empty candidate or series lists return empty lists.}
\details{Grid search fits the supplied paired order specifications and returns
all results. It does not perform automatic differencing selection or stepwise
Auto ARIMA. Independent workers do not call the R API.}
\examples{
set.seed(2)
y <- rnorm(80)
fits <- sarimax_batch(list(y, y + 1), c(1, 0, 0))
grid <- sarimax_grid_search(y, list(c(1, 0, 0), c(0, 0, 1)))
vapply(grid, AIC, numeric(1))
}
