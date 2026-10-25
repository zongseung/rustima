"""Compare installed Python/R Auto ARIMA selections, histories and forecasts."""

from pathlib import Path
import subprocess
import tempfile

import numpy as np
import rustima


def check():
    cases = [
        (True, 0, "aic", False, False),
        (False, 0, "bic", False, False),
        (True, 0, "hqic", True, False),
        (False, 4, "aic", True, False),
        (True, 4, "aic", False, False),
        (True, 0, "aic", False, True),
    ]
    rng = np.random.default_rng(324)
    noise = rng.normal(size=100)
    for t in range(1, len(noise)):
        noise[t] += 0.5 * noise[t - 1]
    x = rng.normal(size=(len(noise), 1))
    with tempfile.TemporaryDirectory(prefix="rustima-auto-parity-") as tmp:
        tmp = Path(tmp)
        for i, (stepwise, period, criterion, exog, integrated) in enumerate(cases):
            y = np.cumsum(noise) if integrated else noise
            y = y + (0.8 * x[:, 0] if exog else 0)
            fit = rustima.auto_arima(
                y, exog=x if exog else None, max_p=1, max_q=1,
                max_P=1, max_Q=1, s=period, criterion=criterion, stepwise=stepwise,
            )
            assert fit.result is not None
            kwargs = {"exog": x[:3]} if exog else {}
            forecast = fit.result.forecast(3, **kwargs).predicted_mean
            expected = np.r_[fit.order, fit.seasonal_order, fit.best_ic,
                             fit.result.llf, fit.result.params, forecast]
            history = np.array([
                [*h["order"], *h["seasonal_order"],
                 h[criterion] if h[criterion] is not None else np.nan, h["converged"]]
                for h in fit.history
            ])
            # Binary input avoids decimal parsing differences near optimizer boundaries.
            data = tmp / "input.bin"
            data.write_bytes(np.column_stack((y, x)).astype("<f8").tobytes(order="F"))
            for workers in (1, 2):
                output, trace = tmp / "result.csv", tmp / "history.csv"
                script = f'''
library(rustima)
d <- matrix(readBin("{data.as_posix()}", "double", n = {len(y) * 2},
                    size = 8, endian = "little"), ncol = 2)
x <- {"d[, 2, drop = FALSE]" if exog else "NULL"}
a <- auto_arima(d[, 1], xreg = x, max_p = 1, max_q = 1, max_P = 1,
                max_Q = 1, s = {period}, criterion = "{criterion}",
                stepwise = {"TRUE" if stepwise else "FALSE"}, nthreads = {workers})
f <- a$result
stopifnot(!is.null(f))
fc <- predict(a, 3, newxreg = {"x[1:3, , drop = FALSE]" if exog else "NULL"})
ic <- if ("{criterion}" == "hqic") -2*f$loglike + 2*f$n_params*log(log(f$n_obs)) else f[["{criterion}"]]
options(digits = 17)
write.table(c(f$spec$order, f$spec$seasonal, ic, f$loglike, f$params, fc$pred),
            "{output.as_posix()}", row.names = FALSE, col.names = FALSE)
h <- do.call(rbind, lapply(a$history, function(e) c(e$order, e$seasonal_order,
                e[["{criterion}"]], as.integer(e$converged))))
write.table(h, "{trace.as_posix()}", row.names = FALSE, col.names = FALSE, na = "NaN")
'''
                subprocess.run(["Rscript", "--vanilla", "-e", script], check=True)
                actual = np.loadtxt(output)
                actual_history = np.loadtxt(trace, ndmin=2)
                np.testing.assert_array_equal(actual[:7], expected[:7])
                np.testing.assert_allclose(actual[7:], expected[7:], atol=1e-8, rtol=1e-8)
                np.testing.assert_array_equal(actual_history[:, :7], history[:, :7])
                np.testing.assert_allclose(actual_history[:, 7:], history[:, 7:],
                                           atol=1e-8, rtol=1e-8, equal_nan=True)
                print(f"PASS Auto {i + 1}: stepwise={stepwise} s={period} {criterion} "
                      f"exog={exog} integrated={integrated} threads={workers}")


if __name__ == "__main__":
    check()
