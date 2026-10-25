"""Compare installed Python and R packages on identical SARIMAX inputs.

Run with Python's numpy/rustima installed and R's rustima on R_LIBS_USER.
"""

from pathlib import Path
import subprocess
import tempfile

import numpy as np
import rustima


def r_vector(values):
    return "c(" + ",".join(format(float(x), ".17g") for x in values) + ")"


def check():
    rng = np.random.default_rng(324)
    x = rng.normal(size=(160, 2))
    noise = rng.normal(size=160)
    for t in range(4, len(noise)):
        noise[t] += 0.35 * noise[t - 1] + 0.2 * noise[t - 4]
    cases = [
        ((1, 0, 0), (0, 0, 0, 0), False, "lbfgsb"),
        ((1, 1, 1), (0, 0, 0, 0), False, "lbfgsb"),
        ((1, 0, 1), (1, 0, 0, 4), False, "lbfgsb"),
        ((1, 0, 0), (0, 0, 0, 0), True, "lbfgsb"),
        ((1, 0, 0), (0, 0, 0, 0), True, "profile-trust-region"),
        ((1, 0, 0), (0, 0, 0, 0), True, "lbfgs"),
        ((1, 0, 0), (0, 0, 0, 0), True, "bfgs"),
        ((1, 0, 0), (0, 0, 0, 0), True, "nm"),
    ]
    with tempfile.TemporaryDirectory(prefix="rustima-r-python-") as tmp:
        tmp = Path(tmp)
        for i, (order, seasonal, exog, method) in enumerate(cases):
            y = noise + (1.5 * x[:, 0] - 0.3 * x[:, 1] if exog else 0)
            kwargs = {"exog": x} if exog else {}
            fit = rustima.sarimax_fit(y, order, seasonal, method=method, **kwargs)
            params = np.asarray(fit["params"])
            forecast = rustima.sarimax_forecast(y, order, seasonal, params,
                steps=4, **kwargs, **({"future_exog": x[:4]} if exog else {}))
            data = tmp / "input.csv"
            np.savetxt(data, np.column_stack((y, x)), delimiter=",")
            out = tmp / "output.csv"
            r = f'''
library(rustima)
d <- as.matrix(read.csv("{data.as_posix()}", header = FALSE))
x <- {"d[, 2:3, drop = FALSE]" if exog else "NULL"}
fit <- sarimax(d[, 1], {r_vector(order)}, seasonal = {r_vector(seasonal)},
               xreg = x, optimizer = "{method}")
stopifnot(fit$converged)
ll <- sarimax_loglike(d[, 1], {r_vector(params)}, {r_vector(order)},
                      seasonal = {r_vector(seasonal)}, xreg = x)
fc <- predict(fit, 4L, newxreg = {"x[1:4, , drop = FALSE]" if exog else "NULL"})
options(digits = 17)
write.table(c(ll, fit$loglike, fit$params, fc$pred), "{out.as_posix()}",
            row.names = FALSE, col.names = FALSE)
'''
            subprocess.run(["Rscript", "--vanilla", "-e", r], check=True)
            actual = np.loadtxt(out)
            assert fit["converged"], (order, seasonal, method)
            np.testing.assert_allclose(actual[0], fit["loglike"], atol=1e-9, rtol=0)
            expected = np.r_[fit["loglike"], params, forecast["mean"]]
            np.testing.assert_allclose(actual[1:], expected, atol=1e-5, rtol=1e-5)
            print(f"PASS {i + 1}: {order} {seasonal} exog={exog} {method}")


if __name__ == "__main__":
    check()
