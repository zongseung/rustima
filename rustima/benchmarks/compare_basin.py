"""Reproduce paired optimizer and auto-ARIMA comparisons as JSON.

Build with `maturin develop --release --features basin` first, then run
`python benchmarks/compare_basin.py --repeats 5 --threads 1 > comparison.json`.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
import warnings


def timed_fit(call, repeats):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        call()
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = call()
            elapsed.append(time.perf_counter() - start)
    return result, statistics.median(elapsed)


def fit_summary(result, elapsed):
    return {**{key: result[key] for key in [
        "params", "loglike", "aic", "bic", "converged", "n_iter", "method",
    ]}, "median_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("repeats and threads must be positive")
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import rustima

    root = Path(__file__).resolve().parents[1]
    fixtures = json.loads((root / "tests/fixtures/statsmodels_reference.json").read_text())
    references = json.loads((root / "tests/fixtures/statsmodels_fit_reference.json").read_text())
    report = {"environment": {
        "platform": platform.platform(), "python": platform.python_version(),
        "rustima": rustima.version(), "basin": "3.0.0", "threads": args.threads,
        "repeats": args.repeats, "seed": 324,
    }, "fits": [], "auto_arima": []}
    for name, case in fixtures.items():
        y = np.asarray(case["data"])
        order, seasonal = tuple(case["order"]), tuple(case["seasonal_order"])
        start = np.asarray(case["params"])
        for mode in ["strict", "automatic"]:
            entry = {"case": name, "mode": mode}
            for backend, prefix in [("legacy", ""), ("basin", "basin-")]:
                method = prefix + ("lbfgsb-strict" if mode == "strict" else "lbfgsb")
                def fit():
                    return rustima.sarimax_fit(y, order, seasonal, method=method,
                                              start_params=start if mode == "strict" else None)
                result, elapsed = timed_fit(fit, args.repeats)
                entry[backend] = fit_summary(result, elapsed)
            entry["loglike_delta"] = entry["basin"]["loglike"] - entry["legacy"]["loglike"]
            entry["max_parameter_delta"] = float(np.max(np.abs(
                np.asarray(entry["basin"]["params"]) - entry["legacy"]["params"])))
            if name in references:
                entry["reference_loglike"] = references[name]["loglike"]
            report["fits"].append(entry)

    rng = np.random.default_rng(324)
    x = rng.normal(size=(200, 1))
    noise = rng.normal(size=200)
    for t in range(4, len(noise)):
        noise[t] += 0.4 * noise[t - 1] + 0.2 * noise[t - 4]
    y = 1.8 * x[:, 0] + noise
    entry = {"case": "sarimax_exog", "mode": "automatic"}
    for backend, method in [("legacy", "lbfgsb"), ("basin", "basin-lbfgsb")]:
        result, elapsed = timed_fit(lambda: rustima.sarimax_fit(
            y, (1, 0, 0), (1, 0, 0, 4), exog=x, method=method), args.repeats)
        entry[backend] = fit_summary(result, elapsed)
    entry["loglike_delta"] = entry["basin"]["loglike"] - entry["legacy"]["loglike"]
    report["fits"].append(entry)

    ar1 = np.asarray(fixtures["ar1"]["data"])
    for stepwise in [True, False]:
        for exog in [None, np.sin(np.arange(len(ar1)) * 0.2).reshape(-1, 1)]:
            entry = {"stepwise": stepwise, "exog": exog is not None}
            for backend, method in [("legacy", "lbfgsb"), ("basin", "basin-lbfgsb")]:
                result, elapsed = timed_fit(lambda: rustima.auto_arima(
                    ar1, max_p=2, max_q=1, d=0, D=0, s=0, exog=exog,
                    stepwise=stepwise, method=method), args.repeats)
                if result.result is None:
                    raise RuntimeError(f"{backend} auto-ARIMA failed: {result.history}")
                entry[backend] = {
                    "order": result.order, "seasonal_order": result.seasonal_order,
                    "best_ic": result.best_ic, "converged": result.result.converged,
                    "method": result.result.method, "history": result.history,
                    "median_seconds": elapsed,
                }
            report["auto_arima"].append(entry)
    def serializable(value):
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if hasattr(value, "tolist"):
            return serializable(value.tolist())
        return value
    print(json.dumps(serializable(report), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
