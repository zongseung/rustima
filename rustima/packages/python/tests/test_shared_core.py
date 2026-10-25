"""Public API regressions for shared validation and Rust Auto ARIMA."""

import math

import numpy as np
import pytest

import rustima


def series():
    y = np.random.default_rng(324).normal(size=80)
    for i in range(1, len(y)):
        y[i] += 0.5 * y[i - 1]
    return y


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), -float("inf"), 0.0, 1.0])
def test_native_apis_reject_invalid_alpha(alpha):
    args = (series(), (1, 0, 0), (0, 0, 0, 0), np.array([0.4, 1.0]))
    for function, kwargs in [
        (rustima.sarimax_forecast, {}),
        (rustima.sarimax_rolling_forecast, {"start": 60}),
        (rustima.sarimax_inference, {}),
    ]:
        with pytest.raises(ValueError, match="alpha"):
            function(*args, alpha=alpha, **kwargs)
    with pytest.raises(ValueError, match="alpha"):
        rustima.sarimax_batch_forecast([args[0]], args[1], args[2], [args[3]], alpha=alpha)


def test_nonfinite_params_and_zero_column_shape_checks():
    with pytest.raises(ValueError, match="finite"):
        rustima.sarimax_loglike(series(), (1, 0, 0), (0, 0, 0, 0), np.array([np.nan, 1.0]))
    with pytest.raises(ValueError, match="rows"):
        rustima.auto_arima(series(), exog=np.empty((79, 0)), d=0, D=0)


@pytest.mark.parametrize("criterion", ["aic", "bic", "hqic"])
@pytest.mark.parametrize("with_exog", [False, True])
def test_auto_grid_matches_explicit_candidates(criterion, with_exog):
    y = series()
    kwargs = {"exog": np.random.default_rng(20).normal(size=(len(y), 1))} if with_exog else {}
    method = "profile-trust-region" if with_exog else "lbfgsb"
    orders = [(p, 0, q) for p in range(2) for q in range(2)]
    raw = rustima.sarimax_grid_search(y, orders, [(0, 0, 0, 0)] * 4, method=method, **kwargs)
    def ic(fit):
        if criterion == "hqic":
            return -2 * fit["loglike"] + 2 * fit["n_params"] * math.log(math.log(fit["n_obs"]))
        return fit[criterion]
    candidates = [f for f in raw if "error" not in f and math.isfinite(ic(f))]
    best = min(candidates, key=ic)
    converged = [f for f in candidates if f["converged"]]
    if converged:
        alternative = min(converged, key=ic)
        if ic(alternative) - ic(best) < 5:
            best = alternative
    selected = rustima.auto_arima(y, max_p=1, max_q=1, d=0, D=0,
                                  stepwise=False, criterion=criterion, **kwargs)
    assert selected.order == tuple(best["order"])
    assert selected.best_ic == pytest.approx(ic(best), abs=1e-10)
    np.testing.assert_array_equal(selected.result.params, best["params"])
    assert len(selected.history) == 4


def test_auto_stepwise_trace_and_differencing_without_statsmodels(monkeypatch, capsys):
    import sys
    monkeypatch.setitem(sys.modules, "statsmodels", None)
    fit = rustima.auto_arima(series(), max_p=1, max_q=1, trace=True)
    assert fit.order[1] == 0
    assert len(fit.result.forecast(2).predicted_mean) == 2
    assert "ARIMA(" in capsys.readouterr().out
    assert fit.history_dataframe().height == len(fit.history)


@pytest.mark.parametrize("kwargs", [
    {"max_p": -1}, {"max_q": 21}, {"max_d": 4}, {"s": 1},
    {"s": 0, "D": 1}, {"criterion": "invalid"}, {"method": "invalid"},
])
def test_auto_rejects_invalid_search_inputs(kwargs):
    expected = RuntimeError if kwargs.get("method") == "invalid" else (ValueError, OverflowError)
    with pytest.raises(expected):
        rustima.auto_arima(series(), **kwargs)


def test_all_failed_candidates_remain_in_history():
    fit = rustima.auto_arima(np.array([1.0]), max_p=1, max_q=1, d=3, D=0,
                            simple_differencing=True, stepwise=False)
    assert fit.result is None
    assert len(fit.history) == 4
    assert all(h["aic"] is None and h["error_message"] for h in fit.history)
    assert fit.order is None and math.isinf(fit.best_ic)


@pytest.mark.parametrize("simple_differencing", [False, True])
def test_forecast_rejects_extra_exog_columns(simple_differencing):
    y = series()
    x = np.arange(len(y), dtype=float).reshape(-1, 1)
    params = np.array([0.3, 0.4, 1.0])
    future = np.ones((2, 2))
    kwargs = dict(steps=2, simple_differencing=simple_differencing)
    with pytest.raises(ValueError, match="column count"):
        rustima.sarimax_forecast(y, (1, 1, 0), (0, 0, 0, 0), params,
                                exog=x, future_exog=future, **kwargs)
    results = rustima.sarimax_batch_forecast(
        [y, y], (1, 1, 0), (0, 0, 0, 0), [params, params],
        exog_list=[x, x], exog_forecast_list=[future, future[:, :1]], **kwargs)
    assert "column count" in results[0]["error"]
    assert len(results[1]["mean"]) == 2


@pytest.mark.parametrize("operation", ["loglike", "fit", "forecast"])
def test_batch_zero_column_exog_preserves_row_validation(operation):
    y = series()
    args = ([y], (1, 0, 0), (0, 0, 0, 0))
    params = np.array([0.4, 1.0])
    if operation == "loglike":
        args += (params,)
    elif operation == "forecast":
        args += ([params],)
    kwargs = {"maxiter": 0} if operation == "fit" else {}
    call = getattr(rustima, f"sarimax_batch_{operation}")
    with pytest.raises(ValueError, match="rows"):
        call(*args, exog_list=[np.empty((len(y) - 1, 0))], **kwargs)
    result = call(*args, exog_list=[np.empty((len(y), 0))], **kwargs)
    assert "error" not in result[0]
