"""Check Basin defaults, method aliases, work limits and reference fits."""

import numpy as np
import pytest

import rustima
from rustima import SARIMAXModel, auto_arima


def test_basin_is_the_default_backend():
    y = np.sin(np.arange(80) * 0.3)
    for method in ["lbfgsb", "basin-lbfgsb"]:
        result = rustima.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0), method=method, maxiter=0)
        assert result["method"] == "basin-lbfgsb"
    with pytest.raises(RuntimeError, match="unknown method"):
        rustima.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0), method="basin-bfgs")


@pytest.mark.parametrize("case_name", ["ar1", "arma11", "arima111", "sarima_100_100_4"])
@pytest.mark.parametrize("concentrate", [False, True])
def test_identical_start_alias_parity(statsmodels_fixtures, case_name, concentrate):
    case = statsmodels_fixtures[case_name]
    start = list(case["params"][:-1] if concentrate else case["params"])
    kwargs = dict(start_params=np.asarray(start), concentrate_scale=concentrate)
    y = np.asarray(case["data"])
    order, seasonal = tuple(case["order"]), tuple(case["seasonal_order"])
    default = rustima.sarimax_fit(y, order, seasonal, method="lbfgsb-strict", **kwargs)
    basin = rustima.sarimax_fit(y, order, seasonal, method="basin-lbfgsb-strict", **kwargs)
    assert basin["converged"] and default["converged"]
    assert abs(basin["loglike"] - default["loglike"]) <= 1e-4
    np.testing.assert_allclose(basin["params"], default["params"], atol=1e-3, rtol=1e-3)
    assert basin["method"] == "basin-lbfgsb-strict"
    assert set(basin) == set(default)


@pytest.mark.parametrize("case_name", ["ar1", "arma11", "arima111", "sarima_100_100_4"])
def test_reference_fit_with_automatic_initialization(statsmodels_fixtures,
                                                    fit_fixtures, case_name):
    case = statsmodels_fixtures[case_name]
    result = rustima.sarimax_fit(np.asarray(case["data"]), tuple(case["order"]),
                                tuple(case["seasonal_order"]), method="basin-lbfgsb")
    assert result["converged"]
    assert result["loglike"] >= fit_fixtures[case_name]["loglike"] - 3.0
    if case_name != "arima111":
        np.testing.assert_allclose(result["params"][:-1], fit_fixtures[case_name]["params"], atol=1e-2, rtol=0)
        assert abs(result["params"][-1] - fit_fixtures[case_name]["scale"]) < 1e-2


@pytest.mark.parametrize("simple_diff", [False, True])
@pytest.mark.parametrize("enforce", [False, True])
@pytest.mark.parametrize("concentrate", [False, True])
def test_sarimax_exogenous_alias_parity(simple_diff, enforce, concentrate):
    rng = np.random.default_rng(324)
    x = rng.normal(size=(160, 1))
    noise = rng.normal(scale=0.5, size=160)
    for t in range(4, len(noise)):
        noise[t] += 0.25 * noise[t - 1] + 0.2 * noise[t - 4]
    y = 1.8 * x[:, 0] + noise
    if simple_diff:
        y = np.cumsum(y)
    kwargs = dict(exog=x, concentrate_scale=concentrate,
                  simple_differencing=simple_diff, enforce_stationarity=enforce,
                  enforce_invertibility=enforce, maxiter=500)
    order = (1, int(simple_diff), 0)
    default = rustima.sarimax_fit(y, order, (1, 0, 0, 4), method="lbfgsb", **kwargs)
    basin = rustima.sarimax_fit(y, order, (1, 0, 0, 4), method="basin-lbfgsb", **kwargs)
    assert basin["converged"]
    assert abs(basin["loglike"] - default["loglike"]) < 1e-4
    np.testing.assert_allclose(basin["params"], default["params"], atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("strategy", ["", "-strict", "-multi", "-adaptive", "-hybrid"])
@pytest.mark.parametrize("budget", [0, 1, 5, 50])
def test_work_limits(statsmodels_fixtures, strategy, budget):
    y = np.asarray(statsmodels_fixtures["arma11"]["data"])
    result = rustima.sarimax_fit(y, (1, 0, 1), (0, 0, 0, 0),
                                start_params=np.array([0.1, 0.1, 1.0]),
                                method=f"basin-lbfgsb{strategy}", maxiter=budget)
    assert result["n_iter"] <= budget
    assert np.isfinite(result["loglike"])
    if budget <= 1:
        assert not result["converged"]


def test_high_level_and_batch_fits(statsmodels_fixtures):
    y = np.asarray(statsmodels_fixtures["ar1"]["data"])
    result = SARIMAXModel(y, order=(1, 0, 0)).fit(method="basin-lbfgsb")
    assert result.converged
    assert "basin-lbfgsb" in result.method
    fits = rustima.sarimax_batch_fit([y, y.copy()], (1, 0, 0), (0, 0, 0, 0),
                                    method="basin-lbfgsb")
    assert all(fit["converged"] for fit in fits)
    np.testing.assert_allclose(fits[0]["params"], fits[1]["params"], atol=0, rtol=0)


@pytest.mark.parametrize("stepwise", [False, True])
@pytest.mark.parametrize("with_exog", [False, True])
def test_auto_arima_selection(statsmodels_fixtures, stepwise, with_exog):
    y = np.asarray(statsmodels_fixtures["ar1"]["data"])
    kwargs = dict(max_p=2, max_q=1, d=0, D=0, s=0, stepwise=stepwise)
    if with_exog:
        kwargs["exog"] = np.sin(np.arange(len(y)) * 0.2).reshape(-1, 1)
    default = auto_arima(y, method="lbfgsb", **kwargs)
    basin = auto_arima(y, method="basin-lbfgsb", **kwargs)
    assert basin.result is not None
    assert basin.order == default.order
    assert basin.seasonal_order == default.seasonal_order
    assert abs(basin.best_ic - default.best_ic) < 1e-3
    assert "basin-lbfgsb" in basin.result.method
