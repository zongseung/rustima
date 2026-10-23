"""Compare optional Basin fits with the vendored L-BFGS-B backend."""

import os

import numpy as np
import pytest

import rustima
from rustima import SARIMAXModel, auto_arima


@pytest.fixture(scope="module")
def basin_enabled():
    y = np.sin(np.arange(80) * 0.3)
    try:
        rustima.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0),
                            method="basin-lbfgsb", maxiter=0)
        enabled = True
    except RuntimeError as error:
        assert "--features basin" in str(error)
        enabled = False
    expected = os.environ.get("RUSTIMA_TEST_BASIN")
    if expected is not None:
        assert enabled == (expected == "1"), "Installed wheel has unexpected Basin support"
    return enabled


@pytest.fixture
def require_basin(basin_enabled):
    if not basin_enabled:
        pytest.skip("Requires Cargo feature basin")


def test_feature_availability(basin_enabled):
    if not basin_enabled:
        for order in [(0, 0, 0), (3, 0, 0)]:
            with pytest.raises(RuntimeError, match="--features basin"):
                rustima.sarimax_fit(np.arange(80, dtype=float), order, (0, 0, 0, 0),
                                    concentrate_scale=True, method="basin-lbfgsb", maxiter=0)
        y = np.arange(80, dtype=float)
        with pytest.raises(RuntimeError, match="--features basin"):
            rustima.sarimax_batch_fit([y], (1, 0, 0), (0, 0, 0, 0), method="basin-lbfgsb")
        with pytest.raises(RuntimeError, match="--features basin"):
            rustima.sarimax_grid_search(y, [(1, 0, 0)], [(0, 0, 0, 0)], method="basin-lbfgsb")
        for stepwise in [False, True]:
            with pytest.raises(RuntimeError, match="--features basin"):
                auto_arima(y, d=0, D=0, s=0, stepwise=stepwise, method="basin-lbfgsb")


@pytest.mark.parametrize("case_name", ["ar1", "arma11", "arima111", "sarima_100_100_4"])
@pytest.mark.parametrize("concentrate", [False, True])
def test_identical_start_parity(require_basin, statsmodels_fixtures, case_name, concentrate):
    case = statsmodels_fixtures[case_name]
    start = list(case["params"][:-1] if concentrate else case["params"])
    kwargs = dict(start_params=np.asarray(start), concentrate_scale=concentrate)
    y = np.asarray(case["data"])
    order, seasonal = tuple(case["order"]), tuple(case["seasonal_order"])
    legacy = rustima.sarimax_fit(y, order, seasonal, method="lbfgsb-strict", **kwargs)
    basin = rustima.sarimax_fit(y, order, seasonal, method="basin-lbfgsb-strict", **kwargs)
    assert basin["converged"] and legacy["converged"]
    assert abs(basin["loglike"] - legacy["loglike"]) <= 1e-4
    np.testing.assert_allclose(basin["params"], legacy["params"], atol=1e-3, rtol=1e-3)
    assert basin["method"] == "basin-lbfgsb-strict"
    assert set(basin) == set(legacy)


@pytest.mark.parametrize("case_name", ["ar1", "arma11", "arima111", "sarima_100_100_4"])
def test_reference_fit_with_automatic_initialization(require_basin, statsmodels_fixtures,
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
def test_sarimax_exogenous_parity(require_basin, simple_diff, enforce, concentrate):
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
    legacy = rustima.sarimax_fit(y, order, (1, 0, 0, 4), method="lbfgsb", **kwargs)
    basin = rustima.sarimax_fit(y, order, (1, 0, 0, 4), method="basin-lbfgsb", **kwargs)
    assert basin["converged"]
    assert abs(basin["loglike"] - legacy["loglike"]) < 1e-4
    np.testing.assert_allclose(basin["params"], legacy["params"], atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("strategy", ["", "-strict", "-multi", "-adaptive", "-hybrid"])
@pytest.mark.parametrize("budget", [0, 1, 5, 50])
def test_work_limits(require_basin, statsmodels_fixtures, strategy, budget):
    y = np.asarray(statsmodels_fixtures["arma11"]["data"])
    result = rustima.sarimax_fit(y, (1, 0, 1), (0, 0, 0, 0),
                                start_params=np.array([0.1, 0.1, 1.0]),
                                method=f"basin-lbfgsb{strategy}", maxiter=budget)
    assert result["n_iter"] <= budget
    assert np.isfinite(result["loglike"])
    if budget <= 1:
        assert not result["converged"]


def test_high_level_and_batch_fits(require_basin, statsmodels_fixtures):
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
def test_auto_arima_selection(require_basin, statsmodels_fixtures, stepwise, with_exog):
    y = np.asarray(statsmodels_fixtures["ar1"]["data"])
    kwargs = dict(max_p=2, max_q=1, d=0, D=0, s=0, stepwise=stepwise)
    if with_exog:
        kwargs["exog"] = np.sin(np.arange(len(y)) * 0.2).reshape(-1, 1)
    legacy = auto_arima(y, method="lbfgsb", **kwargs)
    basin = auto_arima(y, method="basin-lbfgsb", **kwargs)
    assert basin.result is not None
    assert basin.order == legacy.order
    assert basin.seasonal_order == legacy.seasonal_order
    assert abs(basin.best_ic - legacy.best_ic) < 1e-3
    assert "basin-lbfgsb" in basin.result.method
