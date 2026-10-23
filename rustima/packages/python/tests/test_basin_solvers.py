"""Regression checks for the unconstrained Basin solvers."""

import numpy as np
import pytest

import rustima


@pytest.mark.parametrize("method", ["lbfgs", "bfgs", "nm"])
@pytest.mark.parametrize("name", ["ar1", "arma11", "arima111", "sarima_100_100_4"])
@pytest.mark.parametrize("concentrate", [False, True])
def test_basin_solver_matches_reference(method, name, concentrate,
                                        statsmodels_fixtures, fit_fixtures):
    case = statsmodels_fixtures[name]
    y = np.asarray(case["data"])
    order, seasonal = tuple(case["order"]), tuple(case["seasonal_order"])
    fit = rustima.sarimax_fit(y, order, seasonal, method=method,
                             concentrate_scale=concentrate, maxiter=1000)
    assert fit["converged"]
    assert fit["n_iter"] <= 1000
    assert abs(fit["loglike"] - fit_fixtures[name]["loglike"]) < 1e-4
    assert len(fit["params"]) == sum(order[::2]) + sum(seasonal[:3:2]) + (not concentrate)
    ll = rustima.sarimax_loglike(y, order, seasonal, np.asarray(fit["params"]),
                                concentrate_scale=concentrate)
    assert abs(ll - fit["loglike"]) < 1e-8


@pytest.mark.parametrize("method", ["lbfgs", "bfgs", "nm"])
@pytest.mark.parametrize("budget", [0, 1, 5])
def test_basin_solver_respects_iteration_limit(method, budget, statsmodels_fixtures):
    y = np.asarray(statsmodels_fixtures["arma11"]["data"])
    fit = rustima.sarimax_fit(y, (1, 0, 1), (0, 0, 0, 0), method=method,
                             start_params=np.array([0.1, 0.1, 1.0]), maxiter=budget)
    assert np.isfinite(fit["loglike"])
    assert fit["n_iter"] <= budget
    if budget <= 1:
        assert not fit["converged"]


@pytest.mark.parametrize("method", ["lbfgs", "bfgs", "nm"])
def test_basin_solver_batch_matches_single(method):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(140, 1))
    y = 1.5 * x[:, 0] + rng.normal(size=len(x))
    args = dict(method=method, maxiter=500)
    single = rustima.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0), exog=x, **args)
    batch = rustima.sarimax_batch_fit([y, y.copy()], (1, 0, 0), (0, 0, 0, 0),
                                    exog_list=[x, x.copy()], **args)
    assert single["converged"]
    for fit in batch:
        assert fit["converged"]
        np.testing.assert_array_equal(fit["params"], single["params"])
        assert fit["loglike"] == single["loglike"]
