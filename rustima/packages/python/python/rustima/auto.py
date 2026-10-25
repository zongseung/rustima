"""Auto ARIMA using the shared Rust differencing and search engine.

Differencing selection always uses ADF and the seasonal variance-ratio rule;
it no longer switches to a variance heuristic when statsmodels is absent.
"""

import math

import numpy as np

from .model import SARIMAXModel, SARIMAXResult


def _ndiffs(y, max_d=2, test="adf"):
    from . import rustima as native
    return native._difference_orders(np.asarray(y, dtype=np.float64), max_d, 0, 0)[0]


def _nsdiffs(y, s, max_D=1):
    from . import rustima as native
    return native._difference_orders(np.asarray(y, dtype=np.float64), 0, max_D, s)[1]


def _ic(result, criterion="aic"):
    """Extract information criterion from fit result."""
    if criterion == "aic":
        return result.aic
    elif criterion == "bic":
        return result.bic
    elif criterion == "hqic":
        return result.hqic
    raise ValueError(f"Unknown criterion: {criterion}")


def _trace_print(p, d, q, P, D, Q, s, criterion, ic_val):
    """Print a trace line for auto_arima model evaluation."""
    status = f"{ic_val:.3f}" if math.isfinite(ic_val) else "FAILED"
    print(f"  ARIMA({p},{d},{q})({P},{D},{Q})[{s}] : {criterion}={status}")


def auto_arima(
    endog,
    exog=None,
    max_p=5,
    max_q=5,
    max_d=2,
    max_P=2,
    max_Q=2,
    max_D=1,
    s=0,
    d=None,
    D=None,
    trend="n",
    criterion="aic",
    stepwise=True,
    enforce_stationarity=True,
    enforce_invertibility=True,
    method=None,
    maxiter=None,
    simple_differencing=False,
    trace=False,
):
    """Automatic ARIMA order selection.

    Parameters
    ----------
    endog : array_like
        Time series.
    max_p, max_q : int
        Maximum AR and MA orders.
    max_d : int
        Maximum differencing order (auto-detected if d is None).
    max_P, max_Q : int
        Maximum seasonal AR and MA orders.
    max_D : int
        Maximum seasonal differencing.
    s : int
        Seasonal period (0 = non-seasonal).
    d : int or None
        Differencing order. If None, Rust selects it using a constant-regression
        ADF test at 5% with AIC lag selection, independently of statsmodels installation.
    D : int or None
        Seasonal differencing. If None, a seasonal variance ratio below 0.64
        selects one difference, otherwise zero.
    trend : str
        Trend specification: 'n', 'c', 't', 'ct'.
    criterion : str
        'aic', 'bic', or 'hqic'.
    stepwise : bool
        If True, use Hyndman-Khandakar stepwise (fast).
        If False, exhaustive grid search (slow but thorough).
    enforce_stationarity : bool
    enforce_invertibility : bool
    method : str or None
    maxiter : int or None
    trace : bool
        If True, print evaluated models after the Rust search returns.

    Returns
    -------
    AutoARIMAResult
        Best fit result with search history.
    """
    if np.ma.isMaskedArray(endog) or np.ma.isMaskedArray(exog):
        raise ValueError("masked arrays are not supported; fill or drop masked values first")
    endog = np.asarray(endog, dtype=np.float64)
    if exog is not None:
        exog = np.asarray(exog, dtype=np.float64)
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
    from . import rustima as native
    raw = native._auto_arima(
        endog, (max_p, max_d, max_q), (max_P, max_D, max_Q, s),
        d=d, seasonal_d=D, exog=exog, trend=trend, criterion=criterion,
        stepwise=stepwise, enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        simple_differencing=simple_differencing, method=method, maxiter=maxiter,
    )
    history = raw["history"]
    if trace:
        for entry in history:
            p, diff, q = entry["order"]
            P, seasonal_diff, Q, period = entry["seasonal_order"]
            _trace_print(p, diff, q, P, seasonal_diff, Q, period,
                         criterion, entry[criterion] if entry[criterion] is not None else math.inf)
    result = None
    if raw["fit"] is not None:
        fit = raw["fit"]
        model = SARIMAXModel(
            endog, order=fit["order"], seasonal_order=fit["seasonal_order"],
            exog=exog, trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            simple_differencing=simple_differencing,
        )
        model._fit_result = True
        result = SARIMAXResult(model, fit)
    return AutoARIMAResult(result, history, criterion)


class AutoARIMAResult:
    """Result of auto_arima search.

    Attributes
    ----------
    result : SARIMAXResult or None
        Best model fit result.
    history : list[dict]
        Search history with order, seasonal_order, and IC for each model.
    criterion : str
        IC used for selection.
    """

    def __init__(self, result, history, criterion):
        self.result = result
        self.history = history
        self.criterion = criterion

    @property
    def order(self):
        """Best model order (p, d, q)."""
        return self.result.model.order if self.result else None

    @property
    def seasonal_order(self):
        """Best model seasonal order (P, D, Q, s)."""
        return self.result.model.seasonal_order if self.result else None

    @property
    def best_ic(self):
        """Best information criterion value."""
        return _ic(self.result, self.criterion) if self.result else math.inf

    def summary(self, alpha=0.05, inference=None):
        """Return statsmodels-style summary of the best model.

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals.
        inference : str, optional
            ``"none"`` | ``"hessian"`` | ``"opg"`` | ``"both"``.
            Default ``"hessian"``.
        """
        if self.result is None:
            return "auto_arima: no valid model found."
        if inference is None:
            inference = "hessian"
        n_models = len(self.history)
        n_converged = sum(1 for h in self.history if h["converged"])
        body = self.result.summary(alpha=alpha, inference=inference)
        footer = (
            f"Search: {n_models} models evaluated "
            f"({n_converged} converged), criterion={self.criterion}"
        )
        return body + "\n" + footer

    def search_summary(self):
        """Short summary of the search (order, IC, model count)."""
        if self.result is None:
            return "auto_arima: no valid model found."
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        ic = self.best_ic
        n_models = len(self.history)
        n_converged = sum(1 for h in self.history if h["converged"])
        return (
            f"auto_arima: Best ARIMA({p},{d},{q})({P},{D},{Q})[{s}]\n"
            f"  {self.criterion}={ic:.3f}\n"
            f"  Models evaluated: {n_models} ({n_converged} converged)"
        )

    def history_dataframe(self):
        """Return search history as a Polars DataFrame."""
        import polars as pl

        orders = [str(h["order"]) for h in self.history]
        seasonals = [str(h["seasonal_order"]) for h in self.history]
        ics = [h.get(self.criterion) for h in self.history]
        converged = [h["converged"] for h in self.history]
        return pl.DataFrame({
            "order": orders,
            "seasonal_order": seasonals,
            self.criterion: ics,
            "converged": converged,
        })
