"""Mean (ARIMA) and variance (Garch-family ) models"""
"""Mean (ARIMA) and variance (GARCH-family) models."""

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


def arima_order_selection(y: np.ndarray, max_p: int = 3, max_q: int = 3) -> pd.DataFrame:
    """Fit ARMA(p, q) models with a constant and report AIC and BIC."""
    rows = []
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence notices on large orders
            res = ARIMA(y, order=(p, 0, q), trend="c").fit()
        rows.append({"p": p, "q": q, "aic": res.aic, "bic": res.bic})
    return pd.DataFrame(rows).set_index(["p", "q"])


def one_step_forecasts(y: np.ndarray, order: tuple, train_size: int) -> np.ndarray:
    """One-step-ahead mean forecasts on the test period.

    Parameters are estimated on the training period only, then kept fixed.
    """
    train, test = y[:train_size], y[train_size:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(train, order=order, trend="c").fit()
        res_all = res.append(test, refit=False)
    return res_all.fittedvalues[train_size:]

GARCH_SPECS = {
    "GARCH": dict(vol="GARCH", p=1, o=0, q=1),
    "GJR-GARCH": dict(vol="GARCH", p=1, o=1, q=1),
    "EGARCH": dict(vol="EGARCH", p=1, o=1, q=1),
}


def fit_garch(y: pd.Series, model: str = "GARCH", dist: str = "normal"):
    """Fit a GARCH-family model with a constant mean (returns in percent)."""
    am = arch_model(y, mean="Constant", dist=dist, rescale=False, **GARCH_SPECS[model])
    return am.fit(disp="off")


def persistence(res, model: str) -> float:
    """Persistence of a shock in the variance equation."""
    p = res.params
    if model == "GARCH":
        return p["alpha[1]"] + p["beta[1]"]
    if model == "GJR-GARCH":
        return p["alpha[1]"] + 0.5 * p["gamma[1]"] + p["beta[1]"]
    return p["beta[1]"]  # EGARCH


def half_life(pers: float) -> float:
    """Number of days for the effect of a shock to decrease by half."""
    return np.log(0.5) / np.log(pers)