"""Mean (ARIMA) and variance (Garch-family ) models"""
"""Mean (ARIMA) and variance (GARCH-family) models."""

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


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