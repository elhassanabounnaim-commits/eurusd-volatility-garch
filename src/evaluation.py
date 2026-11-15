"""Forecast evaluation: loss functions and Diebold-Mariano test."""
"""Statistical tests and forecast evaluation."""

import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
import warnings


def distribution_summary(x: pd.Series) -> pd.Series:
    """Moments of a series and Jarque-Bera normality test."""
    jb_stat, jb_pvalue = stats.jarque_bera(x)
    return pd.Series({
        "mean": x.mean(),
        "std": x.std(),
        "skewness": stats.skew(x),
        "excess_kurtosis": stats.kurtosis(x),  # normal distribution = 0
        "jarque_bera": jb_stat,
        "jb_pvalue": jb_pvalue,
    })


def ljung_box(x: pd.Series, lags=(10, 20)) -> pd.DataFrame:
    """Ljung-Box test of no autocorrelation up to each lag."""
    return acorr_ljungbox(x, lags=list(lags))


def stationarity_tests(x: pd.Series) -> pd.Series:
    """ADF (H0: unit root) and KPSS (H0: stationary) tests, with a constant.

    KPSS p-values are bounded to [0.01, 0.10] by its lookup table.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_pvalue, *_ = adfuller(x, regression="c", autolag="AIC")
        kpss_stat, kpss_pvalue, *_ = kpss(x, regression="c", nlags="auto")
    return pd.Series({
        "adf_stat": adf_stat,
        "adf_pvalue": adf_pvalue,
        "kpss_stat": kpss_stat,
        "kpss_pvalue": kpss_pvalue,
    })