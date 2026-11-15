"""Data loading and preprocessing for EUR/USD exchange rates."""
"""Data loading and preprocessing for EUR/USD exchange rates."""

from pathlib import Path

import numpy as np
import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU"
RAW_PATH = Path("data/raw/DEXUSEU.csv")


def download_fred(path: Path = RAW_PATH, force: bool = False) -> None:
    """Download the full DEXUSEU series from FRED and save it as a CSV snapshot.

    The download is skipped if the file already exists, so the frozen
    snapshot is never overwritten by accident.
    """
    if path.exists() and not force:
        print(f"Snapshot already exists: {path}")
        return
    df = pd.read_csv(FRED_URL)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")


def load_prices(path: Path = RAW_PATH,
                start: str = "2011-01-01",
                end: str = "2025-12-31") -> pd.Series:
    """Load the snapshot and return a clean daily price series (USD per EUR)."""
    df = pd.read_csv(path)
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"])
    # Missing values (US holidays) are stored as empty cells or "."
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    prices = df.set_index("date")["price"].sort_index()
    prices = prices.loc[start:end].dropna()
    prices = prices[~prices.index.duplicated(keep="first")]
    prices.name = "EURUSD"
    return prices


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log-returns in percent: r_t = 100 * ln(P_t / P_{t-1})."""
    returns = 100 * np.log(prices).diff().dropna()
    returns.name = "log_return"
    return returns

