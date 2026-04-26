"""
Data loaders for the macro-regime validation work.

Pulls the same FRED series the deployed model uses (via ``regime_model.FRED_IDS``)
plus yfinance asset returns for the predictive-validity tests. Cached to a local
parquet directory so re-running the notebook is fast and offline-friendly.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import regime_model as rm

CACHE_DIR = Path(__file__).resolve().parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Core 4 asset universe for forward-return regressions (Yahoo tickers).
ASSET_TICKERS = {
    "SPY": "SPY",       # US equities
    "AGG": "AGG",       # Aggregate bond / duration proxy
    "GLD": "GLD",       # Gold / real assets
    "DXY": "DX-Y.NYB",  # US dollar index
}


# ── FRED ──────────────────────────────────────────────────────────────
def _fred_api_key() -> str | None:
    # 1. Environment variable
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key

    # 2. Search every plausible secrets.toml location
    candidates = [
        Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml",  # project root
        Path.cwd() / ".streamlit" / "secrets.toml",                               # current working dir
        Path.home() / ".streamlit" / "secrets.toml",                              # home dir
    ]
    for f in candidates:
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("FRED_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def fetch_fred(series_id: str, start: str = rm.DEFAULT_START) -> pd.Series:
    """Fetch a FRED series, caching to parquet."""
    cache = CACHE_DIR / f"{series_id}.parquet"
    if cache.exists():
        s = pd.read_parquet(cache).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s.loc[start:]

    api_key = _fred_api_key()
    if not api_key:
        from pathlib import Path as _P
        looked = [
            str(_P(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"),
            str(_P.cwd() / ".streamlit" / "secrets.toml"),
            str(_P.home() / ".streamlit" / "secrets.toml"),
        ]
        raise RuntimeError(
            "FRED_API_KEY not found.\n"
            "Looked in:\n" + "\n".join(f"  {p}" for p in looked) +
            "\n\nMake sure secrets.toml contains:  FRED_API_KEY = \"your_key\""
        )
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs:
        return pd.Series(dtype=float, name=series_id)
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("date")["value"].dropna()
    s.name = series_id
    s.to_frame().to_parquet(cache)
    return s


def fetch_all_fred(start: str = rm.DEFAULT_START) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for sid in rm.FRED_IDS + ["USREC"]:
        s = fetch_fred(sid, start=start)
        if len(s) > 0:
            out[sid] = s
        time.sleep(0.05)  # be polite to FRED
    return out


# ── yfinance asset returns ────────────────────────────────────────────
def fetch_yf_prices(tickers: dict[str, str] | None = None,
                    start: str = "1990-01-01") -> pd.DataFrame:
    """Monthly close prices for the asset panel."""
    import yfinance as yf

    tickers = tickers or ASSET_TICKERS
    cache = CACHE_DIR / "yf_prices.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index)
        return df

    frames = {}
    for label, sym in tickers.items():
        hist = yf.download(sym, start=start, progress=False, auto_adjust=True)
        if hist is None or len(hist) == 0:
            continue
        # yfinance occasionally returns a MultiIndex on columns.
        close = hist["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        frames[label] = close.resample("ME").last()
    df = pd.DataFrame(frames).dropna(how="all")
    df.to_parquet(cache)
    return df


def forward_log_returns(prices: pd.DataFrame, horizons=(1, 3, 6, 12)) -> dict[int, pd.DataFrame]:
    """For each horizon k, return a DataFrame of forward k-month log returns.

    forward_return[t] = log(P[t+k] / P[t]) — i.e., the return realized AFTER t.
    """
    log_p = np.log(prices)
    out = {}
    for k in horizons:
        out[k] = (log_p.shift(-k) - log_p)
    return out


# ── Combined panel ────────────────────────────────────────────────────
def build_panel(start: str = rm.DEFAULT_START) -> dict:
    """Run the full data pipeline: FRED → regime → asset returns → aligned panel."""
    fred = fetch_all_fred(start=start)
    usrec = fred.pop("USREC", None)

    regime_out = rm.compute_regime(fred)
    combined = regime_out["combined"]

    prices = fetch_yf_prices()
    fwd = forward_log_returns(prices)

    # Align everything to month-end.
    combined.index = pd.to_datetime(combined.index).to_period("M").to_timestamp("M")
    if usrec is not None:
        usrec = usrec.resample("ME").last()
        usrec.index = usrec.index.to_period("M").to_timestamp("M")
    for k in fwd:
        fwd[k].index = pd.to_datetime(fwd[k].index).to_period("M").to_timestamp("M")
    prices.index = pd.to_datetime(prices.index).to_period("M").to_timestamp("M")

    return {
        "fred": fred,
        "regime": regime_out,
        "scores": combined[["Growth", "Inflation", "Liquidity"]],
        "labels": combined[["Growth_Class", "Inflation_Class", "Liquidity_Class", "Regime"]],
        "usrec": usrec,
        "prices": prices,
        "forward_returns": fwd,
    }
