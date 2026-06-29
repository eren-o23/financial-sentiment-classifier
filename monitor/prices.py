"""Daily price data via yfinance."""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily closing prices for [start, end] (ISO dates, end inclusive).

    Returns a DataFrame with columns: date (ISO str), close, return (pct change).
    Empty DataFrame if no data (e.g. invalid ticker or non-trading range).
    """
    # yfinance treats `end` as exclusive, so push it out by a day to include it.
    end_exclusive = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.Ticker(ticker).history(start=start, end=end_exclusive, auto_adjust=True)

    if raw.empty:
        return pd.DataFrame(columns=["date", "close", "return"])

    df = raw.reset_index()[["Date", "Close"]].rename(
        columns={"Date": "date", "Close": "close"}
    )
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["return"] = df["close"].pct_change()
    return df.reset_index(drop=True)
