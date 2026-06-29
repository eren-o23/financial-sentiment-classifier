"""Align sentiment with price and flag divergent days.

This is the core insight of the tool: days where market *language* and price move in
opposite directions. Sentiment news on non-trading days (weekends/holidays) is rolled
forward onto the next trading day so it lines up with a price observation. Both the
daily sentiment index and the daily price return are z-scored over the window; a day
is flagged when their signs oppose and both magnitudes clear the threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DIVERGENCE_MIN_OBS, DIVERGENCE_Z_THRESHOLD


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def align_and_flag(
    sentiment: pd.DataFrame,
    prices: pd.DataFrame,
    z_threshold: float = DIVERGENCE_Z_THRESHOLD,
) -> tuple[pd.DataFrame, list[dict]]:
    """Join sentiment onto trading days and flag divergences.

    sentiment: columns date, sentiment_index, n_total, ...
    prices:    columns date, close, return

    Returns (merged_df, divergences). `merged_df` has one row per trading day with
    sentiment rolled forward; `divergences` is a list of flagged days with z-scores.
    """
    empty_merged = pd.DataFrame(
        columns=["date", "close", "return", "sentiment_index", "n_total"]
    )
    if prices.empty:
        return empty_merged, []

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    trading_days = prices["date"].sort_values().reset_index(drop=True)

    # Roll each sentiment date forward to the first trading day >= that date.
    sent = sentiment.copy()
    if sent.empty:
        merged = prices.copy()
        merged["sentiment_index"] = np.nan
        merged["n_total"] = 0
        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
        return merged, []

    sent["date"] = pd.to_datetime(sent["date"])
    sent = sent.sort_values("date")
    sent["trading_day"] = pd.merge_asof(
        sent[["date"]], trading_days.to_frame(name="date"),
        on="date", direction="forward",
    )["date"].values

    # Confidence-weight by headline count when collapsing multiple calendar days
    # onto the same trading day.
    sent = sent.dropna(subset=["trading_day"])
    sent["weighted"] = sent["sentiment_index"] * sent["n_total"]
    rolled = sent.groupby("trading_day").agg(
        sentiment_index=("weighted", "sum"),
        n_total=("n_total", "sum"),
    ).reset_index()
    rolled["sentiment_index"] = rolled["sentiment_index"] / rolled["n_total"]
    rolled = rolled.rename(columns={"trading_day": "date"})

    merged = prices.merge(rolled, on="date", how="left")
    merged["n_total"] = merged["n_total"].fillna(0).astype(int)

    # Z-score over days that have both a return and sentiment observation.
    scored = merged.dropna(subset=["sentiment_index", "return"]).copy()
    divergences: list[dict] = []
    if len(scored) >= DIVERGENCE_MIN_OBS:
        scored["sentiment_z"] = _zscore(scored["sentiment_index"])
        scored["price_z"] = _zscore(scored["return"])
        for _, row in scored.iterrows():
            sz, pz = row["sentiment_z"], row["price_z"]
            opposed = np.sign(sz) != np.sign(pz) and sz != 0 and pz != 0
            if opposed and abs(sz) >= z_threshold and abs(pz) >= z_threshold:
                direction = (
                    "Negative language, price up"
                    if sz < 0
                    else "Positive language, price down"
                )
                divergences.append(
                    {
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "sentiment_z": round(float(sz), 2),
                        "price_z": round(float(pz), 2),
                        "note": direction,
                    }
                )

    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged, divergences
