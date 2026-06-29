"""Aggregate scored headlines into a daily sentiment index.

Per headline, a signed score s = LABEL_SIGN[label] * confidence (positive +1,
neutral 0, negative -1). The daily index is the mean of s over that day's
headlines, giving a confidence-weighted net sentiment in roughly [-1, 1]. Daily
class counts are kept alongside for transparency.
"""
from __future__ import annotations

import pandas as pd

from .config import LABEL_SIGN


def daily_sentiment_index(scored: list[dict]) -> pd.DataFrame:
    """Collapse headline-level scores into one row per calendar date.

    Input rows need: published_date, label, confidence.
    Returns columns: date, sentiment_index, n_total, n_pos, n_neu, n_neg.
    """
    cols = ["date", "sentiment_index", "n_total", "n_pos", "n_neu", "n_neg"]
    if not scored:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(scored)
    df["signed"] = df.apply(
        lambda r: LABEL_SIGN[r["label"]] * float(r["confidence"]), axis=1
    )

    grouped = df.groupby("published_date")
    out = grouped.agg(
        sentiment_index=("signed", "mean"),
        n_total=("label", "size"),
        n_pos=("label", lambda s: (s == "positive").sum()),
        n_neu=("label", lambda s: (s == "neutral").sum()),
        n_neg=("label", lambda s: (s == "negative").sum()),
    ).reset_index().rename(columns={"published_date": "date"})

    out["sentiment_index"] = out["sentiment_index"].round(4)
    return out[cols].sort_values("date").reset_index(drop=True)
