"""End-to-end orchestration: news -> dedup -> classify -> store -> aggregate.

`run_monitor` is the single entry point used by both the CLI and the API. It is
cache-first: only headlines not already in SQLite are fetched-and-classified, so
repeat runs over the same window cost zero API budget and zero re-inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .aggregate import daily_sentiment_index
from .classifier import FinancialSentimentClassifier
from .config import MODEL_VERSION
from .divergence import align_and_flag
from .news import AlphaVantageNewsSource, Headline
from .prices import fetch_prices
from .storage import ScoredHeadline, existing_keys, fetch_scored, insert_scored


@dataclass
class MonitorResult:
    ticker: str
    start: str
    end: str
    sentiment_series: list[dict]
    price_series: list[dict]
    divergences: list[dict]
    stats: dict = field(default_factory=dict)


def ingest_and_score(
    ticker: str,
    start: str,
    end: str,
    classifier: FinancialSentimentClassifier,
    news_source: AlphaVantageNewsSource | None = None,
) -> int:
    """Fetch new headlines, classify the unseen ones, store them. Returns # newly scored."""
    source = news_source or AlphaVantageNewsSource()
    headlines: list[Headline] = source.fetch(ticker, start, end)

    seen = existing_keys(ticker)
    # Re-derive dedup_key via ScoredHeadline so it matches storage's hashing exactly.
    unseen: list[Headline] = []
    for h in headlines:
        probe = ScoredHeadline(
            ticker, h.text, h.url, h.published_date, h.source, "", 0.0, MODEL_VERSION
        )
        if probe.dedup_key not in seen:
            unseen.append(h)

    if not unseen:
        return 0

    preds = classifier.predict([h.text for h in unseen])
    rows = [
        ScoredHeadline(
            ticker=h.ticker,
            text=h.text,
            url=h.url,
            published_date=h.published_date,
            source=h.source,
            label=p.label,
            confidence=p.confidence,
            model_version=MODEL_VERSION,
        )
        for h, p in zip(unseen, preds)
    ]
    return insert_scored(rows)


def run_monitor(
    ticker: str,
    start: str,
    end: str,
    classifier: FinancialSentimentClassifier,
    refresh: bool = True,
    news_source: AlphaVantageNewsSource | None = None,
) -> MonitorResult:
    """Run the full monitor for a ticker/date-range and return aligned series."""
    ticker = ticker.upper()
    newly_scored = 0
    if refresh:
        newly_scored = ingest_and_score(ticker, start, end, classifier, news_source)

    scored = fetch_scored(ticker, start, end)
    sentiment = daily_sentiment_index(scored)
    prices = fetch_prices(ticker, start, end)
    merged, divergences = align_and_flag(sentiment, prices)

    return MonitorResult(
        ticker=ticker,
        start=start,
        end=end,
        sentiment_series=sentiment.to_dict(orient="records"),
        price_series=merged[["date", "close", "return", "sentiment_index", "n_total"]]
        .to_dict(orient="records"),
        divergences=divergences,
        stats={
            "headlines_total": len(scored),
            "headlines_newly_scored": newly_scored,
            "trading_days": int(len(prices)),
            "divergent_days": len(divergences),
        },
    )
