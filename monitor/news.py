"""News ingestion via Alpha Vantage NEWS_SENTIMENT.

A single request with time_from/time_to backfills a whole date range (up to 1000
articles), so one API call typically covers a month. Alpha Vantage ships its own
sentiment scores — we ignore them; our DistilBERT does the scoring. We only use its
per-ticker `relevance_score` to drop articles that merely mention the ticker in
passing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import requests

from .config import (
    ALPHA_VANTAGE_API_KEY,
    ALPHA_VANTAGE_URL,
    RELEVANCE_THRESHOLD,
)


@dataclass
class Headline:
    ticker: str
    text: str
    url: str | None
    published_date: str  # ISO date YYYY-MM-DD
    source: str | None


def _to_av_timestamp(date_str: str, end_of_day: bool = False) -> str:
    """Convert ISO date (YYYY-MM-DD) to Alpha Vantage's YYYYMMDDTHHMM format."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return d.strftime("%Y%m%dT2359" if end_of_day else "%Y%m%dT0000")


class AlphaVantageNewsSource:
    """Fetches ticker headlines from Alpha Vantage within a date range."""

    def __init__(self, api_key: str = ALPHA_VANTAGE_API_KEY, timeout: int = 30):
        if not api_key:
            raise ValueError(
                "ALPHA_VANTAGE_API_KEY is not set. Add it to your .env file "
                "(see .env.example)."
            )
        self.api_key = api_key
        self.timeout = timeout

    def fetch(self, ticker: str, start: str, end: str, limit: int = 1000) -> list[Headline]:
        """Return relevant headlines for `ticker` between ISO dates [start, end]."""
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": _to_av_timestamp(start),
            "time_to": _to_av_timestamp(end, end_of_day=True),
            "limit": limit,
            "sort": "EARLIEST",
            "apikey": self.api_key,
        }
        resp = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()

        # Alpha Vantage signals rate limits / errors as a plain message field.
        if "feed" not in payload:
            note = payload.get("Note") or payload.get("Information") or payload.get(
                "Error Message"
            ) or str(payload)
            raise RuntimeError(f"Alpha Vantage returned no feed: {note}")

        headlines: list[Headline] = []
        for item in payload["feed"]:
            if not self._is_relevant(item, ticker):
                continue
            published = item.get("time_published", "")  # YYYYMMDDTHHMMSS
            if len(published) < 8:
                continue
            iso_date = f"{published[0:4]}-{published[4:6]}-{published[6:8]}"
            headlines.append(
                Headline(
                    ticker=ticker,
                    text=item.get("title", "").strip(),
                    url=item.get("url"),
                    published_date=iso_date,
                    source=item.get("source"),
                )
            )
        return [h for h in headlines if h.text]

    @staticmethod
    def _is_relevant(item: dict, ticker: str) -> bool:
        """Keep only items whose relevance score for this ticker clears the bar."""
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() == ticker.upper():
                try:
                    return float(ts.get("relevance_score", 0)) >= RELEVANCE_THRESHOLD
                except (TypeError, ValueError):
                    return False
        return False
