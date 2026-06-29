"""Cron-ready CLI to ingest + score headlines for a ticker.

Examples:
    python -m monitor.run_daily --ticker AAPL                 # last 30 days
    python -m monitor.run_daily --ticker MSFT --start 2024-05-01 --end 2024-05-31

Cron (run every weekday at 18:00, accumulating the trailing window):
    0 18 * * 1-5  cd /path/to/repo && /opt/anaconda3/bin/python -m monitor.run_daily --ticker AAPL
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

from .classifier import FinancialSentimentClassifier
from .pipeline import run_monitor


def main() -> None:
    today = date.today()
    parser = argparse.ArgumentParser(description="Ingest and score financial news for a ticker.")
    parser.add_argument("--ticker", required=True, help="Stock ticker, e.g. AAPL")
    parser.add_argument(
        "--start", default=(today - timedelta(days=30)).strftime("%Y-%m-%d"),
        help="ISO start date (default: 30 days ago)",
    )
    parser.add_argument(
        "--end", default=today.strftime("%Y-%m-%d"),
        help="ISO end date (default: today)",
    )
    args = parser.parse_args()

    classifier = FinancialSentimentClassifier()
    result = run_monitor(args.ticker, args.start, args.end, classifier=classifier, refresh=True)

    s = result.stats
    print(f"[{result.ticker}] {result.start} → {result.end}")
    print(f"  headlines cached : {s['headlines_total']}")
    print(f"  newly classified : {s['headlines_newly_scored']}")
    print(f"  trading days     : {s['trading_days']}")
    print(f"  divergent days   : {s['divergent_days']}")
    for d in result.divergences:
        print(f"    ! {d['date']}  {d['note']}  (sent z={d['sentiment_z']}, price z={d['price_z']})")


if __name__ == "__main__":
    main()
