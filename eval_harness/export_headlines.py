"""Export cached scored headlines to JSONL for error-analysis annotation.

The output feeds the `error-discovery` skill, which clusters the headlines and
selects a diverse sample for manual ground-truth labelling. Each record carries
the model's prediction at selection time so the annotator can see (and disagree
with) what the model thought.

    python -m eval_harness.export_headlines [--out PATH] [--ticker AAPL ...]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from .config import DB_PATH, PROJECT_ROOT

DEFAULT_OUT = PROJECT_ROOT / "eval_harness" / "golden" / "headlines_export.jsonl"


def export(out_path: Path, tickers: list[str] | None = None, db_path=DB_PATH) -> int:
    """Write all (or selected tickers') scored headlines to JSONL. Returns count."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = (
            "SELECT ticker, text, url, published_date, source, label, confidence "
            "FROM headline_scores"
        )
        params: tuple = ()
        if tickers:
            placeholders = ",".join("?" for _ in tickers)
            query += f" WHERE ticker IN ({placeholders})"
            params = tuple(t.upper() for t in tickers)
        query += " ORDER BY ticker, published_date"
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(
                json.dumps(
                    {
                        "text": r["text"],
                        "ticker": r["ticker"],
                        "published_date": r["published_date"],
                        "source": r["source"],
                        "url": r["url"],
                        "model_prediction_at_selection": r["label"],
                        "confidence_at_selection": r["confidence"],
                    }
                )
                + "\n"
            )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export scored headlines to JSONL.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path.")
    parser.add_argument(
        "--ticker", action="append", default=None,
        help="Restrict to ticker(s); repeatable. Default: all tickers.",
    )
    args = parser.parse_args()
    n = export(Path(args.out), tickers=args.ticker)
    print(f"Exported {n} headlines to {args.out}")


if __name__ == "__main__":
    main()
