"""SQLite persistence for scored headlines.

One table holds each headline plus its model score. A UNIQUE(ticker, dedup_key)
constraint means a headline is only ever classified once — re-running the pipeline
skips already-seen items, preserving the Alpha Vantage 25/day budget.
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS headline_scores (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT    NOT NULL,
    text          TEXT    NOT NULL,
    url           TEXT,
    published_date TEXT   NOT NULL,           -- ISO date (YYYY-MM-DD)
    source        TEXT,
    label         TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    model_version TEXT    NOT NULL,
    dedup_key     TEXT    NOT NULL,
    created_at    TEXT    DEFAULT (datetime('now')),
    UNIQUE(ticker, dedup_key)
);
CREATE INDEX IF NOT EXISTS idx_ticker_date ON headline_scores(ticker, published_date);
"""


@dataclass
class ScoredHeadline:
    ticker: str
    text: str
    url: str | None
    published_date: str
    source: str | None
    label: str
    confidence: float
    model_version: str

    @property
    def dedup_key(self) -> str:
        """Stable key: the URL when present, else a hash of the headline text."""
        if self.url:
            return self.url
        return "h:" + hashlib.sha1(self.text.encode("utf-8")).hexdigest()


@contextmanager
def _connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


def existing_keys(ticker: str, db_path=DB_PATH) -> set[str]:
    """Return the dedup_keys already stored for a ticker (to skip re-classifying)."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT dedup_key FROM headline_scores WHERE ticker = ?", (ticker,)
        ).fetchall()
    return {r["dedup_key"] for r in rows}


def insert_scored(rows: list[ScoredHeadline], db_path=DB_PATH) -> int:
    """Insert scored headlines, ignoring duplicates. Returns rows actually inserted."""
    if not rows:
        return 0
    init_db(db_path)
    with _connect(db_path) as conn:
        before = conn.total_changes
        conn.executemany(
            """
            INSERT OR IGNORE INTO headline_scores
                (ticker, text, url, published_date, source, label, confidence,
                 model_version, dedup_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.ticker, r.text, r.url, r.published_date, r.source,
                    r.label, r.confidence, r.model_version, r.dedup_key,
                )
                for r in rows
            ],
        )
        return conn.total_changes - before


def fetch_scored(ticker: str, start: str, end: str, db_path=DB_PATH) -> list[dict]:
    """Return stored scored headlines for a ticker within [start, end] (ISO dates)."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT text, url, published_date, source, label, confidence
            FROM headline_scores
            WHERE ticker = ? AND published_date BETWEEN ? AND ?
            ORDER BY published_date
            """,
            (ticker, start, end),
        ).fetchall()
    return [dict(r) for r in rows]
