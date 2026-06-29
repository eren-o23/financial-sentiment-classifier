"""SQLite persistence for eval runs (regression history).

Lives in the same `monitor.db` as the headline cache, in its own `eval_runs`
table, following the conventions in `monitor/storage.py`: a SCHEMA string applied
idempotently via `init_db()`, a `_connect` context manager with `sqlite3.Row`, and
SELECTs returned as `list[dict]`. The `headline_scores` table is never touched.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at         TEXT    DEFAULT (datetime('now')),
    model_version      TEXT    NOT NULL,
    golden_size        INTEGER NOT NULL,
    accuracy           REAL    NOT NULL,
    macro_f1           REAL    NOT NULL,
    weighted_f1        REAL,
    consistency_passed INTEGER NOT NULL,   -- 0/1
    prev_run_id        INTEGER,
    macro_f1_delta     REAL,               -- vs previous run, NULL on first run
    regression_flagged INTEGER NOT NULL,   -- 0/1
    report_json        TEXT    NOT NULL    -- full structured EvalReport
);
"""

# Scalar columns surfaced in the history view (everything except the big JSON blob).
_HISTORY_COLS = (
    "id, created_at, model_version, golden_size, accuracy, macro_f1, weighted_f1, "
    "consistency_passed, prev_run_id, macro_f1_delta, regression_flagged"
)


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


def latest_run(db_path=DB_PATH) -> dict | None:
    """Return the most recent eval run (scalar columns only), or None if no runs."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            f"SELECT {_HISTORY_COLS} FROM eval_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def insert_run(
    *,
    model_version: str,
    golden_size: int,
    accuracy: float,
    macro_f1: float,
    weighted_f1: float | None,
    consistency_passed: bool,
    prev_run_id: int | None,
    macro_f1_delta: float | None,
    regression_flagged: bool,
    report: dict,
    db_path=DB_PATH,
) -> int:
    """Insert one eval run and return its new row id."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO eval_runs
                (model_version, golden_size, accuracy, macro_f1, weighted_f1,
                 consistency_passed, prev_run_id, macro_f1_delta, regression_flagged,
                 report_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_version,
                golden_size,
                accuracy,
                macro_f1,
                weighted_f1,
                int(consistency_passed),
                prev_run_id,
                macro_f1_delta,
                int(regression_flagged),
                json.dumps(report),
            ),
        )
        return int(cur.lastrowid)


def fetch_history(limit: int = 20, db_path=DB_PATH) -> list[dict]:
    """Return recent eval runs (newest first), scalar columns only."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT {_HISTORY_COLS} FROM eval_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
