"""Regression comparison: did macro F1 drop versus the previous stored run?

`compare` is called with the current run's macro F1 *before* the new run is
inserted, so `latest_run()` returns the genuine predecessor.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import REGRESSION_F1_DROP_THRESHOLD
from .storage import latest_run


@dataclass
class RegressionResult:
    prev_run_id: int | None
    prev_macro_f1: float | None
    delta: float | None          # current - previous; None on the first run
    flagged: bool
    threshold: float


def compare(
    current_macro_f1: float,
    threshold: float = REGRESSION_F1_DROP_THRESHOLD,
    db_path=None,
) -> RegressionResult:
    """Compare the current macro F1 against the previous run.

    Flags a regression when macro F1 drops by more than `threshold` points. The
    first ever run has no predecessor, so delta is None and nothing is flagged.
    """
    prev = latest_run(db_path) if db_path is not None else latest_run()
    if prev is None:
        return RegressionResult(
            prev_run_id=None,
            prev_macro_f1=None,
            delta=None,
            flagged=False,
            threshold=threshold,
        )

    prev_f1 = prev["macro_f1"]
    delta = round(current_macro_f1 - prev_f1, 4)
    flagged = delta < -threshold
    return RegressionResult(
        prev_run_id=prev["id"],
        prev_macro_f1=prev_f1,
        delta=delta,
        flagged=flagged,
        threshold=threshold,
    )
