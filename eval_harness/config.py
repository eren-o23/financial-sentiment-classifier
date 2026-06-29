"""Configuration for the eval harness.

Reuses the model/label constants and DB path from `monitor.config` so eval numbers
are computed against the exact same label space as training and serving.
"""
from __future__ import annotations

from monitor.config import (  # noqa: F401  (re-exported for convenience)
    DB_PATH,
    ID2LABEL,
    LABEL2ID,
    MODEL_VERSION,
    PROJECT_ROOT,
)

# Class order used for metric reports — matches the training notebook (cell 23).
LABEL_ORDER = ["positive", "negative", "neutral"]

# --- Consistency check ---
# How many times to re-score each headline when checking repeat-run determinism.
N_CONSISTENCY_RUNS = 5
# Batch sizes to compare for the batch-invariance check (single vs batched).
CONSISTENCY_BATCH_SIZES = (1, 8, 32)
# predict() rounds confidence to 4 decimals, so anything within this tolerance is
# considered stable (guards against float-ordering drift across backends).
CONFIDENCE_EPSILON = 1e-3

# --- Regression ---
# Flag a regression when macro F1 drops by more than this many points vs the
# previous stored run. Tunable.
REGRESSION_F1_DROP_THRESHOLD = 0.05

# --- Golden dataset ---
GOLDEN_PATH = PROJECT_ROOT / "eval_harness" / "golden" / "golden_headlines.jsonl"
