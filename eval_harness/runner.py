"""Orchestrate a full eval run: golden -> consistency -> accuracy -> regression.

`run_eval` is the single entry point shared by the CLI and the API. It loads the
golden dataset, runs both checks, compares macro F1 against the previous run,
persists the run to SQLite, and returns a structured report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .accuracy import check_accuracy
from .config import GOLDEN_PATH, MODEL_VERSION
from .consistency import check_consistency
from .golden import load_golden
from .regression import compare
from .storage import insert_run


@dataclass
class EvalReport:
    model_version: str
    golden_size: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict
    confusion_matrix: dict
    consistency: dict
    regression: dict
    misclassifications: list[dict] = field(default_factory=list)
    run_id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def run_eval(classifier, golden_path: Path = GOLDEN_PATH) -> EvalReport:
    """Run the full eval against the golden dataset and persist the result."""
    golden = load_golden(golden_path)

    consistency = check_consistency(classifier, [g.text for g in golden])
    accuracy = check_accuracy(classifier, golden)

    # Compare BEFORE inserting so latest_run() is the true predecessor.
    reg = compare(accuracy["macro_f1"])
    regression = {
        "prev_run_id": reg.prev_run_id,
        "prev_macro_f1": reg.prev_macro_f1,
        "delta": reg.delta,
        "flagged": reg.flagged,
        "threshold": reg.threshold,
    }

    report = EvalReport(
        model_version=MODEL_VERSION,
        golden_size=accuracy["n"],
        accuracy=accuracy["accuracy"],
        macro_f1=accuracy["macro_f1"],
        weighted_f1=accuracy["weighted_f1"],
        per_class=accuracy["per_class"],
        confusion_matrix=accuracy["confusion_matrix"],
        consistency=consistency,
        regression=regression,
        misclassifications=accuracy["misclassifications"],
    )

    run_id = insert_run(
        model_version=report.model_version,
        golden_size=report.golden_size,
        accuracy=report.accuracy,
        macro_f1=report.macro_f1,
        weighted_f1=report.weighted_f1,
        consistency_passed=consistency["passed"],
        prev_run_id=reg.prev_run_id,
        macro_f1_delta=reg.delta,
        regression_flagged=reg.flagged,
        report=report.to_dict(),
    )
    report.run_id = run_id
    return report
