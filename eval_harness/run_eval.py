"""CLI entry point for the eval harness.

Mirrors `monitor/run_daily.py`: parse args, load the classifier once, run the
eval, and print a human-readable stats block.

    python -m eval_harness.run_eval [--golden PATH]
"""
from __future__ import annotations

import argparse

from monitor.classifier import FinancialSentimentClassifier

from .config import GOLDEN_PATH
from .runner import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the classifier eval harness.")
    parser.add_argument(
        "--golden",
        default=str(GOLDEN_PATH),
        help="Path to the golden JSONL dataset (default: committed golden set).",
    )
    args = parser.parse_args()

    classifier = FinancialSentimentClassifier()
    report = run_eval(classifier, golden_path=args.golden)

    c = report.consistency
    r = report.regression
    print(f"\n[eval] run #{report.run_id}  model={report.model_version}")
    print(f"  golden size      : {report.golden_size}")
    print(f"  accuracy         : {report.accuracy:.4f}")
    print(f"  macro F1         : {report.macro_f1:.4f}")
    print(f"  weighted F1      : {report.weighted_f1:.4f}")
    print("  per-class F1     : " + ", ".join(
        f"{cls} {m['f1']:.3f}" for cls, m in report.per_class.items()
    ))
    print(
        "  consistency      : "
        + ("PASS" if c["passed"] else "FAIL")
        + f"  (label_unstable={c['n_label_unstable']}, "
        + f"conf_unstable={c['n_confidence_unstable']}, "
        + f"batch_mismatch={c['n_batch_mismatch']})"
    )
    if r["delta"] is None:
        print("  regression       : baseline run (no previous run to compare)")
    else:
        flag = "  <-- REGRESSION FLAGGED" if r["flagged"] else ""
        print(
            f"  regression       : macro F1 delta {r['delta']:+.4f} vs run "
            f"#{r['prev_run_id']} (threshold {r['threshold']}){flag}"
        )

    if report.misclassifications:
        print(f"\n  {len(report.misclassifications)} misclassification(s), top 5:")
        for m in report.misclassifications[:5]:
            print(
                f"    - [{m['ground_truth']} -> {m['predicted']} @ {m['confidence']:.2f}] "
                f"{m['text'][:90]}"
            )
    print()


if __name__ == "__main__":
    main()
