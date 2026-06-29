"""Accuracy check: compare classifier predictions to golden ground-truth labels.

Uses the same metric computation as the training notebook (sklearn
`classification_report` with the fixed label order), so the macro F1 reported here
is directly comparable to the notebook's test-set result (0.87 on analyst
sentences). A large gap here quantifies the documented train/inference domain
shift onto news headlines.
"""
from __future__ import annotations

from sklearn.metrics import classification_report, confusion_matrix

from .config import LABEL_ORDER
from .golden import GoldenHeadline


def check_accuracy(classifier, golden: list[GoldenHeadline]) -> dict:
    """Score the golden headlines and compute per-class + macro metrics.

    Returns accuracy, macro/weighted F1, a per-class precision/recall/F1 table, a
    confusion matrix, and the list of misclassified examples.
    """
    texts = [g.text for g in golden]
    y_true = [g.ground_truth for g in golden]
    preds = classifier.predict(texts)
    y_pred = [p.label for p in preds]

    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    per_class = {
        cls: {
            "precision": round(report[cls]["precision"], 4),
            "recall": round(report[cls]["recall"], 4),
            "f1": round(report[cls]["f1-score"], 4),
            "support": int(report[cls]["support"]),
        }
        for cls in LABEL_ORDER
    }

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    misclassifications = [
        {
            "text": g.text,
            "ground_truth": g.ground_truth,
            "predicted": p.label,
            "confidence": p.confidence,
            "ticker": g.ticker,
            "rationale": g.rationale,
        }
        for g, p in zip(golden, preds)
        if g.ground_truth != p.label
    ]

    return {
        "n": len(golden),
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": LABEL_ORDER,
            "matrix": cm.tolist(),  # rows = actual, cols = predicted
        },
        "n_misclassified": len(misclassifications),
        "misclassifications": misclassifications,
    }
