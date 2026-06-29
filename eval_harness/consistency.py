"""Consistency checks: is the classifier stable across repeated and batched runs?

The model runs in eval() mode with fixed max_length padding, so it should be
deterministic. These checks turn that expectation into an assertion, guarding
against two regressions:

  * repeat-run drift  — accidental nondeterminism (dropout left on, sampling
    introduced) would make the same text score differently across runs.
  * batch variance    — a padding/batching bug would make a headline score
    differently alone vs inside a batch.

Both are expected to PASS on the current model; a failure signals a real bug.
"""
from __future__ import annotations

from .config import CONFIDENCE_EPSILON, CONSISTENCY_BATCH_SIZES, N_CONSISTENCY_RUNS


def check_consistency(
    classifier,
    texts: list[str],
    n_runs: int = N_CONSISTENCY_RUNS,
    batch_sizes: tuple[int, ...] = CONSISTENCY_BATCH_SIZES,
    epsilon: float = CONFIDENCE_EPSILON,
) -> dict:
    """Run repeat-run and batch-invariance checks over `texts`.

    Returns a dict with a top-level `passed` flag, per-check counts, and a small
    list of example divergences (capped) for the report.
    """
    examples: list[dict] = []

    # --- Repeat-run determinism: score the whole set n_runs times. ---
    runs = [classifier.predict(texts) for _ in range(n_runs)]
    n_label_unstable = 0
    n_confidence_unstable = 0
    for i, text in enumerate(texts):
        labels = {runs[r][i].label for r in range(n_runs)}
        confs = [runs[r][i].confidence for r in range(n_runs)]
        conf_range = max(confs) - min(confs)
        label_unstable = len(labels) > 1
        conf_unstable = conf_range > epsilon
        if label_unstable:
            n_label_unstable += 1
        if conf_unstable:
            n_confidence_unstable += 1
        if (label_unstable or conf_unstable) and len(examples) < 10:
            examples.append(
                {
                    "check": "repeat_run",
                    "text": text,
                    "labels": sorted(labels),
                    "confidence_range": round(conf_range, 6),
                }
            )

    # --- Batch-invariance: each text alone vs inside batches of varying size. ---
    # `baseline` scores every text in a batch of 1; other batch sizes must agree.
    baseline = [classifier.predict([t])[0] for t in texts]
    n_batch_mismatch = 0
    for bs in batch_sizes:
        if bs == 1:
            continue
        batched = classifier.predict(texts, batch_size=bs)
        for i, text in enumerate(texts):
            label_mismatch = batched[i].label != baseline[i].label
            conf_mismatch = abs(batched[i].confidence - baseline[i].confidence) > epsilon
            if label_mismatch or conf_mismatch:
                n_batch_mismatch += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "check": "batch_invariance",
                            "text": text,
                            "batch_size": bs,
                            "alone": (baseline[i].label, baseline[i].confidence),
                            "batched": (batched[i].label, batched[i].confidence),
                        }
                    )

    passed = (
        n_label_unstable == 0
        and n_confidence_unstable == 0
        and n_batch_mismatch == 0
    )
    return {
        "passed": passed,
        "n_texts": len(texts),
        "n_runs": n_runs,
        "batch_sizes": list(batch_sizes),
        "epsilon": epsilon,
        "n_label_unstable": n_label_unstable,
        "n_confidence_unstable": n_confidence_unstable,
        "n_batch_mismatch": n_batch_mismatch,
        "examples": examples,
    }
