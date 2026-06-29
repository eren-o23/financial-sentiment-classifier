"""Evaluation and regression-testing layer for the financial sentiment classifier.

Measures the classifier on a committed golden dataset of manually-labelled
headlines: a consistency check (repeat-run + batch-invariance), an accuracy check
(per-class precision/recall/F1 via sklearn, mirroring the training notebook), and a
regression runner that stores every run in SQLite and flags macro-F1 drops versus
the previous run.
"""
