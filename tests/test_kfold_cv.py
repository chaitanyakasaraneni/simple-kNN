"""Tests for kFoldCV."""

import pytest
import numpy as np
from simple_kNN.kFoldCV import kFoldCV

kfcv = kFoldCV()


def test_print_metrics_perfect():
    assert kfcv.printMetrics([0, 1, 2], [0, 1, 2]) == 100.0


def test_print_metrics_half():
    assert kfcv.printMetrics([0, 1], [0, 0]) == 50.0


def test_print_metrics_mismatch():
    with pytest.raises(ValueError):
        kfcv.printMetrics([0, 1], [0])


def test_cross_val_split_count():
    dataset = list(range(100))
    folds = kfcv.crossValSplit(dataset, 10)
    assert len(folds) == 10


def test_cross_val_split_sizes():
    dataset = list(range(100))
    folds = kfcv.crossValSplit(dataset, 10)
    for fold in folds:
        assert len(fold) == 10


def test_kfcv_evaluate_runs():
    """kFCVEvaluate should complete without error on a simple dataset."""
    from simple_kNN.datasets import load_iris
    X, y = load_iris()
    dataset = [list(X[i]) + [int(y[i])] for i in range(len(X))]
    # Should not raise
    kfcv.kFCVEvaluate(dataset, 5, 3, "euclidean")
