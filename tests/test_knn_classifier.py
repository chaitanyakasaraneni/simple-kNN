"""Tests for kNNClassifier."""

import pytest
import numpy as np
from simple_kNN.kNNClassifier import kNNClassifier


# Simple linearly-separable 2-class dataset
X_train = [[1, 1], [1, 2], [2, 1],   # class 0
           [8, 8], [9, 8], [8, 9]]    # class 1
y_train = [0, 0, 0, 1, 1, 1]

X_test  = [[1, 1], [9, 9]]
y_test  = [0, 1]


@pytest.fixture
def fitted_knn():
    knn = kNNClassifier(k=3, distanceMetric="euclidean")
    knn.fit(X_train, y_train)
    return knn


def test_fit_stores_data(fitted_knn):
    assert fitted_knn._trainData.shape == (6, 2)
    assert len(fitted_knn._trainLabels) == 6


def test_predict_euclidean(fitted_knn):
    preds = fitted_knn.predict(X_test)
    assert preds == y_test


def test_predict_manhattan():
    knn = kNNClassifier(k=3, distanceMetric="manhattan")
    knn.fit(X_train, y_train)
    assert knn.predict(X_test) == y_test


def test_predict_hamming():
    knn = kNNClassifier(k=3, distanceMetric="hamming")
    knn.fit(X_train, y_train)
    assert knn.predict(X_test) == y_test


def test_predict_override_k(fitted_knn):
    # Should still classify correctly with k=1
    preds = fitted_knn.predict(X_test, k=1)
    assert preds == y_test


def test_predict_override_metric(fitted_knn):
    preds = fitted_knn.predict(X_test, distanceMetric="manhattan")
    assert preds == y_test


def test_get_neighbors_length(fitted_knn):
    neighbors = fitted_knn.getNeighbors([1, 1])
    assert len(neighbors) == 3


def test_get_neighbors_sorted(fitted_knn):
    neighbors = fitted_knn.getNeighbors([1, 1])
    dists = [n[1] for n in neighbors]
    assert dists == sorted(dists)


def test_unknown_metric_raises():
    knn = kNNClassifier(k=3, distanceMetric="cosine")
    knn.fit(X_train, y_train)
    with pytest.raises(ValueError, match="Unknown distance metric"):
        knn.predict(X_test)


def test_mismatched_train_raises():
    knn = kNNClassifier()
    with pytest.raises(ValueError):
        knn.fit([[1, 2], [3, 4]], [0])  # 2 rows, 1 label


def test_predict_iris():
    """Smoke-test on the bundled iris dataset."""
    from simple_kNN.datasets import load_iris
    X, y = load_iris()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    knn = kNNClassifier(k=5, distanceMetric="euclidean")
    knn.fit(X[:split], y[:split])
    preds = knn.predict(X[split:])
    accuracy = np.mean(np.array(preds) == y[split:])
    assert accuracy > 0.90, f"Expected >90% accuracy on iris, got {accuracy:.1%}"
