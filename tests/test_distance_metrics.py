"""Tests for distanceMetrics."""

import math
import pytest
import numpy as np
from simple_kNN.distanceMetrics import distanceMetrics

dm = distanceMetrics()


def test_euclidean_basic():
    assert math.isclose(dm.euclideanDistance([0, 0], [3, 4]), 5.0)


def test_euclidean_zero():
    assert dm.euclideanDistance([1, 2, 3], [1, 2, 3]) == 0.0


def test_euclidean_numpy():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert math.isclose(dm.euclideanDistance(a, b), math.sqrt(2))


def test_euclidean_shape_mismatch():
    with pytest.raises(ValueError):
        dm.euclideanDistance([1, 2], [1, 2, 3])


def test_manhattan_basic():
    assert dm.manhattanDistance([0, 0], [3, 4]) == 7.0


def test_manhattan_zero():
    assert dm.manhattanDistance([5, 5], [5, 5]) == 0.0


def test_hamming_basic():
    assert dm.hammingDistance([1, 2, 3], [1, 9, 3]) == 1


def test_hamming_all_different():
    assert dm.hammingDistance([1, 2], [3, 4]) == 2


def test_hamming_shape_mismatch():
    with pytest.raises(ValueError):
        dm.hammingDistance([1], [1, 2])
