"""Distance metrics for use with kNN."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


class distanceMetrics:
    """
    Distance metrics for use with kNN.

    Supports euclidean, manhattan, and hamming distances.
    All methods accept plain Python lists or numpy arrays.
    """

    def euclideanDistance(self, vector1: ArrayLike, vector2: ArrayLike) -> float:
        """
        Calculate the Euclidean (L2) distance between two vectors.

        Formula: sqrt( sum( (x_i - y_i)^2 ) )
        """
        a = np.asarray(vector1, dtype=np.float64)
        b = np.asarray(vector2, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError("Undefined for sequences of unequal length.")
        diff = a - b
        return float(np.sqrt(diff @ diff))

    def manhattanDistance(self, vector1: ArrayLike, vector2: ArrayLike) -> float:
        """
        Calculate the Manhattan (L1) distance between two vectors.

        Formula: sum( |x_i - y_i| )
        """
        a = np.asarray(vector1, dtype=np.float64)
        b = np.asarray(vector2, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError("Undefined for sequences of unequal length.")
        return float(np.sum(np.abs(a - b)))

    def hammingDistance(self, vector1: ArrayLike, vector2: ArrayLike) -> int:
        """
        Calculate the Hamming distance between two vectors.

        Formula: number of positions where elements differ.
        """
        a = np.asarray(vector1)
        b = np.asarray(vector2)
        if a.shape != b.shape:
            raise ValueError("Undefined for sequences of unequal length.")
        return int(np.sum(a != b))
