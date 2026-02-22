"""k-Nearest Neighbors classifier."""

from __future__ import annotations

import numpy as np
from collections import Counter
from numpy.typing import ArrayLike


class kNNClassifier:
    """
    k-Nearest Neighbors classifier.

    Supports euclidean, manhattan, and hamming distance metrics.
    All distance computations are fully vectorized via NumPy —
    no Python loops over training points at prediction time.
    """

    def __init__(self, k: int = 3, distanceMetric: str = "euclidean") -> None:
        """
        Parameters
        ----------
        k : int
            Number of neighbors to use (default 3).
        distanceMetric : str
            One of 'euclidean', 'manhattan', or 'hamming' (default 'euclidean').
        """
        self.k = k
        self.distanceMetric = distanceMetric

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, xTrain: ArrayLike, yTrain: ArrayLike) -> "kNNClassifier":
        """
        Store training data (lazy / instance-based learning).

        Parameters
        ----------
        xTrain : array-like of shape (n_samples, n_features)
        yTrain : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        self._trainData = np.asarray(xTrain, dtype=np.float64)
        self._trainLabels = np.asarray(yTrain)
        if len(self._trainData) != len(self._trainLabels):
            raise ValueError("xTrain and yTrain must have the same number of rows.")
        return self

    # ------------------------------------------------------------------
    # Vectorised distance computation
    # ------------------------------------------------------------------

    def _computeDistances(self, testMatrix: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between every test row and every
        training row in one vectorised pass.

        Returns
        -------
        distances : ndarray of shape (n_test, n_train)
        """
        metric = self.distanceMetric.lower()

        if metric == "euclidean":
            # Use the identity ||a-b||² = ||a||² + ||b||² - 2·a·bᵀ
            # This is O(n·m·f) via BLAS matmul — fastest for large arrays.
            sq_test  = np.einsum("ij,ij->i", testMatrix, testMatrix)[:, np.newaxis]
            sq_train = np.einsum("ij,ij->i", self._trainData, self._trainData)
            cross    = testMatrix @ self._trainData.T
            dist_sq  = sq_test + sq_train - 2.0 * cross
            # Clamp tiny negatives caused by floating-point rounding before sqrt
            return np.sqrt(np.maximum(dist_sq, 0.0))

        elif metric == "manhattan":
            # Broadcast: (n_test, 1, f) - (1, n_train, f) → sum over f
            return np.sum(
                np.abs(
                    testMatrix[:, np.newaxis, :] - self._trainData[np.newaxis, :, :]
                ),
                axis=2,
            )

        elif metric == "hamming":
            return np.sum(
                testMatrix[:, np.newaxis, :] != self._trainData[np.newaxis, :, :],
                axis=2,
            ).astype(np.float64)

        else:
            raise ValueError(
                f"Unknown distance metric '{self.distanceMetric}'. "
                "Choose from 'euclidean', 'manhattan', or 'hamming'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def getNeighbors(self, testRow: ArrayLike) -> list:
        """
        Return the k nearest neighbors for a single test row.

        Parameters
        ----------
        testRow : array-like of shape (n_features,)

        Returns
        -------
        list of [trainRow, distance, label] for each of the k neighbors,
        sorted by ascending distance.
        """
        test = np.asarray(testRow, dtype=np.float64).reshape(1, -1)
        dists = self._computeDistances(test)[0]           # shape (n_train,)
        k_idx = np.argpartition(dists, self.k)[: self.k]  # unordered top-k
        k_idx = k_idx[np.argsort(dists[k_idx])]           # sort those k
        return [
            [self._trainData[i].tolist(), float(dists[i]), self._trainLabels[i]]
            for i in k_idx
        ]

    def predict(
        self,
        xTest: ArrayLike,
        k: int | None = None,
        distanceMetric: str | None = None,
    ) -> list:
        """
        Predict class labels for test samples.

        Parameters
        ----------
        xTest : array-like of shape (n_test, n_features)
        k : int, optional
            Override the k set at construction time.
        distanceMetric : str, optional
            Override the metric set at construction time.

        Returns
        -------
        predictions : list of predicted labels (length n_test)
        """
        if k is not None:
            self.k = k
        if distanceMetric is not None:
            self.distanceMetric = distanceMetric

        testMatrix  = np.asarray(xTest, dtype=np.float64)
        distMatrix  = self._computeDistances(testMatrix)          # (n_test, n_train)
        n_test      = len(testMatrix)

        # argpartition is O(n_train) vs O(n_train·log n_train) for full argsort
        k_indices = np.argpartition(distMatrix, self.k, axis=1)[:, : self.k]

        train_labels = self._trainLabels
        predictions  = [
            Counter(train_labels[k_indices[i]].tolist()).most_common(1)[0][0]
            for i in range(n_test)
        ]
        return predictions
