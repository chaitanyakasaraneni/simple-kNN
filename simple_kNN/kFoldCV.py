"""k-Fold Cross Validation for the kNN classifier."""

from __future__ import annotations

import numpy as np
from .kNNClassifier import kNNClassifier


class kFoldCV:
    """k-Fold Cross Validation for the kNN classifier."""

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def printMetrics(self, actual, predictions) -> float:
        """
        Calculate and display prediction accuracy.

        Parameters
        ----------
        actual : array-like
        predictions : array-like

        Returns
        -------
        float – accuracy percentage in [0, 100]
        """
        actual      = np.asarray(actual)
        predictions = np.asarray(predictions)
        if len(actual) != len(predictions):
            raise ValueError("actual and predictions must have the same length.")
        return float(np.mean(actual == predictions) * 100.0)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def crossValSplit(self, dataset: list, numFolds: int) -> list[list]:
        """
        Split dataset into *numFolds* roughly equal folds (without replacement).

        Parameters
        ----------
        dataset : list of rows
        numFolds : int

        Returns
        -------
        list of folds, each fold being a list of rows
        """
        data      = np.asarray(dataset, dtype=object)
        indices   = np.random.permutation(len(data))
        fold_size = len(data) // numFolds
        return [
            data[indices[i * fold_size : (i + 1) * fold_size]].tolist()
            for i in range(numFolds)
        ]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def kFCVEvaluate(self, dataset: list, numFolds: int, *args) -> None:
        """
        Run k-Fold Cross Validation and print per-fold accuracy.

        Parameters
        ----------
        dataset : list of rows (last column is the label)
        numFolds : int
        *args : forwarded to kNNClassifier.predict (k, distanceMetric)
        """
        knn   = kNNClassifier()
        folds = self.crossValSplit(dataset, numFolds)
        print(f"\nDistance Metric: {args[-1]}\n")
        scores: list[float] = []

        for i, fold in enumerate(folds):
            # Build train set from all other folds — list-of-lists, no copies
            trainSet = [row for j, f in enumerate(folds) if j != i for row in f]

            trainFeatures = [row[:-1] for row in trainSet]
            trainLabels   = [row[-1]  for row in trainSet]
            knn.fit(trainFeatures, trainLabels)

            testFeatures = [row[:-1] for row in fold]
            actual       = [row[-1]  for row in fold]

            predicted = knn.predict(testFeatures, *args)
            accuracy  = self.printMetrics(actual, predicted)
            scores.append(accuracy)

        print("*" * 20)
        print(f"Scores: {scores}")
        print("*" * 20)
        print(f"\nMaximum Accuracy: {max(scores):.3f}%")
        print(f"\nMean Accuracy:    {sum(scores) / len(scores):.3f}%")
