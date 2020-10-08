# simple-kNN
This repository is for Continuous Integration of my simple k-Nearest Neighbors (kNN) algorithm to pypi package.

For notebook version please visit [this repository]()

#### *k*-Nearest Neighbors
*k*-Nearest Neighbors, kNN for short, is a very simple but powerful technique used for making predictions. The principle behind kNN is to use **“most similar historical examples to the new data.”**

#### *k*-Nearest Neighbors in 4 easy steps
 - Choose a value for *k*
 - Find the distance of the new point to each record of training data
 - Get the k-Nearest Neighbors
 - Making Predictions
   - For classification problem, the new data point belongs to the class that most of the neighbors belong to. 
   - For regression problem, the prediction can be average or weighted average of the label of k-Nearest Neighbors

Finally, we evaluate the model using *k*-Fold Cross Validation technique

#### *k*-Fold Cross Validation
This technique involves randomly dividing the dataset into k-groups or folds of approximately equal size. The first fold is kept for testing and the model is trained on remaining k-1 folds.

## Installation

    pip install word_knn


#### References
- More info on Cross Validation can be seen [here](https://medium.com/datadriveninvestor/k-fold-and-other-cross-validation-techniques-6c03a2563f1e)
- [kNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [kFold Cross Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
