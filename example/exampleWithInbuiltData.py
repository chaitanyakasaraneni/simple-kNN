import numpy as np
import pandas as pd
import operator
from random import randrange
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from simple_kNN.kFoldCV import kFoldCV
from simple_kNN.kNNClassifier import kNNClassifier
from simple_kNN.datasets import load_iris
from simple_kNN.datasets import load_breast_cancer


print('\nIris Data')
iris = []
iris_data, iris_labels = load_iris()
for ind,elem in enumerate(iris_data):
    element = list(iris_data[ind])
    element.append(iris_labels[ind])
    iris.append(element)

kfcv = kFoldCV()
kfcv.kFCVEvaluate(iris, 10, 3, 'euclidean')


print('\nBreast Cancer Data')
breastCancer = []
bc_data, bc_labels = load_breast_cancer()
for ind,elem in enumerate(bc_data):
    element = list(bc_data[ind])
    element.append(bc_labels[ind])
    breastCancer.append(element)

kfcv.kFCVEvaluate(breastCancer, 10, 3, 'manhattan')
