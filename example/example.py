import numpy as np
import pandas as pd
import operator
from random import randrange
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from simple_kNN.distanceMetrics import distanceMetrics
from simple_kNN.kFoldCV import kFoldCV
from simple_kNN.kNNClassifier import kNNClassifier

def readData(fileName):
    '''
    Description:
        This method is to read the data from a given file
    '''
    data = []
    labels = []

    with open(fileName, "r") as file:
        lines = file.readlines()
    for line in lines:
        splitline = line.strip().split(',')
        data.append(splitline)
        labels.append(splitline[-1])
    return data, labels

def readDatawithoutkfcv(fileName):
    '''
    Description:
        This method is to read the data from a given file
    '''
    data = []
    labels = []

    with open(fileName, "r") as file:
        lines = file.readlines()
    for line in lines:
        splitline = line.strip().split(',')
        data.append(splitline[:-1])
        labels.append(splitline[-1])
    return data, labels

# ### Hayes-Roth Data

# In[7]:

print('***** Without KFold Cross Validation *****')
trainFile = 'Datasets/HayesRoth/hayes-roth.data'

trainData, trainLabel = readDatawithoutkfcv(trainFile)

trainFeatures = []
for row in trainData:
    index = row[0:]
    temp = [int(item) for item in index]
    trainFeatures.append(temp)
    
trainLabels = [int(label) for label in trainLabel]

knn=kNNClassifier()
knn.fit(trainFeatures, trainLabels)
testFile = 'Datasets/HayesRoth/hayes-roth.test'

testData, testLabel = readData(testFile)

testFeatures = []
for row in testData:
    index = row[0:]
    temp = [int(item) for item in index]
    testFeatures.append(temp)
    
testLabels = [int(label) for label in testLabel]
eucPredictions = knn.predict(testFeatures, 3, 'euclidean')
print('***** Confusion Matrix *****')
print(confusion_matrix(testLabels, eucPredictions))
# **Create an object for k-Fold cross validation class**

# In[8]:
print('***** With KFold Cross Validation *****')
trainData, trainLabel = readData(trainFile)

trainFeatures = []
for row in trainData:
    index = row[0:]
    temp = [int(item) for item in index]
    trainFeatures.append(temp)
    
trainLabels = [int(label) for label in trainLabel]
kfcv = kFoldCV()


# **Call the Evaluation function of kFCV class**
#
# *kfcv.kFCVEvaluate(data, foldCount, neighborCount, distanceMetric)*

# In[9]:
print('*'*20)
print('Hayes Roth Data')


kfcv.kFCVEvaluate(trainFeatures, 10, 3, 'euclidean')


# In[10]:


kfcv.kFCVEvaluate(trainFeatures, 10, 3, 'manhattan')


# In[11]:


kfcv.kFCVEvaluate(trainFeatures, 10, 3, 'hamming')


# ### Car Evaluation Data

# In[12]:


carFile = 'Datasets/CarEvaluation/car.data'

carData, carLabel = readData(carFile)
df = pd.DataFrame(carData)
df = df.apply(preprocessing.LabelEncoder().fit_transform)
carFeatures = df.values.tolist()
carLabels = [car[-1] for car in carFeatures]


# In[13]:
print('*'*20)
print('Car Evaluation Data')
kfcv.kFCVEvaluate(carFeatures, 10, 3, 'euclidean')


# In[14]:


kfcv.kFCVEvaluate(carFeatures, 10, 3, 'manhattan')


# In[15]:


kfcv.kFCVEvaluate(carFeatures, 10, 3, 'hamming')


# ### Breast Cancer Data

# In[16]:

print('*'*20)
print('Breast Cancer Data')

cancerFile = 'Datasets/BreastCancer/breast-cancer.data'

cancerData, cancerLabel = readData(cancerFile)
cdf = pd.DataFrame(cancerData)
cdf = cdf.apply(preprocessing.LabelEncoder().fit_transform)
cancerFeatures = cdf.values.tolist()
cancerLabels = [cancer[-1] for cancer in cancerFeatures]


# In[17]:


kfcv.kFCVEvaluate(cancerFeatures, 10, 3, 'euclidean')


# In[18]:


kfcv.kFCVEvaluate(cancerFeatures, 10, 3, 'manhattan')


# In[19]:


kfcv.kFCVEvaluate(cancerFeatures, 10, 3, 'hamming')

