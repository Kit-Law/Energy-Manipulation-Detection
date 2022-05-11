import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

dirname = os.path.dirname(__file__)

#The classifier used will be a C-Support Vector Classification with a RBF kernel
#This scored the highest acuraccy in testing at 94.9%
classifier = SVC(kernel="rbf", gamma="auto", decision_function_shape='ovo', C=1000)

#Read the training and testing data
trainingData = pd.read_csv(os.path.join(dirname, '../../Data/TrainingData.txt'), header=None)
testingData = pd.read_csv(os.path.join(dirname, '../../Data/TestingData.txt'), header=None)
trainingData = np.array(trainingData)
testingData = np.array(testingData)

#Split the data from the class identifer
y = trainingData[:,24]
X = np.delete(trainingData, 24, 1)

#Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
testingDataNormalized = scaler.transform(testingData)

#Fit the classifier and label the testing split
clf = classifier.fit(X, y)
preditctions = clf.predict(testingDataNormalized)

#Write the labels to the output file
f = open(os.path.join(dirname, '../../Output/TestingResults.txt'), "w")
for i in range(len(testingData)):
    for dataPoint in testingData[i]:
        f.write(format(dataPoint, ".14f"))
        f.write(",")
    f.write(str(int(preditctions[i])))
    f.write("\n")
f.close()