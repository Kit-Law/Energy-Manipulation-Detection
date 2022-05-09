import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

#The classifier used will be a C-Support Vector Classification with a RBF kernel
#This scored the highest acuraccy in testing at 93.9%
#The data seems to have around 4 turning points so a 5 degrees is enough freedom to
#make it acurate without overfitting
classifier = SVC(kernel="rbf", gamma="auto", C=10, degree=5)

#Read the training and testing data
trainingData = pd.read_csv('..\..\Data\TrainingData.txt', header=None)
testingData = pd.read_csv('..\..\Data\TestingData.txt', header=None)
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
f = open('..\..\Output\TestingResults.txt', "w")
for i in range(len(testingData)):
    for dataPoint in testingData[i]:
        f.write(format(dataPoint, ".14f"))
        f.write(",")
    f.write(str(int(preditctions[i])))
    f.write("\n")
f.close()