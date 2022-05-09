import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

x = []
y = []

with open ("U:\COMP3217 Security of Cyber Physical Systems\Energy-Manipulation-Detection\Data\TrainingData.txt") as f:
    lines = f.readlines()
    lines = lines[:]
    for ind, item in enumerate(lines):
        lines[ind] = lines[ind].strip("\n").split(",")
        lines[ind] = [float(v) for v in lines[ind]]
        x.append(lines[ind][0:-1])
        y.append(lines[ind][-1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x = np.array(x)
y = np.array(y)
x_train_full = x
y_train_full = y

##scaling between 0 and 1 - normalising
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train_full_2 = scaler.transform(x_train_full)
##

#KNN
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(x_train, y_train)
y_pred = neigh.predict(x_test)
y_pred_train = neigh.predict(x_train_full_2)
print("\nAccuracy for KNN on Training Set:",metrics.accuracy_score(y_train_full, y_pred_train))
print("Accuracy for KNN on 20% Testing Set:",metrics.accuracy_score(y_test, y_pred))

#Decision Tree
tree = DecisionTreeClassifier().fit(x_train, y_train)
print("\nAccuracy for Decision Tree classifier on Training set:",tree.score(x_train_full_2, y_train_full))
print("Accuracy for Decision Tree classifier on 20% Testing set:",tree.score(x_test, y_test))

#Logistic Regression
log = linear_model.LogisticRegression(random_state = 2,solver = 'liblinear',multi_class = 'auto')
log = log.fit(x_train, y_train)
print("\nAccuracy for Logistic Regression on Training set:",log.score(x_train_full_2, y_train_full))
print("Accuracy for Logistic Regression on 20% Testing set:",log.score(x_test, y_test))

#Support Vector Machine
svm = SVC()
svm.fit(x_train, y_train)
#print("\nAccuracy for SVM classifier on 75% Training set:",svm.score(x_train, y_train))
print("\nAccuracy for SVM classifier on Training set:",svm.score(x_train_full_2, y_train_full))
print("Accuracy for SVM classifier on 20% Testing set:",svm.score(x_test, y_test))

#Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
print("\nAccuracy for LDA classifier on Training set:",lda.score(x_train_full_2, y_train_full))
print("Accuracy for LDA classifier on 20% Testing set:",lda.score(x_test, y_test))

#Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train)
print("\nAccuracy for GNB classifier on Training set:",gnb.score(x_train_full_2, y_train_full))
print("Accuracy for GNB classifier on 20% Testing set:",gnb.score(x_test, y_test))

#Multi Layer Perception
reg = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000).fit(x_train, y_train)
y_pred_mlp=reg.predict(x_test)
y_pred_mlp_train=reg.predict(x_train_full_2)
print("\nAccuracy for MLP on Training Set:",r2_score(y_pred_mlp_train, y_train_full))
print("Accuracy for MLP on 20% Testing Set:",r2_score(y_pred_mlp, y_test))