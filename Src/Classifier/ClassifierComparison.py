import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import pandas as pd

h = 0.02  # step size in the mesh

names = [
    "Linear SVC",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

data = pd.read_csv('..\Data\TrainingData.txt', header=None)
data = np.array(data)
data[::2], data[1::2] = [data[(data[:,24]==0)], data[(data[:,24]==1)]]

y = data[:,24]
X = np.delete(data, 24, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Creates a list containing 5 lists, each of 8 items, all set to 0
i, j = 10, 10
results = [[0 for x in range(i)] for y in range(j)] 

print("  [Algorithm]---------------------------------[Results]----------------------------------[Average]  ")

i, j = 0, 0
kf = KFold(n_splits=10)
for train, test in kf.split(X):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    clf = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
    preditctions = clf.predict(X_test)

    correct = len([i for i,(predition, actual) in enumerate(zip(preditctions, y_test)) if predition == actual])
    results[j][i] = correct / len(preditctions)
    
    i += 1

print((names[j], results[j], format(sum(results[j][:]) / len(results), ".3f")))