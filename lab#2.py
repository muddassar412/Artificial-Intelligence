# Importing the required libraries
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
# import the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=1)
 #1. GAUSSIAN NAIVE BAYES
gnb = GaussianNB()
 # train the model
gnb.fit(X_train, y_train)
 # make predictions
gnb_pred = gnb.predict(X_test)
 # print the accuracy
print("Accuracy of Gaussian Naive Bayes: ",
accuracy_score(y_test, gnb_pred))
#2. DECISION TREE CLASSIFIER
dt = DecisionTreeClassifier(random_state=0)
 # train the model
dt.fit(X_train, y_train)
 # make predictions
dt_pred = dt.predict(X_test)
 # print the accuracy
print("Accuracy of Decision Tree Classifier: ",
accuracy_score(y_test, dt_pred))
#3. SUPPORT VECTOR MACHINE
svm_clf = svm.SVC(kernel='linear')  # Linear Kernel
 # train the model
svm_clf.fit(X_train, y_train)
# make predictions using svm
svm_clf_pred = svm_clf.predict(X_test)
 # print the accuracy
print("Accuracy of Support Vector Machine: ",
accuracy_score(y_test, svm_clf_pred))
