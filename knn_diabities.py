#IMPORT LIBARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

#LOAD DATASET
dataset = pd.read_csv('/Users/subhangisati/Downloads/diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)

#USE KNN
knn = KNeighborsClassifier(n_neighbors=1)

#TRAIN
knn.fit(X_train, y_train)

#TEST
y_pred = knn.predict(X_test)

#ACCURACY
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:",f1)
