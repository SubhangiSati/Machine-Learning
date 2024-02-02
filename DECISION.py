import pandas as pd
import numpy as np
#IMPORT LIBRARIES
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#LOAD IRIS DATASET
iris = load_iris()

#CREATE DATAFRAME
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
df['target'] = df['target'].astype(int)

#SPLIT DATA IN X,Y
X = df.drop('target',axis=1)
Y = df['target']

# SPLIT DATA IN TRAIN TEST
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

# DECISION TREE USING ID3 ALGO
dt = DecisionTreeClassifier(criterion="entropy")

# TRAIN
dt.fit(X_train, Y_train)

# PREDICT
Y_pred = dt.predict(X_test)

# FIND ACCURACY
accuracy = accuracy_score(Y_test,Y_pred)*100
print("\nACCURACY:", accuracy)

# PLOT DECISION TREE
plt.figure(figsize=(20,10))
plot_tree(dt, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

