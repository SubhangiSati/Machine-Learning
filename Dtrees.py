import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Loading Iris dataset
iris = load_iris()

# Creating dataframe
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
df['target'] = df['target'].astype(int)

# Splitting dataset into X & Y
X = df.drop('target',axis=1)
Y = df['target']

# Splitting dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

# Decision Tree classifier with ID3 Algorithm
dt = DecisionTreeClassifier(criterion="entropy")

# Training the model
dt.fit(X_train, Y_train)

# Predicting on test set
Y_pred = dt.predict(X_test)

# Finding accuracy of model
accuracy = accuracy_score(Y_test,Y_pred)*100
print("\nACCURACY:", accuracy)

# Plotting Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dt, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
