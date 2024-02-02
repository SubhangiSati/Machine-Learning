import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Fit logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=10000)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy of model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of model: {accuracy:.2f}")

# Plot decision boundary
x_min, x_max = iris.data[:, 0].min() - 0.5, iris.data[:, 0].max() + 0.5
y_min, y_max = iris.data[:, 1].min() - 0.5, iris.data[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.ravel().shape), np.zeros(xx.ravel().shape)])
Z = Z.reshape(xx.shape)
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.8)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Multiple Logistic Regression on Iris Dataset')
plt.show()
