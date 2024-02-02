import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = load_iris()
X, y = iris.data[:, :2], iris.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression on training set
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict target values for the test set
y_pred = reg.predict(X_test)

# Compute R-squared score of the model on the test set
accuracy = r2_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy}")

# Plot linear regression line
plt.scatter(X_test[:, 0], y_test, color='blue')
plt.plot(X_test[:, 0], y_pred, color='red', linewidth=2)
plt.xlabel('Sepal length')
plt.ylabel('Target')
plt.title('Linear Regression')
plt.show()
