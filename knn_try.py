import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load Iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)#30% TRAINING , RANDOMLY SPLIT DATA

# Define the range of k values to test
k_values = np.arange(1, 30, 2) #VALUES OF K

# Initialize an empty list to store accuracy scores
accuracy_scores = []

# Loop over the k values and train the k-NN classifier for each value
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    

    # Compute and print the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
   
# Plot the accuracy scores as a function of k
plt.plot(k_values, accuracy_scores)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('k-NN accuracy on Iris dataset')
plt.show()

# Print the best k value and accuracy score
best_k = k_values[np.argmax(accuracy_scores)]
best_accuracy = accuracy_scores[np.argmax(accuracy_scores)]
print(f"Best k: {best_k}")
print(f"Accuracy: {best_accuracy * 100:.2f}%")

for k in k_values:
    # Plot the confusion matrix
    plt.figure()
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix for k = {k}")
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

