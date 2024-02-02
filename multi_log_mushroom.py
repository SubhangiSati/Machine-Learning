import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load mushroom dataset
mushroom_df = pd.read_csv("/Users/subhangisati/Downloads/mushrooms.csv")
# One-hot encode categorical variables
encoded_df = pd.get_dummies(mushroom_df.drop(columns=['class']), 
                            prefix_sep='_', 
                            drop_first=True)

# Add target variable to encoded dataframe
encoded_df['class'] = mushroom_df['class'].apply(lambda x: 1 if x == 'p' else 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_df.drop(columns=['class']), 
                                                    encoded_df['class'], 
                                                    test_size=0.2, 
                                                    random_state=42)
# Fit logistic regression model to training data
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
# Make predictions on testing data
y_pred = logreg.predict(X_test)
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Compute accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
from sklearn.metrics import roc_curve, roc_auc_score

# Compute ROC curve and AUC score
y_prob = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
