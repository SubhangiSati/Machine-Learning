import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler#to bring all the features on the same scale for efficiency of model
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder#convert categorial to numerical values here class

# Reading Dataset
df1= pd.read_csv('/Users/subhangisati/Downloads/mushrooms.csv')
df1.head()

# Label Encoding
lb = LabelEncoder()
df1 = df1.apply(lb.fit_transform)
df1.head()

# Splitting dataset into X & Y
X = df1.drop('class',axis=1)
Y = df1['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)#30% for training

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# KNN on Mushroom
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)

# Confusion Matrix & Accuracy of KNN
acc=(knn.score(X_test,Y_test))*100
print("\nAccuracy:", acc)
results=[]
results=knn.predict(X_test)
print("\nF1 Score:", f1_score(Y_test,results,average='micro'))
c1=confusion_matrix(Y_test,results)
cm=ConfusionMatrixDisplay(confusion_matrix=c1, display_labels=knn.classes_)
print("\n\nConfusion Matrix:\n")
cm.plot(cmap='Greens')
plt.show()
