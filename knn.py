import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
data_set = pd.read_csv("/Users/subhangisati/Downloads/Seed_Data.csv")
data_set.info
feature_dataset = data_set.iloc[:, 0:7]
label_dataset = data_set.iloc[:, 7]
sns.pairplot(data = data_set, hue = 'target')
feature_data_train, feature_data_test, label_data_train, label_data_test = train_test_split(feature_dataset, label_dataset, test_size = 0.4)
scaler = StandardScaler()
scaler.fit(feature_data_train)
feature_data_train = scaler.transform(feature_data_train)
feature_data_test = scaler.transform(feature_data_test)

acc_1 = []
acc_2 = []
for k in range(3, 17):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(feature_data_train, label_data_train)
    predict_classifier_1 = classifier.predict(feature_data_train)
    acc_1.append(np.mean(label_data_train == predict_classifier_1))
    predict_classifier_2 = classifier.predict(feature_data_test)
    acc_2.append(np.mean(label_data_test == predict_classifier_2))
plt.plot(range(3, 17), acc_1, label = 'train',color='r')
plt.plot(range(3, 17), acc_2, label = 'test',color='g')
plt.show()
classifier = KNeighborsClassifier(n_neighbors = 13) # anything except 9 as its accuracy rate is lowest
classifier.fit(feature_data_train, label_data_train)
predict_classifier = classifier.predict(feature_data_test)

print("THE ACCURACY OF THE MODEL IS~ ", sklearn.metrics.accuracy_score(label_data_test, predict_classifier))
print("THE CLASSIFICATION REPORT OF THE MODEL IS~ ", sklearn.metrics.classification_report(label_data_test, predict_classifier))
print("THE CONFUSION MATRIX OF THE MODEL IS ~ \n", sklearn.metrics.confusion_matrix(label_data_test, predict_classifier))
sns.residplot(x = label_data_test, y = predict_classifier, color = 'b')
plt.show()
