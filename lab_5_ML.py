#TRAIN A LOGISTIC REGRESSION CLASSIFIER TO PREDICT WHEATHER A FLOWER IS IRIS VIRGINICA OR NOT
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
iris= datasets.load_iris()
'''print(list(iris.keys()))
print(iris['data'])
print(iris['target'])
print(iris['DESCR'])'''

x=iris["data"][:,3:]
y=(iris["target"]==2).astype(np.int)

#TRAIN A LOGISTIC REGRESSION CLASSIFIER
clf=LogisticRegression()
clf.fit(x,y)
example=clf.predict(([[2.6]]))
print(example)
