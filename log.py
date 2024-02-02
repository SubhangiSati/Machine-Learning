#LOGISTIC REGRESSION TO PREDICT IF THE SEPALWIDTH IS FOR VIRGINICA OR NOT
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
iris=datasets.load_iris()
x=iris["data"][:,3:]#column 3 ,sepal width
y=(iris["target"]==2).astype(np.int64)#2= virginica
print(x)
clf=LogisticRegression()
clf.fit(x,y)#train
ex=clf.predict([[1.6]])#any random value
print(ex)
x_new=np.linspace(0,3,100).reshape(-1,1)#sigmodial function x=0-3 values must be plotted that to 100 values
print(x_new)
y_prob=clf.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1],"b-",label="virginica")
plt.show()
