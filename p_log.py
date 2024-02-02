from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
iris=datasets.load_iris()
x=iris["data"][:,3:]
y=(iris["target"]==1).astype(np.int64)#is sepal width of versicolor or not
print(x)
clf=LogisticRegression()
clf.fit(x,y)
ex=clf.predict([[0.8]])
print(ex)
x_new=np.linspace(0,3,100).reshape(-1,1)
print(x_new)
y_prob=clf.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1],"r-")
plt.show()
