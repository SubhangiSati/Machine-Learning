from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import seaborn as sns
data = load_iris()

print(data.data[:5])
print(data.feature_names)
data = load_iris()
print(data.target[:10])
print(data.target_names)

sns.set_theme(style="darkgrid")
df = sns.load_dataset("iris")
sns.pairplot(df,hue='species',palette='icefire')
from sklearn import metrics
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
LogR = LogisticRegression(C=0.02)
LogR.fit(X_train,y_train)
yhat=LogR.predict(X_test)
print(yhat[0:5])
print(y_test[0:5])
print("LogR Accuracy = ", metrics.accuracy_score(y_test,yhat))
