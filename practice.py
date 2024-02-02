import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix as cm
df = pd.read_csv("/Users/subhangisati/Downloads/Iris.csv")
X = df.iloc[:,:4]
y = df["Species"]
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size = 0.3)
 
#Applying logistic regression on the imbalanced dataset
clas = LogisticRegression(random_state = 0)
clas.fit(X_train,y_train)
y_pred = clas.predict(X_test)
logistic = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto', max_iter=300)
logistic.fit(X_train, y_train) 
c_matrix=cm(y_test,y_pred)
print("ACCURACY FOR LOGISTIC REGRESSION MODEL~ ",logistic.score(X_test,y_test)) 
sc=StandardScaler()
X_train=sc.fit_transform(X_train) 
X_test=sc.transform(X_test)
print("CONFUSION MATRIC~ \n",c_matrix)
print("\n F1 SCORE~  ",f1_score(y_test,y_pred,average='micro')*100)