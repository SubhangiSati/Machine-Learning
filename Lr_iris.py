import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt
d_f = pd.read_csv("/Users/subhangisati/Downloads/Iris.csv")  #READING FILE
X = d_f.iloc[:,:4]
Y = d_f["Species"] #Y HAVE ALL 3 SPECIES
X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size = 0.4) 
#used train_test_split we used 30% for training and 70% for testing


#NOW USING LOGISTIC REGRESSION ON IRIS DATASET
l = LogisticRegression(random_state =0)
l.fit(X_train,Y_train) #FOR TRAINING
Y_pred = l.predict(X_test) #FOR TESTING
logistic = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto', max_iter=3000)
logistic.fit(X_train, Y_train) 
c_matrix=cm(Y_test,Y_pred)
print("ACCURACY FOR LOGISTIC REGRESSION MODEL~ ",logistic.score(X_test,Y_test)) 
sc=StandardScaler() #STANDADIZE FEATURES 
X_train=sc.fit_transform(X_train) 
X_test=sc.transform(X_test)
print("CONFUSION MATRIX~ \n",c_matrix)
print("\n f1 SCORE~ ",f1_score(Y_test,Y_pred,average='micro')*100)


#USING SVM ON IRIS DATASET
cf= SVC(random_state=0) 
cf.fit(X_train,Y_train)
print("ACCURACY OF SVM MODEL~ ", cf.score(X_test,Y_test))
print("CONFUSION MATRIX~ \n",c_matrix)
print("\n F1 SCORE~ ",f1_score(Y_test,Y_pred,average='micro')*100)

#USING RBF ON IRIS DATASET
cf= SVC(kernel='rbf',random_state=0) #PARAMETERS HAVE DEFAULT VALUES
cf.fit(X_train,Y_train)
print("ACCURACY OF RBF SVM MODEL~ ", cf.score(X_test,Y_test)) 
print("CONFUSION MATRIX~ \n",c_matrix)
print("\n F1 SCORE~ ",f1_score(Y_test,Y_pred,average='micro')*100)