import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix as cm
data=pd.read_csv("/Users/subhangisati/Desktop/mushrooms.csv") #READING FILE

# GENERATE DUMMY EXCEPT TARGET VAR USING get_dummies
try_column = [list(data.columns)[i] for i in np.arange(1,23)]
data = pd.get_dummies(data, columns=try_column)

# SPLITTING DATA
x = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
#USING train_test_split AND SPLIT 30% DATA FOR TRAINING AND 70% FOR TESTING
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# LOGISTIC REGRESSION 
l = LogisticRegression(random_state =0)
l.fit(x_train,y_train) #FOR TRAINING
y_pred = l.predict(x_test) #FOR TESTING
logistic = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto', max_iter=3000)
logistic.fit(x_train, y_train) 
c_matrix=cm(y_test,y_pred)
print("ACCURACY FOR LOGISTIC REGRESSION MODEL~ ",logistic.score(x_test,y_test)) 
sc=StandardScaler() #STANDADIZE FEATURES 
x_train=sc.fit_transform(x_train) 
x_test=sc.transform(x_test)
print("CONFUSION MATRIX~ \n",c_matrix)
print("\n f1 SCORE~ ",f1_score(y_test,y_pred,average='micro')*100)

# CONFUSION MATRIX VISUAL
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, cmap='Greens', annot=True, fmt='d', cbar=False, yticklabels=['EDIBLE', 'POISINOUS'],
            xticklabels=['PREDICTED EDIBLE', 'PREDICTED POISINOUS'])
plt.show()
