import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
df= pd.read_csv('/Users/subhangisati/Downloads/housing.csv') 
df.head()
df.describe()
X = df.iloc[:, :-1].values#second last column 
y = df.iloc[:,1].values#first column
#OLS of model 
model = sm.OLS(y, X).fit()
summary = model.summary()
print(summary)
#splitting the model
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
r = LinearRegression() #applying linear regression
r.fit(X_train, y_train)#testing 
y_pred = r.predict(X_test)#predicting
plt.scatter(X_train, y_train,color='r') 
plt.plot(X_test, y_pred,color='b') 
plt.show()

