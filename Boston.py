#MULTIPLE LINEAR REGRESSION
#import libraries 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

#read file
d = pd.read_csv("/Users/subhangisati/Downloads/boston.csv")
d.head()
d.info()

x = d.iloc[:, :3]#all parameters of x 
x.info()

y = d.iloc[:, 3]
y.info()

#split the data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

#apply linear regression
model = LinearRegression()

#train
model.fit(x_train, y_train)

#test
lab_pred = model.predict(x_test)


#Accuracy
accuracy = model.score(x_test, y_test)

print("Accuracy :", accuracy*100, "%")

