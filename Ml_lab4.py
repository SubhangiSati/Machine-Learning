'''import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plotting and Visualizing data
import os

import statsmodels.api as sm
data = pd.read_csv('/Users/subhangisati/Downloads/Iris.csv')
x=data["Species"]
y=data.drop(["Species"],axis=1)

x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
summary = model.summary()
print(summary)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)
print(x[:10])
print('\n')
print(y[:10])
from sklearn.linear_model import LinearRegression

# create an empty linear regression model like below and give it a good variable name
radio_model = LinearRegression()

# to create the model, we use fit(x,y)
radio_model.fit(x,y)


y_pred = radio_model.predict(x)
plt.scatter(x,y,color = 'b')
plt.plot(x,radio_model.predict(x),color = 'r')
plt.title('Sales v/s Radio Budget')
plt.xlabel('Sales')
plt.ylabel('Radio Budget')
plt.show()

plt.scatter(x, y)'''

import pandas
import statsmodels.api as sm
df = pandas.read_csv("/Users/subhangisati/Downloads/restaurants.csv")
X = df['Food_Quality']
Y = df['Price']
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
summary = model.summary()
print(summary)
'''import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x='Food_Quality',y='Price',hue='Food_Quality')
plt.show()'''
