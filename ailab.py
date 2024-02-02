import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("/Users/subhangisati/Desktop/mushrooms.csv")
print(data.head(10)) #to print first 10 rows
df=pd.DataFrame(data) #to print whole data
print(df)

#TO CHECK FOR BALANCED AND IMBALANCED 
print(df['class'].value_counts())

#PROVE IT IS IMBALANCED BY PLOTTING GRAPH
plt.title("GRAPH SHOWS DIFFERENT SPECIES OF MUSHROOM")
plt.hist(data["class"])
plt.show()














