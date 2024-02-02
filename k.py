import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder#CATEGORIAL TO NUMERICAL VARIABLE 
from sklearn.model_selection import train_test_split#DATASET DIVIDED IN TRAINING AND TESTING
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns 
data_set = pd.read_csv("/Users/subhangisati/Downloads/Iris.csv")
#print(data_set)
#print(data_set['Species'].unique())
#print(data_set.isnull().values.any())
#print(data_set.plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm'))
sns.pairplot(data_set,hue='Species',height=2).add_legend()
