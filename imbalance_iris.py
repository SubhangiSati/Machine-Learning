#IMBALANCE THE IRIS DATASET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
d_f = pd.read_csv("/Users/subhangisati/Desktop/mushrooms.csv") #READING FILE

d_f.drop([10,20,30,40,50,60,400,70,80,90,6234,933,112,1435,1323,1121,3216,1222,1243,7656,8013,4367,2937,1897,1934,1644,2376,1137,84,76,87,1000,2000,12,34,55,23,1234,829,213,435,2423,6453,800,900,200,300,400,500,5000,5555,4342,653,34,54,234,54,4322,123,543,58,999],inplace=True) #DROP FEW VALUES USING .drop() FOR IMBALANCING DATASET
print(d_f)
x = d_f["class"] 
plt.hist(x, bins=15, color="green") #GRAPH PLOTTING
plt.title("IMBALANCING MUSHROOMS DATASET")
plt.xlabel("SPECIES")
plt.ylabel("COUNT")
plt.show()

