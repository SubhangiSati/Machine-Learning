import pandas as pd
import scipy.stats as stats

# LOAD IRIS
iris = pd.read_csv("/Users/subhangisati/Downloads/Iris.csv")

summary = iris.describe()

#SKEWNESS AND KUTOSIS FOR EACH VAR
skewness = iris.skew()
kurtosis = iris.kurtosis()

# ADD S & K TO SUMMARY DF
summary.loc['skewness'] = skewness
summary.loc['kurtosis'] = kurtosis

# SUMMARY
print(summary)

# MIN &MAX FOR EACH
print('Minimum values:')
print(iris.min())
print('\nMaximum values:')
print(iris.max())
