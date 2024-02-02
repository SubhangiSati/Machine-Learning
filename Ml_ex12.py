import pandas as pd
import scipy.stats as stats

# load iris dataset
iris = pd.read_csv("/Users/subhangisati/Downloads/Iris.csv")

summary = iris.describe()

# Compute skewness and kurtosis for each variable
skewness = iris.skew()
kurtosis = iris.kurtosis()

# Add skewness and kurtosis to summary dataframe
summary.loc['skewness'] = skewness
summary.loc['kurtosis'] = kurtosis

# Print summary dataframe
print(summary)

# Print minimum and maximum values for each variable
print('Minimum values:')
print(iris.min())
print('\nMaximum values:')
print(iris.max())
