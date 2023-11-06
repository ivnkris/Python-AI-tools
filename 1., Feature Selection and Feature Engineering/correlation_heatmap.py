import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
X = data.iloc[:, 0:4]    #coloumns considered for evaluation
Y = data.iloc[:, -1]     #pick last coloumn for target feature

#get the correlation of each feature in the dataset
correlation_matrix = data.corr(numeric_only=True)
top_corr_features = correlation_matrix.index
plt.figure(figsize=(20,20))

# plot heat map with seaborn
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap='RdYlGn')
plt.show()