import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt

# import your dataset here
data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
X = data.iloc[:, 0:4]    #coloumns considered for evaluation
Y = data.iloc[:, -1]

model = ExtraTreesClassifier()
model.fit(X, Y)

# use inbuilt feature_importances_ class of tree based classifier
print(model.feature_importances_)

# plot graph of 5 most important feature for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()