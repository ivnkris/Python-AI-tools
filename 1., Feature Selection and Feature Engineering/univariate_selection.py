import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
X = data.iloc[:, 0:4]    #coloumns considered for evaluation
Y = data.iloc[:, -1]

# apply SelectKBest to extract best 3 features using chi-squared
bestfeatures = SelectKBest(score_func=chi2, k=3)
fit = bestfeatures.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)

dfcoloumns = pd.DataFrame(X.columns)
scores = pd.concat([dfcoloumns, dfscores], axis=1)
scores.columns = ['specs', 'score']
print(scores.nlargest(3, 'score'))