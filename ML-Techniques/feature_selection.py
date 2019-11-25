from sklearn import datasets
from sklearn.feature_selection import SelectKBest, mutual_info_regression , f_regression
import matplotlib.pyplot as plt
import numpy as np

#Import built-in diabetes data set in sklearn library.
diabetes = datasets.load_diabetes()

#Seperate data into train and test splits.
train_X = diabetes.data[:-42]
train_Y = diabetes.target[:-42]

test_X = diabetes.data[-42:]
test_Y = diabetes.target[:-42]

"""Feature selection is a best practise if you are struggling with high-dimensional space 
and want to eliminiate some feature subset and give more weight to higher-priority ones.
Some features are unlikely to change your model's accuracy because of it's unimportance. 
Univariate Feature selection comes handy in these cases. It handles with un-wanted features 
in following ways.Firstly, assign a score to each feature by evaulating them with one of methods 
from mutual-info-regression or f-score and filtered out according to below techniques
SelectKBest = Select k best feature and eliminate rest.
SelectPercentile = Take out of features that have lower percentiles than user-speficied percentile input."""

indices = np.arange(train_X.shape[-1])
mutualInfoScore = mutual_info_regression(train_X[:,:],train_Y)
fScore = list(f_regression(train_X[:,:],train_Y))[0].tolist()

plt.figure(figsize=(15, 5))
plt.subplot(221)
plt.bar(indices , mutualInfoScore , width = 0.1,label=r'Mutual information score', color='darkorange',edgecolor='black')#tick_label=[i + 1 for i in range(10)])
plt.xlabel("Feature number")
plt.ylabel("Mutual Info Score")
plt.subplot(222)
plt.bar(indices, fScore , width = 0.1,label=r'F score', color='navy',edgecolor='black')#tick_label=[i + 1 for i in range(10)])
plt.xlabel("Feature number")
plt.ylabel("F Score")
plt.show()

X_new = SelectKBest(mutual_info_regression, 5).fit_transform(train_X,train_Y)


