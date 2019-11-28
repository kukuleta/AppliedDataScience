#Import necessary libraries to perform pre-processing on data
#Difflib is a built-in python library providing string matching utilities based on similarity of strings.
#pandas-profiling is a extend of pandas provinding overview of data such as missing values, correlation between features, metadata about feature.
#It is valuable when it comes a moment to inspect data for data pre-processing.

import numpy as np
import pandas as pd
import pandas_profiling
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from difflib import SequenceMatcher,get_close_matches
from sklearn.datasets import load_boston
from sklearn import pipeline
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV
from sklearn.model_selection import train_test_split

"""Receives a string, array of possible strings that the string could be matched with 
Methods provided in difflib library is not enough to make complex matches.
For example, They are not well at case-sensivitiy. "X" and "x" are not likely  to look same in point of it's view.
Returns closest choice in possibleMatches to string parameter."""

For example; "abc", ["aec","sad"] => aec
def get_closest_choice(string,possibleMatches):
    max_ratio = 0.0
    closest_match = ""
    for possibleMatch in map(str.lower,possibleMatches): #Traverse over possible matches to quantify how much each choice is likely to match.
        ratioBetween = SequenceMatcher(None,string.lower(),possibleMatch).ratio()
        if ratioBetween > max_ratio:
            max_ratio = ratioBetween
            closest_match = possibleMatch
    return closest_match

#The following lines provide a way to replace typing erros in Item_Fat_Content feature and replaces them with correct ones.

train_data = pd.read_csv("../data/bigMartSales.csv") #Read csv data format and convert it to DataFrame
fatRate = pd.CategoricalIndex(["Low Fat","Regular"]) #Specify correct discrete values for Item_Fat_Content column.
distinctFatRates = pd.CategoricalIndex(pd.unique(train_data["Item_Fat_Content"])) #Get unique discrete values from Item_Fat_Content column
dataNotMatched = distinctFatRates.difference(fatRate) #Compare and take errors arising due to data entry errors.
for entry in dataNotMatched.to_list(): #The loop here carry out getting not-matched entrys and convert them to correct ones by making string comparison 
    closedRate = get_closest_choice(entry,fatRate.to_list())
    getAccurateCategory = get_close_matches(closedRate,fatRate.to_list())
    train_data["Item_Fat_Content"] = train_data["Item_Fat_Content"].replace(entry,getAccurateCategory[0])
    
#Seperate features and labels from training set and remove Item_Identifier because it's not relevant information

Train_X = train_data.drop(columns=["Item_Outlet_Sales","Item_Identifier"])
Train_Y = train_data["Item_Outlet_Sales"]
Test_X = train_data.drop(columns=["Item_Identifier"])

#The section of code handles with missing values by imputing them.

Train_X['Item_Weight'].fillna((Train_X['Item_Weight'].mean()), inplace=True)
Train_X['Item_Visibility'] = Train_X['Item_Visibility'].replace(0,np.mean(Train_X['Item_Visibility']))
Train_X['Outlet_Establishment_Year'] = 2013 - Train_X['Outlet_Establishment_Year']
Train_X['Outlet_Size'].fillna('Small',inplace=True)

dummy_variable_list = Train_X.select_dtypes(["object"]).columns.tolist()
dummies = pd.get_dummies(Train_X, columns = dummy_variable_list)
Train_X.drop(dummy_variable_list, axis = 1, inplace = True)
Train_X = pd.concat([Train_X, dummies], axis = 1)

"""The following two lines are related to need to represent categorical values in a appropiate format for training procedure. 
Encoding schema preferred is One Hot Encoder. 
Categorical values are represented in DataFrame as object rather than categorical. 
Therefore, I filtered accorging to object dtype rather than categorical"""

dummy_variable_list = Train_X.select_dtypes(["object"]).columns.tolist()
dummies = pd.get_dummies(Train_X, columns = dummy_variable_list)
Train_X.drop(dummy_variable_list, axis = 1, inplace = True)
Train_X = pd.concat([Train_X, dummies], axis = 1)

"""The method train_test_split is related to over-fitting problem. train_set_split is doing shuffling operation on your data and take random subset 
according to value specified in test_size parameter. 
The paramater random_state determines the behaviour you want randomized and take subset of data or not."""

Train_X, Test_X , Train_Y, Test_Y = train_test_split(Train_X, Train_Y, test_size = 0.2, random_state = 1)

#The first attempt is to try ordinal linear regression model. Train & Score 

print(OrdLinearReqression.score(Test_X,Test_Y)) #Let's see how much likely our model is matched exactly to response variable.


