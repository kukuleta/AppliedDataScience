#Import necessary libraries to perform pre-processing on data
#Difflib is a built-in python library providing string matching utilities based on similarity of strings.
#pandas-profiling is a extend of pandas provinding overview of data such as missing values, correlation between features, metadata about feature.
#It is valuable when it comes a moment to inspect data for data pre-processing.

import numpy as np
import pandas as pd
import pandas_profiling
from sklearn import preprocessing
from difflib import SequenceMatcher,get_close_matches

#Receives a string, array of possible strings that the string could be matched with 
#Methods provided in difflib library is not enough to make complex matches.
#For example, They are not well at case-sensivitiy. "X" and "x" are not likely  to look same in point of it's view.
#Returns closest choice in possibleMatches to string parameter.
#For example; "abc", ["aec","sad"] => aec
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

train_data = pandas.read_csv("../Data/bigMartSales/train_set.csv") #Read csv data format and convert it to DataFrame
fatRate = pd.CategoricalIndex(["Low Fat","Regular"]) #Specify correct discrete values for Item_Fat_Content column.
distinctFatRates = pd.CategoricalIndex(pd.unique(train_data["Item_Fat_Content"])) #Get unique discrete values from Item_Fat_Content column
dataNotMatched = distinctFatRates.difference(fatRate) #Compare and take errors arising due to data entry errors.
for entry in dataNotMatched.to_list(): #The loop here carry out getting not-matched entrys and convert them to correct ones by making string comparison 
    closedRate = get_closest_choice(entry,fatRate.to_list())
    getAccurateCategory = get_close_matches(closedRate,fatRate.to_list())
    train_data["Item_Fat_Content"] = train_data["Item_Fat_Content"].replace(entry,getAccurateCategory[0])
    
#Seperate features and labels from training set
train_Y = train_data["Item_Outlet_Sales"]
train_X = train_data.drop(columns=["Item_Outlet_Sales"])

train_X = train_data.drop(columns=["Item_Identifier"])

#The following two lines are related to need to represent categorical values in a appropiate format for training procedure. 
#Encoding schema preferred is One Hot Encoder. 
dummy_variable_list = train_X.select_dtypes(["category"]).tolist()
pd.get_dummies(train_X,columns=dummy_variable_list)