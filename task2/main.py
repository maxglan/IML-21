#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: georgengin
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# read the csv file into a Panda df
#train = np.loadtxt(open('train.csv', "rb"), delimiter=",", skiprows=1)
trainf = pd.read_csv("train_features.csv")
trainl = pd.read_csv("train_labels.csv")

# print(trainf.pid[12]) prints the 12+1th item of the pid column

#creates array with all patient ids in the given order
id = trainf.pid.unique()

#list with names of the features
features = list(trainf.columns)

#we get the warning:
#   See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   trainf[f][values] = 0
#   /Users/georgengin/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexing.py:670: SettingWithCopyWarning: 
#   A value is trying to be set on a copy of a slice from a DataFrame

pd.options.mode.chained_assignment = None  # default='warn'


#here we attempt to deal with the missing data points (NaNs)
for i in id:
    for f in features:
        
        #returns indices of rows for a specific patient id
        values = trainf.index[trainf['pid'] == i  ].tolist()
        
        #check whether all entries of a specific feature of a patient are NaNs
        
        #if yes: replace all with 0
        if np.all( np.isnan( trainf[f][values] ) ) == True:
            trainf[f][values] = 0
            
        #else if any entry is a nan eg local minimum
        elif np.any( np.isnan( trainf[f][values] ) ) == True:
            minimum = min( trainf[f][values] )
        
            #check wether specific entry is nan and then replace it with minimum
            for v in values:
                if np.isnan(trainf[f][v]) == True:
                    trainf[f][v] = minimum
            
            
