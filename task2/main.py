#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: georgengin
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from numba import njit

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

trainf_arr = trainf.to_numpy(float, True, 0.)

#@njit
def trying_numba(trainf, num_ids, num_feat):
    #here we attempt to deal with the missing data points (NaNs)
    for i in range(num_ids):
        for f in range(num_feat):
            
            #check whether all entries of a specific feature of a patient are NaNs
            
            #if yes: replace all with 0
            if np.all( np.isnan( trainf[i:i+12,f] ) ) == True:
                trainf[i:i+12,f] = 0
                
            #else if any entry is a nan eg local minimum
            elif np.any( np.isnan( trainf[i:i+12,f] ) ) == True:
                minimum = np.nanmin( trainf[i:i+12,f] )
            
                #check wether specific entry is nan and then replace it with minimum
                for v in range(len(trainf[i:i+12,f])):
                    if np.isnan( trainf[i+v,f] ) == True:
                        trainf[i+v,f] = minimum
        i = i + 12
                    
    return trainf


reshaped_arr = np.zeros((len(id), 37*12))

hallo = np.reshape(trainf_arr[0:0+12, :], (-1,))

for i, id_i in enumerate(np.arange(0, len(id), 12)):
    reshaped_arr[i,:] = np.reshape(trainf_arr[id_i:id_i+12, :], (-1,), order = 'F')
    
#get rid of the multiple patien IDs:
reshaped_arr = reshaped_arr[:, 11:]


#trainf_arr = trying_numba(trainf_arr, len(id), len(features))
