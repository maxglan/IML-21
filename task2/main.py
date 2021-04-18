#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: georgengin
"""

""" Imports """

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from numba import njit

from subtask1 import subtask1
from subtask2 import subtask2
from subtask3 import subtask3


""" Read the csv file """

trainf = pd.read_csv("train_features.csv")
trainl = pd.read_csv("train_labels.csv")

#creates array with all patient ids in the given order
id = trainf.pid.unique()

#list with names of the features
features = list(trainf.columns)

pd.options.mode.chained_assignment = None  # default='warn'

trainf_arr = trainf.to_numpy(float, True, 0.)


"""  Deal with missing data points """
#@njit
def trying_numba(trainf, num_ids, num_feat):
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
                for v in range(trainf[i:i+12,f].shape[0]):
                    if np.isnan( trainf[i+v,f] ) == True:
                        trainf[i+v,f] = minimum
        #i = i + 12
                    
    return trainf


""" Reshaping to use in SVM """
reshaped_arr = np.zeros((len(id), 37*12))

hallo = np.reshape(trainf_arr[0:0+12, :], (-1,))

for i, id_i in enumerate(np.arange(0, len(id), 12)):
    reshaped_arr[i,:] = np.reshape(trainf_arr[id_i:id_i+12, :], (-1,), order = 'F')
    
#get rid of the multiple patien IDs:
reshaped_arr = reshaped_arr[:, 11:]


trainf_arr = trying_numba(trainf_arr, len(id), len(features))


""" Subtasks """

subtask1(trainf=trainf_arr, trainl=trainl)
subtask2(trainf=trainf_arr, trainl=trainl)
subtask3(trainf=trainf_arr, trainl=trainl)




