#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
testf = pd.read_csv("train_features.csv")


#creates array with all patient ids in the given order
id = trainf.pid.unique()

#list with names of the features
features = list(trainf.columns)

#Creates a numpy array out of the pd dataframe
trainf_arr = trainf.to_numpy(float, True)
testf_arr = testf.to_numpy(float, True)


"""  Deal with missing data points """
#@njit
def deal_with_nans(t_arr, num_ids, num_feat):
    """
    Parameters
    ----------
    t_arr : ndarray
        contains the training data
    num_ids : int64
        number of patient ids
    num_feat : TYPE
        number of features (== columns of t_arr)

    Returns
    -------
    Nothing, t_arr is call-by-reference
    
    If all the data of a patient's feature is missing, we set all the value to zero.
    If only some data is missing, we set the nans to the minimum of that patient's
    feature.
    """
    for i in np.arange(0, num_ids*12, 12):
        for f in range(num_feat):
            #check whether all entries of a specific feature of a patient are NaNs
            
            #if yes: replace all with 0
            if np.all( np.isnan( t_arr[i:i+12,f] ) ) == True:
                t_arr[i:i+12,f] = 0
                
            #else if any entry is a nan eg local minimum
            elif np.any( np.isnan( t_arr[i:i+12,f] ) ) == True:
                minimum = np.nanmin( t_arr[i:i+12,f] )
            
                #check wether specific entry is nan and then replace it with minimum
                for v in range(12):
                    if np.isnan( t_arr[i+v,f] ) == True:
                        t_arr[i+v,f] = minimum
                        
    """ Reshaping to use in SVM """
    t_reshaped = np.zeros((len(id), 37*12))
    
    for i, id_i in enumerate(np.arange(0, len(id), 12)):
        t_reshaped[i,:] = np.reshape(t_arr[id_i:id_i+12, :], (-1,), order = 'F')
        
    #get rid of multiple patient IDs:
    t_reshaped = t_reshaped[:, 11:]
    
    return t_reshaped
                
#returns properly reshaped and filled arrays
train_features = deal_with_nans(trainf_arr, len(id), len(features))
test_features = deal_with_nans(testf_arr, len(id), len(features))


""" Subtasks """

k1 = subtask1(trainf=train_features, trainl=trainl, test=test_features)
# k = subtask2(train_features , trainl, test_features )
prediction = subtask3(train_features , trainl, test_features )



