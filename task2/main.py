#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Imports """

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from numba import njit

from subtask1g import subtask1
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

    Returns the preprocessed and reshaped array
    -------    
    If all the data of a patient's feature is missing, we set all the value to zero.
    If only some data is missing, we set the nans to the minimum of that patient's
    feature.
    """
    
    #Creates a numpy array out of the pd dataframe
    arr = t_arr.to_numpy(float, True)
    t_reshaped = np.zeros((num_ids, 37*12))
    
    for i in np.arange(0, num_ids*12, 12):
        for f in range(num_feat):
            #check whether all entries of a specific feature of a patient are NaNs
            
            #if yes: replace all with 0
            if np.all( np.isnan( arr[i:i+12,f] ) ) == True:
                arr[i:i+12,f] = 0
                
            #else if any entry is a nan eg local minimum
            elif np.any( np.isnan( arr[i:i+12,f] ) ) == True:
                minimum = np.nanmin( arr[i:i+12,f] )
            
                #check wether specific entry is nan and then replace it with minimum
                for v in range(12):
                    if np.isnan( arr[i+v,f] ) == True:
                        arr[i+v,f] = minimum
                        
    """ Reshaping to use in SVM """
    
    # for i, id_i in enumerate(np.arange(0, num_ids*12, 12)):
    #     t_reshaped[i,:] = np.reshape(arr[id_i:id_i+12, :], (-1,), order = 'F')
    for i in range(num_ids):
        t_reshaped[i,:] = np.reshape(arr[i*12: i*12 +12, :], (-1,), order = 'F')
        
    #get rid of multiple patient IDs:
    t = t_reshaped[:, 11:]
    
    return t, arr
      

""" if we use SVM we first have to normalize the data using maxabsscalar (good for data with many 0s)"""
# def normalize(arr):

#returns properly reshaped and filled arrays
train_features, tarr = deal_with_nans(trainf, len(id), len(features))
test_features , tarr2 = deal_with_nans(testf, len(id), len(features))


""" Subtasks """
# prediction1 = subtask1(train_features , trainl, test_features )
# prediction2 = subtask2(train_features , trainl, test_features )
prediction3 = subtask3(train_features , trainl, test_features )

"""combining and converting the subtask's output to the wanted output file"""
df = pd.read_csv("sample.csv")
df[:,1:10] = prediction1
df[:,11] = prediction2
df[:,12:] = prediction3
df.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')


