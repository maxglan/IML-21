#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Imports """

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MaxAbsScaler
from numba import njit

from subtask1g import subtask1
from subtask2 import subtask2
from subtask3 import subtask3

from score_submission import get_score, TESTS


""" Read the csv file """

print(" Read CSV file.")
trainf = pd.read_csv("train_features.csv")
trainl = pd.read_csv("train_labels.csv")
testf = pd.read_csv("test_features.csv")


#creates array with all patient ids in the given order
idtrain = trainf.pid.unique()
idtest = testf.pid.unique()

#list with names of the features
features = list(trainf.columns)

"""  Deal with missing data points """
#@njit
def deal_with_nans(t_arr, num_ids, num_feat):
    """ Returns the preprocessed and reshaped array """
    
    print(" Deal with missing data.")
    
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
    for i in range(num_ids):
        t_reshaped[i,:] = np.reshape(arr[i*12: i*12 +12, :], (-1,), order = 'F')
        
    #get rid of multiple patient IDs:
    t = t_reshaped[:, 11:]
    
    return t
      

""" Normalize the data """
# If we use non-linear SVM we first have to normalize the data using 
# maxabsscalar (good for data with many 0s)
def normalize(arr):
    # Scaling to [-1,1]
    transformer = MaxAbsScaler().fit(arr)
    norm_arr = transformer.transform(arr)
    
    return norm_arr
    
def normalize_combined(train_features, test_features):
    """ Idea: The function combines the two set of features and normalizes them as whole.  """
    
    print(" Normalize the data.")
    
    all_features = np.concatenate((train_features, test_features))
    norm_all_features = normalize(all_features)
    
    norm_train_features = norm_all_features[:len(train_features), :]
    norm_test_features = norm_all_features[len(train_features):, :]
    
    return norm_train_features, norm_test_features
    

#returns properly reshaped and filled arrays
train_features = deal_with_nans(trainf, len(idtrain), len(features))
test_features = deal_with_nans(testf, len(idtest), len(features))

# normalised versions
norm_train_features, norm_test_features = normalize_combined(train_features, test_features)


""" Subtasks """

# prediction1 = subtask1(train_features , trainl, test_features )
prediction1 = subtask1(norm_train_features , trainl, norm_test_features )

# prediction2 = subtask2(train_features , trainl, test_features )
prediction2 = subtask2(norm_train_features , trainl, norm_test_features )

prediction3 = subtask3(train_features , trainl, test_features )


""" Combining and converting the subtask's output"""
df = pd.read_csv("sample.csv")

df[["LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", 
              "LABEL_Alkalinephos", "LABEL_Bilirubin_total", "LABEL_Lactate", 
              "LABEL_TroponinI", "LABEL_SaO2", "LABEL_Bilirubin_direct", 
              "LABEL_EtCO2"]] = prediction1

df["LABEL_Sepsis"] = prediction2

df[["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]] = prediction3

df.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')