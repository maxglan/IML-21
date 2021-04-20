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
from subtask3g import subtask3


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
    """ Returns the preprocessed and reshaped array 
    the pid and age just got one entry per row now """
    
    print(" Deal with missing data.")
    
    #Creates a numpy array out of the pd dataframe
    arr = t_arr.to_numpy(float, True)
    # t_reshaped = np.zeros((num_ids, 37*12))
    # here we already handle the pid, time and age directly (pid:1 column, time 12, age:1)
    t_reshaped = np.zeros((num_ids, 37*12- 2*11))
    num_feat_out= 12
    
    for i,j in zip(np.arange(0, num_ids*12, 12), range(num_ids)):
        #direct entry for pid and age
        t_reshaped[j,0] = arr[i,0]
        t_reshaped[j,1:13] = arr[i: i+12, 1]    
        t_reshaped[j,14] = arr[i,2] 
        
        for f in np.arange(3, num_feat):
            #check whether all entries of a specific feature of a patient are NaNs
            
            #if yes: replace all with 0
            if np.all( np.isnan( arr[i:i+12,f] ) ) == True:
                t_reshaped[j, 14+(f-3)*num_feat_out: 14 + (f-3)*num_feat_out + num_feat_out]=0
                
            #else if any entry is a nan eg local minimum
            elif np.any( np.isnan( arr[i:i+12,f] ) ) == True:
                minimum = np.nanmin( arr[i:i+12,f] )
            
                #check wether specific entry is nan and then replace it with minimum
                for v in range(12):
                    if np.isnan( arr[i+v,f] ) == True:
                        t_reshaped[j, 14+(f-3)*num_feat_out + v] = minimum
    
    return t_reshaped
      
def deal_with_nans_badly(t_arr, num_ids, num_feat):
    """ Returns the preprocessed and reshaped array 
    the pid and age just got one entry per row now, and we reduce to 5 features"""
    
    num_feat_out=5
    
    print(" Deal with missing data.")
    
    #Creates a numpy array out of the pd dataframe
    # 5 features -> 34*5 for other features
    #we only make one entry for age and pid, but 12 for the time
    arr = t_arr.to_numpy(float, True)
    t_reshaped = np.zeros((num_ids, 34*num_feat_out + 2 + 12))
    
    for i,j in zip(np.arange(0, num_ids*12, 12), range(num_ids)):
        # direct entry for pid and age
        # t_reshaped[j,0] = arr[i,0]
        # t_reshaped[j,1] = arr[i,1]
        
        t_reshaped[j,0] = arr[i,0]
        t_reshaped[j,1:13] = arr[i: i+12, 1]    
        t_reshaped[j,14] = arr[i,2] 
        
        for f in np.arange(3, num_feat):
            #check whether all entries of a specific feature of a patient are NaNs
            
            a=arr[i:i+12,f]
            
            #if yes: replace all with 0
            if np.all( np.isnan( a ) ) == True:
                t_reshaped[j, 14+(f-3)*num_feat_out: 14 + (f-3)*num_feat_out + num_feat_out]=0
                
            #else if any entry is a nan eg local minimum
            elif np.any( np.isnan( a ) ) == True:

                mean = np.nanmean( a )
                minimum = np.nanmin( a )
                maximum = np.nanmax( a )
                
                start = a[np.isfinite(a)][0]
                end = a[np.isfinite(a)][-1]          
                trend = end-start
                
                number = np.count_nonzero(~np.isnan(a))
                
                t_reshaped[j, 14+(f-3)*num_feat_out]=mean
                t_reshaped[j, 14+(f-3)*num_feat_out + 1]= trend
                t_reshaped[j, 14+(f-3)*num_feat_out + 2]= minimum
                t_reshaped[j, 14+(f-3)*num_feat_out + 3]= maximum
                t_reshaped[j, 14+(f-3)*num_feat_out + 4]= number
    
    return t_reshaped

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
    

"""calculating nans"""
#returns properly reshaped and filled arrays
train_features = deal_with_nans(trainf, len(idtrain), len(features))
test_features = deal_with_nans(testf, len(idtest), len(features))

# normalised versions
norm_train_features, norm_test_features = normalize_combined(train_features, test_features)

# """ calculating nans badly"""
# #returns badly reshaped and filled arrays
# train_features_bad = deal_with_nans_badly(trainf, len(idtrain), len(features))
# test_features_bad = deal_with_nans_badly(testf, len(idtest), len(features))

# # normalised bad versions
# norm_train_features_bad, norm_test_features_bad = normalize_combined(train_features_bad, test_features_bad)

""" Subtasks """

# prediction1 = subtask1(train_features , trainl, test_features )
prediction1 = subtask1(norm_train_features , trainl, norm_test_features )

# prediction2 = subtask2(train_features , trainl, test_features )
prediction2 = subtask2(norm_train_features , trainl, norm_test_features )

prediction3 = subtask3(norm_train_features , trainl, norm_test_features )

"""bad subtasks"""
# prediction1 = subtask1(norm_train_features_bad , trainl, norm_test_features_bad )

# prediction2 = subtask2(norm_train_features_bad , trainl, norm_test_features_bad )

# prediction3 = subtask3(train_features_bad , trainl, test_features_bad )


""" Combining and converting the subtask's output"""
df = pd.read_csv("sample.csv")

df[["LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", 
              "LABEL_Alkalinephos", "LABEL_Bilirubin_total", "LABEL_Lactate", 
              "LABEL_TroponinI", "LABEL_SaO2", "LABEL_Bilirubin_direct", 
              "LABEL_EtCO2"]] = prediction1

df["LABEL_Sepsis"] = prediction2

df[["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]] = prediction3

df.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')