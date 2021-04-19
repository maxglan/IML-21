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
    
    # for i, id_i in enumerate(np.arange(0, num_ids*12, 12)):
    #     t_reshaped[i,:] = np.reshape(arr[id_i:id_i+12, :], (-1,), order = 'F')
    for i in range(num_ids):
        t_reshaped[i,:] = np.reshape(arr[i*12: i*12 +12, :], (-1,), order = 'F')
        
    #get rid of multiple patient IDs:
    t = t_reshaped[:, 11:]
    
    return t
      

""" Normalize the data """
# If we use non-linear SVM we first have to normalize the data using 
# maxabsscalar (good for data with many 0s)
def normalize(arr):
    """
    Parameters
    ----------
    arr : ndarray
    
    Returns
    -------
    norm_arr: ndarray
        Normalized and scaled version of arr. We normalize 
        each feature (column)separately by its absolute value. 
        
        For example: 
        arr =  [[ 1., -1.,  2.],        norm_arr = [[ 0.5, -1. ,  1. ],
               [ 2.,  0.,  0.],                    [ 1. ,  0. ,  0. ],
               [ 0.,  1., -1.]]                    [ 0. ,  1. , -0.5]])

    """
    
    # Scaling to [-1,1]
    transformer = MaxAbsScaler().fit(arr)
    norm_arr = transformer.transform(arr)
    
    return norm_arr
    
def normalize_combined(train_features, test_features):
    """
    Idea
    ----------
    The function combines the two set of features and normalizes them as whole. 

    Parameters
    ----------
    train_features: ndarray
    test_features: ndarray

    Returns
    -------
    norm_train_features: ndarray
    norm_test_features: ndarray
    
    Example: 
        train_features = [[ 1., -1.],   norm_train_features = [[ 0.01 -1.  ]
                          [ 2.,  0.],                          [ 0.02  0.  ]
                                                               [ 0.02  0.  ]
                          [ 0.,  1.]])

         test_features = [[ 100., -1.], norm_train_features = [[ 1.  -1. ]
                          [50.0,  0.]                          [ 0.5  0. ]
                          [ 0,  1.]]                           [ 0.   1. ]]
    """
    
    print(" Normalize the data.")
    
    all_features = np.concatenate((train_features, test_features))
    norm_all_features = normalize(all_features)
    
    norm_train_features = norm_all_features[:len(train_features), :]
    norm_test_features = norm_all_features[len(train_features):, :]
    
    return norm_train_features, norm_test_features
    

#returns properly reshaped and filled arrays
train_features = deal_with_nans(trainf, len(id), len(features))
test_features = deal_with_nans(testf, len(id), len(features))

# normalised versions
norm_train_features, norm_test_features = normalize_combined(train_features, test_features)


""" Subtasks """

# prediction1 = subtask1(train_features , trainl, test_features )
#prediction1 = subtask1(norm_train_features , trainl, norm_test_features )
prediction1 = pd.read_csv("sample.csv")[:,1:10]


# prediction2 = subtask2(train_features , trainl, test_features )
prediction2 = subtask2(norm_train_features , trainl, norm_test_features )

prediction3 = subtask3(train_features , trainl, test_features )


""" Combining and converting the subtask's output"""

print(" Store predictions.")

column_names = list(pd.read_csv("sample.csv").columns)
print(column_names)

pid_list = pd.read_csv("test_features.csv")['pid'].values
print(pid_list)
   
#prediction1 = pd.read_csv("sample.csv")[:,1:10]

df = pd.DataFrame(columns=column_names)

for i in range(len(pid_list)): 
    new_row = [pid_list[i]]
    new_row.append(prediction1[i])
    new_row.append(prediction2[i])
    new_row.append(prediction3[i])
    df.append(new_row, ignore_index=True)

#df = pd.read_csv("sample.csv")
#df[:,1:10] = prediction1
#df[:,11] = prediction2
#df[:,12:] = prediction3

df.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')


""" Score submission """

df_submission = pd.read_csv('prediction.zip')

# generate a baseline based on sample.zip
df_true = pd.read_csv('test_features.zip')
for label in TESTS + ['LABEL_Sepsis']:
    # round classification labels
    df_true[label] = np.around(df_true[label].values)

print('Score of sample.zip with itself as groundtruth', get_score(df_true, df_submission))


