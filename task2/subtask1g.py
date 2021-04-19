#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 10:23:32 2021

@author: georgengin
"""


import numpy as np
import pandas as pd
from sklearn import svm
import xgboost as xgb
from scipy.special import expit
from numba import njit

""" Functions """
#@njit
def subtask1(trainf, trainl, test): 
    """
    takes training features, training labels and a test set of features 
    
    returns array with predicted labels
    
    """
    print(" Start subtask 1.")

    labels = ["LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", 
              "LABEL_Alkalinephos", "LABEL_Bilirubin_total", "LABEL_Lactate", 
              "LABEL_TroponinI", "LABEL_SaO2", "LABEL_Bilirubin_direct", 
              "LABEL_EtCO2"]

    model={}

    prediction = np.zeros((len(test), len(labels)))

    for l, i in zip(labels, range(len(labels))):
        model[l] = svm.SVC(kernel='rbf')
        model[l].fit(trainf, trainl[l])
        
        z = model[l].decision_function(test)
        print("Training the label " + l + ".")
        prediction[:,i] = expit(z)
    
    print( "End subtask 1 ")
    
    return prediction
