
""" Imports """

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn import svm
import xgboost as xgb



""" Functions """

def subtask2(trainf, trainl, test): 
    """
    takes training features, training labels and a test set of features 
    
    returns array with predicted labels
    
    """
    
    print(" Start subtask 2 - Training the label Sepsis")
    
    y = trainl.LABEL_Sepsis
    model = xgb.XGBClassifier()
    model.fit( trainf,y)
    result = model.predict_proba(test)
    prediction = expit(result[:,1])
    
    return prediction
