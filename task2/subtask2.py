
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
    y = trainl.LABEL_Sepsis
    #clf = svm.SVR()
    # clf = svm.SVC()
    # clf.fit(trainf, y)
    # z = clf.decision_function(test)

    # return expit(z)
    model = xgb.XGBClassifier()
    model.fit( trainf,y)
    
    result = model.predict(test)
    
    return result

#sample = pd.read_csv("sample.csv")