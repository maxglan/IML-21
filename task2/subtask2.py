
""" Imports """

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn import svm


""" Functions """

def subtask2(trainf, trainl, test): 
    """
    takes training features, training labels and a test set of features 
    
    returns array with predicted labels
    
    """
    y = trainl.LABEL_Sepsis
    #clf = svm.SVR()
    clf = svm.SVC()
    clf.fit(trainf, y)
    z = clf.decision_function(test)

    return expit(z)

#sample = pd.read_csv("sample.csv")