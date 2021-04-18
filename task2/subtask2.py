
""" Imports """

import numpy as np
import pandas as pd
from sklearn import svm


""" Functions """

def subtask2(trainf, trainl, test): 
    """
    takes training features, training labels and a test set of features 
    
    returns array with predicted labels
    
    """
    y = trainl.LABEL_Sepsis
    clf = svm.SVR()
    clf.fit(trainf, y)
   
    return clf.predict(test)