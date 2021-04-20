
""" Imports """

import numpy as np
from sklearn import svm
import pandas as pd

""" Functions """

def subtask3(trainf, trainl, test): 
    """
    returns the predicted data using support vector regression
    """
    
    print(" Start subtask 3.")

    label= ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]
    y={}
    model={}

    prediction = np.zeros((12664, 4))

    # performing 10 fold cross validation for 5 different lambdas on each of the 4 labels
    for i, j in zip(label, range(4)):
        
        print("Training the label " + i + ".")
    
        y[i] = trainl[i]
        model[i] = svm.SVR(kernel= 'rbf')
        model[i].fit(trainf , y[i])
        
        prediction[:, j] = model[i].predict(test)
    
    return prediction