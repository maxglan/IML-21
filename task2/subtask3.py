
""" Imports """

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import pandas as pd

""" Functions """

def subtask3(trainf, trainl, test): 
    """
    returns the predicted data using Ridge regression and 10 fold Cross Validation
    """
    
    # 10 folds
    k = 10
    
    label= ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]
    l = np.array([0.1, 1, 10 , 100, 200]) #lambda
    length = len(l)
    y={}
    model={}
    best_lambda = []
    
    prediction = np.zeros((18995, 4))

    # performing 10 fold cross validation for 5 different lambdas on each of the 4 labels
    for i, k in zip(label, range(4)):
    
        y[i] = trainl[i]
        model[i] = []
        
        score = np.zeros(length)
        
        for j in l:
            
            model[i].append( Ridge(j) )
        
            # T returns the array of scores of the estimator for each run of the cross validation
            T = cross_val_score(model[i][j], trainf, y[i], scoring='r2', cv=k)
            score[j] = 0.1 * sum(T)
        
        #fitting our model with the best lambda
        best_lambda = np.argmax(score)
        model[i][best_lambda].fit(trainf , y[i])
        
        prediction[:, k] = model[i][best_lambda].predict(test)
    
    return prediction