
""" Imports """

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import pandas as pd

""" Functions """

def subtask3(trainf, trainl, test): 
    """
    Arguments. 
    
    Description of the function. 
    
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

    for i, l in zip(label, range(4)):
    
        y[i] = trainl[i]
        model[i] = []
        
        score = np.zeros(length)
        
        for j in range(length):
            
            model[i].append( Ridge(j) )
        
            # T returns the array of scores of the estimator for each run of the cross validation
            T = cross_val_score(model[i][j], trainf, y[i], scoring='r2', cv=k)
            score[j] = 0.1 * sum(T)
        
        best_lambda = np.argmax(score)
        print(best_lambda)
        model[i][best_lambda].fit(trainf , y[i])
        
        prediction[:, l] = model[i][best_lambda].predict(test)
    
    return prediction