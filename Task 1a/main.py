#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

#read the csv file
train = np.loadtxt(open('train.csv', "rb"), delimiter=",", skiprows=1)

l=np.array([0.1, 1, 10 , 100, 200]) #lambda
#10 folds
k=10

for i in l:
#performing k fold validation for each lambda
    model=Ridge(i)
    #extracting the training data and the results
    X=train[:, 1:]
    y=train[:, 0]
    Y=np.reshape(y,(150,1))
    #T returns the array of scores of the estimator for each run of the cross validation
    T=cross_val_score(model, X, Y, scoring= 'neg_root_mean_squared_error' , cv=k)
    ##takes the mean of the RMSEs for each lambda
    solution=np.sum(T*0.1,axis=1).tolist()
    print(solution)
    
    #export to csv
    f = open('solution.csv','w')
    f.write(solution) #Give your csv text here.
    f.close()




#KFold(n_splits=10)
#model.fit(X, y)
#RMSE = mean_squared_error(ymean, ypredict)**0.5
#print(RMSE)
