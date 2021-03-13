#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# read the csv file
train = np.loadtxt(open('train.csv', "rb"), delimiter=",", skiprows=1)

l = np.array([0.1, 1, 10 , 100, 200]) #lambda
length = len(l)
# 10 folds
k = 10

# array in which the solutions are stored
solution = np.zeros(length)

# Set seed
np.random.seed(42)

for i,j in zip(l, range(length)):
# performing k fold validation for each lambda
    model = Ridge(i)
    # extracting the training data and the results
    X = train[:, 1:]
    y = train[:, 0]
    
    # create cross validation groups
    labels = np.arange(0, len(y)) % k
    np.random.shuffle(labels)
    
    train_indices = [[n for n in range(len(y)) if labels[n] != m] for m in range(k)]
    test_indices = [[n for n in range(len(y)) if labels[n] == m] for m in range(k)]

    cv = zip(train_indices, test_indices) 

    # T returns the array of scores of the estimator for each run of the cross validation
    T = cross_val_score(model, 
                        X, 
                        y, 
                        scoring='neg_root_mean_squared_error', 
                        cv=cv)
    print(T)
    # takes the mean of the RMSEs for each lambda and stores it in solutions
    solution[j] = 0.1 * sum(-T)

# export to csv
np.savetxt("solution.csv", solution)
print(solution)
