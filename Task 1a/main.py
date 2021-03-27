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

#randomnizing the order of the rows before splitting into k folds and fixing the seed
np.random.seed(42)
np.random.shuffle(train)


# extracting the training data and the results
X = train[:, 1:]
y = train[:, 0]


# initialize array in which the solutions are stored
solution = np.zeros(length)

for i,j in zip(l, range(length)):
# performing k fold validation for each lambda
    model = Ridge(i)

    # T returns the array of scores of the estimator for each run of the cross validation
    T = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=k)
    # takes the mean of the RMSEs for each lambda and stores it in solutions
    solution[j] = 0.1 * sum(-T)

# export to csv
np.savetxt("solution.csv", solution)
print(solution)
