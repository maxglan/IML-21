#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import linregress

# read the csv file
train = np.loadtxt(open('train.csv', "rb"), delimiter=",", skiprows=1)


# extracting the training data and the results
X = train[:, 2:]
y = train[:, 1]


#calculating our new basis
def features(x):
    f= [x,x**2, np.exp(x), np.cos(x)]
    f1= np.reshape(f, -1)
    f1= np.append(f1, 1)
    return f1

#calculating the values in our basis
phi=np.zeros((700, 21))

for i in range(len(y)):
    phi[i]= features(X[i])


model = LinearRegression(fit_intercept=False).fit(phi, y)

#Alternative way to solve the regression problem using scipy
solution = np.linalg.lstsq(phi,y)[0]
print(solution)


#fit_intercept=False

#saving the solution
#solution= model.coef_

# export to csv
np.savetxt("solution.csv", solution)
  

# for i,j in zip(l, range(length)):
# # performing k fold validation for each lambda
#     model = Ridge(i)

#     # T returns the array of scores of the estimator for each run of the cross validation
#     T = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=k)
#     # takes the mean of the RMSEs for each lambda and stores it in solutions
#     solution[j] = 0.1 * sum(-T)


