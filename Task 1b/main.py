#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression

# read the csv file
train = np.loadtxt(open('train.csv', "rb"), delimiter=",", skiprows=1)


# extracting the training data and the results
X = train[:, 2:]
y = train[:, 1]


#function for calculating our new basis
def features(x):
    f= [x,x**2, np.exp(x), np.cos(x)]
    f1= np.reshape(f, -1)
    f1= np.append(f1, 1)
    return f1

#creating array for our basis& filling it
phi=np.zeros((700, 21))

for i in range(len(y)):
    phi[i]= features(X[i])


#fit_intercept=False as we have implemented the constant offset of our linear function as the 21st entry of phi
model = LinearRegression(fit_intercept=False).fit(phi, y)

#saving the solution
solution= model.coef_

# export to csv
np.savetxt("solution.csv", solution)
