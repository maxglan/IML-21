#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 18:41:49 2021

@author: georgengin
"""
import numpy as np
from numpy.linalg import norm

test = np.loadtxt("test_features.csv")

def cos_sim(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))

def similarity(A, B, C):
    x = len(A[:,0])
    result = np.zeros(x)
    for i in range(x):
        ab = cos_sim(A[i,:], B[i,:])
        ac = cos_sim(A[i,:], C[i,:])
        if ab > ac:
            result[i]=1
    return result

length = len(test[0,:])
t = int(length/ 3)

a = test[:, :t]
b = test[:, t: 2*t]
c = test[:, 2*t:]

result = similarity(a, b, c)

np.savetxt("stupid_result.txt", result)