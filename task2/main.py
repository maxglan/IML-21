#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: georgengin
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# read the csv file into a Panda df
#train = np.loadtxt(open('train.csv', "rb"), delimiter=",", skiprows=1)
trainf = pd.read_csv("train_features.csv")
trainl = pd.read_csv("train_labels.csv")

# print(trainf.pid[12]) prints the 12+1th item of the pid column

#creates array with all patient ids in the given order
id = trainf.pid.unique()

print(trainf.pid[12])