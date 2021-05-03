#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Imports """

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from numba import njit

""" Read the csv file """

print("Read CSV files")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = np.loadtxt("sample.csv")

#this gives me just the activity label of the training set
active = train.iloc[:,1]

"""create a dataframe with 4 acids * 26 letters = 104 entries in each row, 
where we convert the char of an acid into an int via the function ord"""

def chartoint(df):
    length = len(df.index)
    # 4 acids * 26 letters
    Sequence = np.zeros((length, 26*4))
    
    for i in range(length):
        l = list(df.iloc[i,0])
        
        # -65 since A in Ascii starts at position 65
        number = [ord(k)-65 for k in l]
        
        for j, m in zip(number, range(len(number))):
            Sequence[i, j +26*m] += 1
        
    return Sequence
    
# creating the sequence dataframes
train_sequence = chartoint(train)
test_sequence = chartoint(test)

"""training our Neural Network"""

print("train NN")
clf = MLPClassifier(hidden_layer_sizes = (26*4, 69, 13), verbose=True)
clf.fit(train_sequence, active)
solution = clf.predict(test_sequence)

"""alternative approach using XGB"""

print("Booooooooooooooooooooooooooost")
model = xgb.XGBClassifier()
model.fit(train_sequence, active)
solution_xgb = model.predict(test_sequence)


""" store our solution in the sample file"""
np.savetxt("prediction.csv", solution, fmt='%u')
np.savetxt("prediction_xgb.csv", solution_xgb, fmt='%u')