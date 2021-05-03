#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Imports """

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from numba import njit

""" Read the csv file """

print("Read CSV files")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = np.loadtxt("sample.csv")

#this gives me just the activity label of the training set
active = train.iloc[:,1]

"""create a dataframe with 4 entries in each row (one for each Amino Acid), 
where we convert the char of an acid into an int via the function ord"""

def chartoint(df):
    length = len(df.index)
    Sequence = pd.DataFrame(columns=['1', '2', '3', '4'], index=range(length))
    
    for i in range(length):
        l = list(df.iloc[i,0])
        number = [ord(k) for k in l]
        Sequence.iloc[i,:] = number
        
    return Sequence
    
# creating the sequence dataframes
train_sequence = chartoint(train)
test_sequence = chartoint(test)

"""training our Neural Network"""
print("train NN")
clf = MLPClassifier(hidden_layer_sizes = (1690, 690, 69), max_iter= 6900, verbose=True)
clf.fit(train_sequence, active)

solution = clf.predict(test_sequence)


print("The solution data has", np.sum(solution) , "active acids, should be on order of", int(np.sum(active)/(112/48)) )


""" store our solution in the sample file"""
np.savetxt("prediction.csv", solution, fmt='%u')