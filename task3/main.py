#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Imports """

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MaxAbsScaler
from numba import njit

""" Read the csv file """

print(" Read CSV files")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample.csv")

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
    
