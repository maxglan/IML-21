#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:32:18 2021

@author: georgengin
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from numba import njit


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