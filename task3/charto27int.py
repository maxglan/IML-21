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
    # 4 acids * 26 letters
    Sequence = np.zeros((length, 26*4))
    
    for i in range(length):
        l = list(df.iloc[i,0])
        
        # -65 since A in Ascii starts at position 65
        number = [ord(k)-65 for k in l]
        
        for j, m in zip(number, range(len(number))):
            Sequence[i, j +26*m] += 1
        
    return Sequence