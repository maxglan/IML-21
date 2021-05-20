#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:19:47 2021

@author: georgengin
"""
import numpy as np

# import os 
# import tensorflow as tf 
# from tensorflow.keras.preprocessing.image import ImageDataGenerator 
# from tensorflow.keras import layers 
# from tensorflow.keras import Model
# import matplotlib.pyplot as plt

# import efficientnet.keras as efn
# base_model = efn.EfficientNetB0(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')

triplets = np.loadtxt("train_triplets.txt")


"""the second row of our image list (B) is the binary 1 row, so we can switch B and C 
in some instances to also have some 0's as classification and not always 1"""

def training_arr(train_triplets):
    new_triplets = np.copy(train_triplets)
    length = len(train_triplets[:,0])
    train_binary = np.ones(length)

    for i in range(int((length-1) / 2)):

        new_triplets[i*2, 1] = train_triplets[i*2, 2]
        new_triplets[i*2, 2] = train_triplets[i*2, 1]
        
        train_binary[i*2] = 0
        
    return new_triplets, train_binary
    
new_triplets, train_binary = training_arr(triplets)