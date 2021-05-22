#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:19:47 2021

@author: georgengin
"""
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds
import tensorflow as tf 
from tensorflow.keras.applications import EfiicientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt

train_triplets = np.loadtxt("train_triplets.txt")
test_triplets = np.loadtxt("test_triplets.txt")



"""the second row of our image list (B) is the binary 1 row (as it is closer to A), so we can switch B and C 
in some instances to also have some 0's as classification and not always 1"""

def shuffle_class(triplets):
    new_triplets = np.copy(triplets)
    length = len(triplets[:,0])
    binary = np.ones(length)

    for i in range(int((length-1) / 2)):

        new_triplets[i*2, 1] = triplets[i*2, 2]
        new_triplets[i*2, 2] = triplets[i*2, 1]
        
        binary[i*2] = 0
        
    return new_triplets, binary
    
new_train, train_binary = shuffle_class(train_triplets)
new_test, test_binary = shuffle_class(train_triplets)


""" To do:
    1) import images into format required by NN (arrays or whatever, size reformatting to (224,224,3))
    2) augment images (eg Rotations) for better accuracy
    3) apply NN
"""

"""1"""
#imput shape for B0, varies for other B's
image_size= (224,224)
input_shape = (image_size, 3)

batch_size = 100

"""2"""
def triplet_loss(a, b, c):
    margin=0
    return max(0, np.linalg.norm(a, b) - np.linalg.norm(a, c) + margin)
     




# ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
# ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))



model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape, classes= 10, drop_connect_rate = 0.4)
