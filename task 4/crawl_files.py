#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 09:31:02 2021

@author: georgengin
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications import EfficientNetB0

"""function that produces all image names ordered"""
def list_creator(tt):
    l=[str(0)] * 10000
    for i in tt:
        length = len(str(i))
        for j in range(4 -length):
            l[i] = str(l[i]  + "0")
        
        l[i] = l[i] + str(i)
    return l

tt = np.arange(0, 10000, dtype = int)
images = list_creator(tt)

"""1) importing images into format required by NN 
we first feature transform every single picture, and then later use this processed data to fill the anchor , positive and negative sets.
This saves ressources (10k images instead of 60k*3 * 2 train and test)"""

#required size for EfficientNetB0
target_shape = (224,224)

def preprocess_image(filename):
    """
    Load the specified file as a JPG image, preprocess it and resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image[None]

# creating lists for the anchor, positive and negative images
cwd = os.getcwd()

all_images = list(
    [str(cwd + "/food/" + f +".jpg") for f in images])


complete_dataset = tf.data.Dataset.from_tensor_slices(all_images)
complete_dataset = complete_dataset.map(preprocess_image)

base_model = EfficientNetB0(input_shape = target_shape + (3,), include_top = False, weights = 'imagenet')

def feature_trafo(anchor):
    features_anchor = base_model.predict(anchor)
    
    shape= np.shape(features_anchor)
    
    f_a = np.reshape(features_anchor, (shape[0], -1))
    return f_a

complete = feature_trafo(complete_dataset)
np.savetxt("complete_B0.csv", complete)