#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 17:42:45 2021

@author: georgengin
"""
import matplotlib.pyplot as plt
import numpy as np
import os
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




train_triplets = np.loadtxt("train_triplets.txt")
test_triplets = np.loadtxt("test_triplets.txt")

#required size for EfficientNetB0
target_shape = (224,224)


"""implementation from https://keras.io/examples/vision/siamese_network/"""

def preprocess_image(filename):
    """
    Load the specified file as a JPG image, preprocess it and resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

# creating lists for the anchor, positive and negative images

anchor_images = list(
    [str("/food/" + str(int(f)) +".jpg") for f in train_triplets[:,0]])

positive_images = sorted(
    [str("/food/" + str(int(f)) +".jpg") for f in train_triplets[:,1]])

negative_images = sorted(
    [str("/food/" + str(int(f)) +".jpg") for f in train_triplets[:,2]])

image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)



# Let's now split our dataset in train and validation.
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(8)