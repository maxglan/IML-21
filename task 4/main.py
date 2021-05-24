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
# import efficientnet.keras as efn
from tensorflow.keras.applications import resnet



#importing the txt with the image names
train_triplets = np.genfromtxt("train_triplets.txt", dtype='str')
test_triplets = np.genfromtxt("test_triplets.txt", dtype='str')


"""implementation adapted from https://keras.io/examples/vision/siamese_network/"""

"""1) importing images into format required by NN """

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

anchor_images = list(
    [str(cwd + "/food/" + f +".jpg") for f in train_triplets[:,0]])

positive_images = sorted(
    [str(cwd + "/food/" + f +".jpg") for f in train_triplets[:,1]])

negative_images = sorted(
    [str(cwd + "/food/" + f +".jpg") for f in train_triplets[:,2]])

image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)

anchor_dataset = anchor_dataset.map(preprocess_image)
positive_dataset = positive_dataset.map(preprocess_image)
negative_dataset = negative_dataset.map(preprocess_image)

# ###testing stuff
# two_images= anchor_images[0:2]
# image_set= tf.data.Dataset.from_tensor_slices(two_images)
# image_set= image_set.map(preprocess_image)

#repeat for test data
anchor_test = list(
    [str(cwd + "/food/" + f +".jpg") for f in test_triplets[:,0]])

positive_test = sorted(
    [str(cwd + "/food/" + f +".jpg") for f in test_triplets[:,1]])

negative_test = sorted(
    [str(cwd + "/food/" + f +".jpg") for f in test_triplets[:,2]])

anchor_test = tf.data.Dataset.from_tensor_slices(anchor_test)
positive_test = tf.data.Dataset.from_tensor_slices(positive_test)
negative_test = tf.data.Dataset.from_tensor_slices(negative_test)

anchor_test = anchor_test.map(preprocess_image)
positive_test = positive_test.map(preprocess_image)
negative_test = negative_test.map(preprocess_image)

"""Setting up the embedding generator model """
base_model = resnet.ResNet50(input_shape = target_shape + (3,), include_top = False, weights = 'imagenet')


def feature_trafo(anchor, positive, negative):
    """takes all three lists of images and returns the output in the form (n_images, anchor_output + positive_output + negative_output)
    here each x_output is a reshaped array of the oroginal multi dimensional output array of the used NN """
    """Pretraining anchor"""
    features_anchor = base_model.predict(anchor)
    """Pretraining positive"""
    features_positive = base_model.predict(positive)
    """Pretraining negative"""
    features_negative = base_model.predict(negative)
    
    shape= np.shape(features_anchor)
    
    f_a = np.reshape(features_anchor, (shape[0], -1))
    f_p = np.reshape(features_positive, (shape[0], -1))
    f_n = np.reshape(features_negative, (shape[0], -1))
    
    return np.concatenate((f_a, f_p, f_n), axis=1)

# cat = base_model.predict(image_set)
# cats = feature_trafo(image_set, image_set,image_set)

"""calculating the feature transformation using the pretrained NN and saving them for future use"""
training_data = feature_trafo(anchor_dataset, positive_dataset, negative_dataset)
test_data = feature_trafo(anchor_test, positive_test, negative_test)

np.savetxt("train_features.csv", training_data)
np.savetxt("test_features.csv", test_data)


"""defining our new NN"""
length = len(training_data[0,:])
t = int(length/ 3)

def triplet_loss(true, A):
    margin=0.4
    a = A[:t]
    b = A[t: 2*t]
    c = A[2*t:]
    return max(0, np.linalg.norm(a, b) - np.linalg.norm(a, c) + margin)

def NN(input_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(420,input_dim=input_size, activation="relu"))
    model.add(tf.keras.layers.Dense(13, activation="relu"))
    model.compile()
    return model

single_input = tf.keras.Input(np.shape(training_data[0,:]))

single_model = NN(training_data[0,:])

concat = tf.keras.concatenate([single_model, single_model, single_model])

model = tf.keras.Model([single_input, single_input, single_input], concat)
model.compile( optimizer = 'adam', loss = triplet_loss)









"""1) calculate features using imagenet
    2) concatenate them into a single vector
    3) train a NN using that vector with the the loss specified
    4) """