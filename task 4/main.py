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
from tensorflow.keras.applications import EfficientNetB0
# import efficientnet.keras as efn


#importing the txt with the image names
train_triplets = np.genfromtxt("train_triplets.txt", dtype='str')
test_triplets = np.genfromtxt("test_triplets.txt", dtype='str')

""" 1) functions for preprocessing images into format required by NN """

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
    return image

def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

""" 1.5) importing images into tf.Dataset"""

cwd = os.getcwd()

anchor_images = list(
    [str(cwd + "/food/" + f +".jpg") for f in train_triplets[:,0]])

positive_images = sorted(
    [str(cwd + "/food/" + f +".jpg") for f in train_triplets[:,1]])

negative_images = sorted(
    [str(cwd + "/food/" + f +".jpg") for f in train_triplets[:,2]])

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)

dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

#split into validation set
image_count = len(anchor_images)
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(100)
train_dataset = train_dataset.prefetch(8)

val_dataset = val_dataset.batch(100)
val_dataset = val_dataset.prefetch(8)

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

dataset_test = tf.data.Dataset.zip((anchor_test, positive_test, negative_test))
dataset_test = dataset_test.map(preprocess_triplets)

""" 2) Setting up the embedding generator model """
#size of output layer
emb_size = 12

base_model = EfficientNetB0(input_shape = target_shape + (3,), include_top = False, weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False

flatten = layers.Flatten()(base_model.output)
dense1 = layers.Dense(64, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
output = layers.Dense(emb_size)(dense1)

embedding = Model(base_model.input, output, name="Embedding")
embedding.summary()

"""creating the Siamese Network"""
input_anchor = tf.keras.layers.Input(target_shape + (3,))
input_positive = tf.keras.layers.Input(target_shape + (3,))
input_negative = tf.keras.layers.Input(target_shape + (3,))

embedding_anchor = embedding(input_anchor)
embedding_positive = embedding(input_positive)
embedding_negative = embedding(input_negative)

output_emb = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

siam_model = tf.keras.models.Model(inputs = [input_anchor, input_positive, input_negative], 
                                   outputs = output_emb)

siam_model.summary()

#defining the triplet loss
margin = 0.2

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = np.linalg.norm(anchor - positive)
    negative_dist = np.linalg.norm(anchor - negative)
    return tf.maximum(positive_dist - negative_dist + margin, 0.)

"""fitting model"""
siam_model.compile(loss = triplet_loss, optimizer ='adam')

siam_model.fit(train_dataset, epochs=1, validation_data=val_dataset)

# """ alternative fitting version"""
# anchor_dataset = anchor_dataset.map(preprocess_image)
# positive_dataset = positive_dataset.map(preprocess_image)
# negative_dataset = negative_dataset.map(preprocess_image)

# siam_model.compile(loss = triplet_loss, optimizer ='adam')
# siam_model.fit([anchor_dataset, positive_dataset, negative_dataset], epochs=1)

"""applying on test set"""
# predicted_vectors = siam_model.predict(dataset_test)

# #create ourput file
# def choose(prediction):
#     length = len(predicted_vectors[:,0])
#     result = np.zeros(length)
#     for i in range(length):
#         anchor, positive, negative = prediction[i,:emb_size], prediction[i,emb_size:2*emb_size], prediction[i,2*emb_size:]
#         positive_dist = np.linalg.norm(anchor - positive)
#         negative_dist = np.linalg.norm(anchor - negative)
#         if positive_dist > negative_dist:
#             result[i] = 1
#     return result

# result = choose(predicted_vectors)
# np.savetxt("stupid_result.txt", result)
    

