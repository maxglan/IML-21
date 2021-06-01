#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 17:42:45 2021

@author: georgengin
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
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
from tensorflow.python.keras.models import Sequential
#from tensorflow.python.experimental.numpy import heaviside

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
    # return image[None]

def preprocess_image_N(filename):
    """
    Load the specified file as a JPG image, preprocess it and resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image[None]
    
def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

def preprocess_triplets_N(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and preprocess them.
    """

    return (
        preprocess_image_N(anchor),
        preprocess_image_N(positive),
        preprocess_image_N(negative),
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
#image_count = len(anchor_images)
image_count = 4000

image_round = (round(image_count * 0.8), image_count - round(image_count * 0.8)) #= (47612, 11903)


train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(100)
# train_dataset = train_dataset.prefetch(8)

val_dataset = val_dataset.batch(100)
# val_dataset = val_dataset.prefetch(8)

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
dataset_test = dataset_test.map(preprocess_triplets_N)

#dataset_test is <MapDataset shapes: ((224, 224, 3), (224, 224, 3), (224, 224, 3)), types: (tf.float32, tf.float32, tf.float32)>

""" 2) Setting up the embedding generator model """
#size of output layer
emb_size = 8

base_model = EfficientNetB0(input_shape = target_shape + (3,), include_top = False, weights = 'imagenet')
base_model.trainable = False

pool = layers.MaxPool2D(pool_size=(7,7))
pool_layer = pool(base_model.output)
flatten = layers.Flatten()(pool_layer)
dense1 = layers.Dense(32, activation="relu")(flatten)
output = layers.Dense(emb_size)(dense1)

embedding = Model(base_model.input, output, name="Embedding")
# embedding.summary()

"""3) creating the Siamese Network"""
input_anchor = layers.Input(target_shape + (3,))
input_positive = layers.Input(target_shape + (3,))
input_negative = layers.Input(target_shape + (3,))

embedding_anchor = embedding(input_anchor)
embedding_positive = embedding(input_positive)
embedding_negative = embedding(input_negative)

output_emb = layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

siam_model = tf.keras.models.Model(inputs = [input_anchor, input_positive, input_negative], 
                                   outputs = output_emb)

"""
siam_model asks for input: (<KerasTensor: shape=(None, 224, 224, 3) dtype=float32 (created by layer 'input_2')>, 
          <KerasTensor: shape=(None, 224, 224, 3) dtype=float32 (created by layer 'input_3')>, 
          <KerasTensor: shape=(None, 224, 224, 3) dtype=float32 (created by layer 'input_4')>)

siam_model has output: # KerasTensor(type_spec=TensorSpec(shape=(None, 24), dtype=tf.float32, name=None)
"""
# siam_model.summary()

#defining the triplet loss
margin = 0.2

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + margin, 0.)

cosine_similarity = metrics.CosineSimilarity()


def pos_sim(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = cosine_similarity(anchor,positive)
    #negative_dist = cosine_similarity(anchor,negative)
    return positive_dist

def neg_sim(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    #positive_dist = cosine_similarity(anchor,positive)
    negative_dist = cosine_similarity(anchor,negative)
    return negative_dist

"""3.5) create dummy y values, see model.fit documentation"""
train_dummy_np = np.zeros((image_round[0], emb_size * 3))
val_dummy_np = np.zeros((image_round[1], emb_size * 3))

train_dummy = tf.data.Dataset.from_tensor_slices(train_dummy_np)
train_dummy = train_dummy.batch(100)
val_dummy = tf.data.Dataset.from_tensor_slices(train_dummy_np)
val_dummy = val_dummy.batch(100)

adam = tf.keras.optimizers.Adam(learning_rate=0.02, epsilon=0.1) 

sgd = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.2, nesterov=False, name='SGD')

opt = sgd


"""4) fitting model"""
siam_model.compile(loss = triplet_loss, optimizer = opt, metrics = [pos_sim, neg_sim])

input_fit = tf.data.Dataset.zip((train_dataset, train_dummy))
val_fit = tf.data.Dataset.zip((val_dataset, val_dummy))

siam_model.fit(input_fit, epochs=10, validation_data = val_fit, verbose=1)
# siam_model.fit(train_dataset, epochs=1, validation_data = val_dataset, verbose=1)


"""5) applying on test set"""
test_dummy_np = np.zeros((image_count, emb_size * 3))
test_dummy = tf.data.Dataset.from_tensor_slices(test_dummy_np)
test_fit = tf.data.Dataset.zip((dataset_test, test_dummy))

predicted_vectors = siam_model.predict(test_fit)

#create ourput file
def choose(prediction):
    length = len(predicted_vectors[0,:])
    result = np.zeros(length, dtype = int)
    for i in range(length):
        anchor, positive, negative = prediction[i,:emb_size], prediction[i,emb_size:2*emb_size], prediction[i,2*emb_size:]
        positive_dist = np.linalg.norm(anchor - positive)
        negative_dist = np.linalg.norm(anchor - negative)
        if positive_dist > negative_dist:
            result[i] = 1
    return result

result = choose(predicted_vectors)
np.savetxt("stupid_result.txt", result, fmt='%1.0i')