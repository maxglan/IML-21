# -*- coding: utf-8 -*-
"""
Created on June 1

@author: roman

Dual network
"""

import numpy as np
import pandas as pd
import sys

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications import EfficientNetB0

import pathlib

""" Parameters """

#required size for EfficientNetB0
target_shape = (224,224)

# max_image_count
max_train_count = 1000
max_test_count = 1000

batch_size = 20
epochs = 20

# Optimizer 

# learning_rate = 1e-4, epsilon = 1e-7
adam = tf.keras.optimizers.Adam(learning_rate=0.1, epsilon=0.1) 

# learning_rate=0.01, momentum=0.0
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')

opt = adam


""" Load data """

# Create paths
images_path = pathlib.Path().absolute() / "food"

#importing the txt with the image names
train_triplets = np.genfromtxt("train_triplets.txt", dtype='str')
test_triplets = np.genfromtxt("test_triplets.txt", dtype='str')


""" Change data format """

def modify_for_training(triplets):
    """ Make the dateset ready for binary classification """
    
    number = triplets.shape[0]
    tupels = []
    y = []
    
    # Split triplets into tupels. Use symmetry between AB and BA. 
    for n in range(number):
            A = triplets[n,0]
            B = triplets[n,1]
            C = triplets[n,2]
            
            # Similar taste
            tupels.append([A,B])
            tupels.append([B,A])
            y.append([1])
            y.append([1])
            
            # Different taste
            tupels.append([A,C])
            tupels.append([C,A])
            y.append([0])
            y.append([0])
        
    # Convert to np.array
    tupels = np.array(tupels)
    y = np.array(y)
           
    # Shuffle input and output the same way
    shuffle_indices = np.random.permutation(number*4)
    tupels = tupels[shuffle_indices]
    y = y[shuffle_indices]
    
    return tupels, y


def modify_for_test(triplets):
    """ Make the dateset ready for binary classification """
    
    number = triplets.shape[0]
    tupels = []
    
    # Split triplets into tupels. Use symmetry between AB and BA. 
    for n in range(number):
            A = triplets[n,0]
            B = triplets[n,1]
            C = triplets[n,2]

            tupels.append([A,B])
            tupels.append([A,C])
            
    tupels = np.array(tupels)    
    
    return tupels

train_tupels, train_y = modify_for_training(train_triplets)
print(train_tupels[:10])
print(train_y[:10])

test_tupels = modify_for_test(test_triplets)
print(test_tupels[:10])


def path_to_dataset(tupels, max_image_count: int=1000): 
    """

    Parameters
    ----------
    triplets :  2D array with strings. 
                First index: [0, inf)
                Second index: {0,1} for {X, Y}
                
    max_image_count:  Maximum number of images that are processed into the set

    Returns
    -------
    tf.data.Dataset

    """
    
    # Reduce size of dataset
    tupels = tupels[:max_image_count,:]
    
    # Get paths
    X_images = [str(images_path) + "/" +  str(n) + ".jpg" for n in tupels[:,0]]
    Y_images = [str(images_path) + "/" + str(n) + ".jpg" for n in tupels[:,1]]

    # Convert path to datasets
    X_dataset = tf.data.Dataset.from_tensor_slices(X_images)
    Y_dataset = tf.data.Dataset.from_tensor_slices(Y_images)

    # Combine datasets
    dataset = tf.data.Dataset.zip((X_dataset, Y_dataset))

    # dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(preprocess_tupels)
    
    return dataset

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    try: 
      print("Read " + filename)
      image_string = tf.io.read_file(filename)
    except: 
      print("The image with string: " + str(filename) + "could not be loaded")
      image_string = tf.io.read_file(images_path + "00000.jpg")
    
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image

def preprocess_tupels(X, Y):
    """
    Given the filenames corresponding to the two images, load and
    preprocess them.
    """

    return (
        preprocess_image(X),
        preprocess_image(Y),
    )


# Create tf.data.Datasets
train_dataset = path_to_dataset(train_tupels, max_image_count=max_train_count)  
train_y = tf.data.Dataset.from_tensor_slices(train_y)
train_dataset = tf.data.Dataset.zip((train_dataset, train_y))

test_dataset = path_to_dataset(test_tupels, max_image_count=max_test_count) 


# Batch datasets
train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)



""" Model """

# Input

X_input = layers.Input(name="X", shape=target_shape + (3,))
Y_input = layers.Input(name="Y", shape=target_shape + (3,))


# Pre trained part

base_cnn_X = EfficientNetB0(weights="imagenet", input_shape=target_shape + (3,), include_top=False)
for layer in base_cnn_X.layers:
    layer.trainable = False
    layer._name = str(layer._name) + '_X'
base_cnn_X = Model(inputs=base_cnn_X.input, outputs=base_cnn_X.outputs, name="base_cnn_X")

base_cnn_Y = EfficientNetB0(weights="imagenet", input_shape=target_shape + (3,), include_top=False)
for layer in base_cnn_Y.layers:
    layer.trainable = False
    layer._name = str(layer._name) + '_Y'
base_cnn_Y = Model(inputs=base_cnn_Y.input, outputs=base_cnn_Y.outputs, name="base_cnn_Y")


class ConcatenationLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, X, Y):
        return layers.Concatenate()([X, Y])

leaky_relu = layers.LeakyReLU(alpha=0.1)
dim_red = layers.Dense(15, activation=leaky_relu, input_shape=[7, 7, 1280])

concat = ConcatenationLayer()(
    dim_red(layers.Flatten()(base_cnn_X(resnet.preprocess_input(X_input)))),
    dim_red(layers.Flatten()(base_cnn_Y(resnet.preprocess_input(Y_input)))),
)

model = Model(
    inputs=[X_input, Y_input], outputs=concat
)

print(model.outputs)
print(model.outputs[0])

flatten = layers.Flatten()(model.outputs[0]) 
dense = layers.Dense(15, activation=leaky_relu)(flatten)
#dense = layers.BatchNormalization()(dense)
classifer_layer = layers.Dense(1, activation="sigmoid")(dense)

model = Model(inputs=[X_input, Y_input], outputs=classifer_layer)
model.summary()

tf.keras.utils.plot_model(model)


""" Train binary classifier """

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x=train_dataset, 
          batch_size=batch_size, 
          epochs=epochs)

""" Predict """


""" Save results """


""" Finish """

print("Finish")


