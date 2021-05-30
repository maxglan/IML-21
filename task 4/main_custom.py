# -*- coding: utf-8 -*-
"""
Created on Sat May 29 17:21:12 2021

@author: roman

Custom network 
"""

import numpy as np
import pandas as pd
import sys

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications import EfficientNetB0

#import coremltools as ct

import pathlib

""" Parameters """

#required size for EfficientNetB0
target_shape = (224,224)

# max_image_count
max_train_count = 4000
max_test_count = 50

batch_size = 30
epochs = 10


""" Load data """

# Create paths
images_path = pathlib.Path().absolute() / "food"

#importing the txt with the image names
train_triplets = np.genfromtxt("train_triplets.txt", dtype='str')
test_triplets = np.genfromtxt("test_triplets.txt", dtype='str')



""" Change data format """

def modify_for_classification(triplets, do_shuffle=False):
    """ Make the dateset ready for binary classification """
    
    if do_shuffle: 
        np.random.shuffle(triplets)
    
    number = triplets.shape[0]
    
    y = np.random.randint(2, size=number) # array with 0 and 1
    
    for n in range(number):
        if y[n] == 0:
            B = triplets[n,1]
            C = triplets[n,2]
            triplets[n,1] = C
            triplets[n,2] = B 
        
    return triplets, y

train_triplets, train_y = modify_for_classification(triplets=train_triplets, 
                                                  do_shuffle=True)


def path_to_dataset(triplets, max_image_count: int=1000): 
    """

    Parameters
    ----------
    triplets :  2D array with strings. 
                First index: [0, inf)
                Second index: {0,1,2} for {A, B, C}
                
    max_image_count:  Maximum number of images that are processed into the set

    Returns
    -------
    tf.data.Dataset

    """
    
    # Reduce size of dataset
    triplets = triplets[:max_image_count,:]
    
    # Get paths
    A_images = [str(images_path) + "/" +  str(n) + ".jpg" for n in triplets[:,0]]
    B_images = [str(images_path) + "/" + str(n) + ".jpg" for n in triplets[:,1]]
    C_images = [str(images_path) + "/" +  str(n) + ".jpg" for n in triplets[:,2]]

    # Convert path to datasets
    A_dataset = tf.data.Dataset.from_tensor_slices(A_images)
    B_dataset = tf.data.Dataset.from_tensor_slices(B_images)
    C_dataset = tf.data.Dataset.from_tensor_slices(C_images)
    
    # Combine datasets
    dataset = tf.data.Dataset.zip((A_dataset, B_dataset, C_dataset))

    
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(preprocess_triplets)
    
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

def preprocess_triplets(A, B, C):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(A),
        preprocess_image(B),
        preprocess_image(C),
    )


# Create tf.data.Datasets
train_dataset = path_to_dataset(train_triplets, max_image_count=max_train_count)  
test_dataset = path_to_dataset(test_triplets, max_image_count=max_test_count) 

train_y = tf.data.Dataset.from_tensor_slices([[train_y[i]] for i in range(max_train_count)])

train_dataset = tf.data.Dataset.zip((train_dataset, train_y))

# Batch datasets
train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

print(train_dataset)

""" Model 0 
# encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")

input_A = layers.Input(name="A", shape=target_shape + (3,))
input_B = layers.Input(name="B", shape=target_shape + (3,))
input_C = layers.Input(name="C", shape=target_shape + (3,))

A = 

x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

decoder_input = keras.Input(shape=(16,), name="encoded_img")
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
autoencoder.summary()
 
"""


""" Model """

# Pre trained part


base_cnn_A = EfficientNetB0(weights="imagenet", input_shape=target_shape + (3,), include_top=False)
for layer in base_cnn_A.layers:
    layer.trainable = False
    layer._name = str(layer._name) + '_A'
base_cnn_A = Model(inputs=base_cnn_A.input, outputs=base_cnn_A.outputs, name="base_cnn_A")

base_cnn_B = EfficientNetB0(weights="imagenet", input_shape=target_shape + (3,), include_top=False)
for layer in base_cnn_B.layers:
    layer.trainable = False
    layer._name = str(layer._name) + '_B'
base_cnn_B = Model(inputs=base_cnn_B.input, outputs=base_cnn_B.outputs, name="base_cnn_B")
    
base_cnn_C = EfficientNetB0(weights="imagenet", input_shape=target_shape + (3,), include_top=False)
for layer in base_cnn_C.layers:
    layer.trainable = False
    layer._name = str(layer._name) + '_C'
base_cnn_C = Model(inputs=base_cnn_C.input, outputs=base_cnn_C.outputs, name="base_cnn_C")


class ConcatenationLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        return layers.Concatenate()([anchor, positive, negative])


A_input = layers.Input(name="A", shape=target_shape + (3,))
B_input = layers.Input(name="B", shape=target_shape + (3,))
C_input = layers.Input(name="C", shape=target_shape + (3,))

concat = ConcatenationLayer()(
    base_cnn_A(resnet.preprocess_input(A_input)),
    base_cnn_B(resnet.preprocess_input(B_input)),
    base_cnn_C(resnet.preprocess_input(C_input)),
)

model = Model(
    inputs=[A_input, B_input, C_input], outputs=concat
)



print(len(model.outputs))

flatten = layers.Flatten()(model.outputs[0])
dense1 = layers.Dense(50, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense3 = layers.Dense(9, activation="relu")(dense1)
dense3 = layers.BatchNormalization()(dense3)
classifer_layer = layers.Dense(1, activation="sigmoid")(dense3)

model = Model(inputs=[A_input, B_input, C_input], outputs=classifer_layer)
model.summary()


""" Train binary classifier """



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=train_dataset, 
          batch_size=batch_size, 
          epochs=epochs)


""" Predict """


""" Save results """


""" Finish """

print("Finish")





