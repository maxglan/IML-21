# -*- coding: utf-8 -*-
"""
Created on Sat May 29 17:21:12 2021

@author: roman

Custom network 
"""

import numpy as np
import pandas as pd

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
max_train_count = 50
max_test_count = 50

batch_size = 5
epochs = 2


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
    print(number)
    
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
    # dataset = dataset.shuffle(buffer_size=1024)
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

# Batch datasets
train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


""" Model """

# Pre trained part

model = models.Sequential()

base_cnn = EfficientNetB0(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

for layer in base_cnn.layers:
    layer.trainable = False
    
model.add(base_cnn)

model.add(layers.Flatten())
model.add(layers.Dense(50, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dense(9, activation="relu"))
model.add(layers.BatchNormalization())

# flatten = layers.Flatten()(base_cnn.output)
# dense1 = layers.Dense(50, activation="relu")(flatten)
# dense1 = layers.BatchNormalization()(dense1)
# dense2 = layers.Dense(9, activation="relu")(dense1)
# output = layers.BatchNormalization()(dense2)

embedding = Model(inputs=model.input, outputs=model.output, name="Embedding")

input_A = layers.Input(name="A", shape=target_shape + (3,))
input_B = layers.Input(name="B", shape=target_shape + (3,))
input_C = layers.Input(name="C", shape=target_shape + (3,))

# input_A = embedding(resnet.preprocess_input(input_A))
# input_B = embedding(resnet.preprocess_input(input_B))
# input_C = embedding(resnet.preprocess_input(input_C))

# Combined input
combined = layers.concatenate([input_A, input_B, input_C])              
combined = layers.Dense(9, activation="relu")(combined)

combined = layers.BatchNormalization()(combined)
combined = layers.Dense(1, activation="sigmoid")(combined)

# our model will accept the inputs of the two branches and
# then output a single value
#model = Model(inputs=[x.input, y.input], outputs=z)

model = Model(inputs=[input_A, input_B, input_C], 
              outputs=combined)


""" Train binary classifier """

model.compile()

model.fit(x=train_dataset, 
          y=train_y[:max_train_count], 
          batch_size=batch_size, 
          epochs=epochs, 
   )

""" Predict """


""" Save results """





