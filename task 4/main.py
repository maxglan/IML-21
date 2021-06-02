#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:32:08 2021

@author: georgengin

"""
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from sklearn.neural_network import MLPClassifier

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
This saves ressources (10k images instead of 60k*3 *2)"""

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

# creating lists for the images
cwd = "/content/drive/My Drive/task 4"

all_images = list(
    [str(cwd + "/food/" + f +".jpg") for f in images])


complete_dataset = tf.data.Dataset.from_tensor_slices(all_images)
complete_dataset = complete_dataset.map(preprocess_image)

base_model = EfficientNetB0(input_shape = target_shape + (3,), include_top = False, weights = 'imagenet')

base_model.trainable = False
pool = layers.MaxPool2D(pool_size=(7,7))
output = pool(base_model.output)
#pool_layer = pool(base_model.output)
#output = layers.Flatten()(pool_layer)
embedding = Model(base_model.input, output, name="Embedding")

#calculating the embedding (preprocessing)
def feature_trafo(anchor):
    features_anchor = embedding.predict(anchor)
    
    shape= np.shape(features_anchor)
    f_a = np.reshape(features_anchor, (shape[0], -1))
    return f_a

complete = feature_trafo(complete_dataset)
print(np.shape(complete))
np.savetxt("/content/drive/My Drive/task 4/complete_B0.csv", complete)
train_triplets = np.loadtxt("/content/drive/My Drive/task 4/train_triplets.txt", dtype = int)
test_triplets = np.loadtxt("/content/drive/My Drive/task 4/test_triplets.txt", dtype = int)
complete = np.loadtxt("/content/drive/My Drive/task 4/complete_B0.csv")

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

def alternative_shuffle_class(triplets):
    old_triplets = np.copy(triplets)
    new_triplets = np.copy(triplets)

    length = len(triplets[:,0])
    binary = np.ones(2*length)

    #change pos and negative
    for i in range(length):

        new_triplets[i, 1] = triplets[i, 2]
        new_triplets[i, 2] = triplets[i, 1]
        binary[length + i] = 0

    combined = np.concatenate((old_triplets, new_triplets))
    print(np.shape(combined))
    np.random.seed(42)
    np.random.shuffle(combined)
    np.random.shuffle(binary)
    
    return combined, binary

big_train, big_binary = alternative_shuffle_class(train_triplets)

def preprocess(triplets, binary):
  #concatenate the given preprocessed vectors of the triplet
  length = len(triplets[:,0])
  # 3*1280 = length of Pre-trained output *3
  x = np.zeros((length, 3*1280))
  for i in range(length):
    x[i] = np.concatenate((complete[triplets[i,0]], complete[triplets[i,1]], complete[triplets[i,2]]))
  return x

data = preprocess(new_train, train_binary)
test_data = preprocess(test_triplets, train_binary)
big_data = preprocess(big_train, big_binary)

"""train NN"""
# clf = MLPClassifier(hidden_layer_sizes = (3*1280, 420, 42), verbose=True)
# clf.fit(data, train_binary)


"""train NN in manual batches"""
clf = MLPClassifier(hidden_layer_sizes = (3*1280, 420, 42), verbose=True, max_iter = 1)
for i in range(1):
  clf.partial_fit(big_data[:30000,:], big_binary[:30000], classes=[0, 1])
  clf.partial_fit(big_data[30000:60000,:], big_binary[30000:60000])
  clf.partial_fit(big_data[60000:90000,:], big_binary[60000:90000])
  clf.partial_fit(big_data[90000:,:], big_binary[90000:])

solution = clf.predict(test_data)
np.savetxt("/content/drive/My Drive/task 4/prediction.txt", solution, fmt='%u')