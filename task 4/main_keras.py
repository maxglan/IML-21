#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 17:42:45 2021

@author: georgengin
"""
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import efficientnet
from numpy.linalg import norm




#importing the txt with the image names
train_triplets = np.genfromtxt("train_triplets.txt", dtype='str')
test_triplets = np.genfromtxt("test_triplets.txt", dtype='str')

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

# Let's now split our dataset in train and validation
dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

train_dataset = dataset.take(round(image_count * 0.1))
val_dataset = dataset.skip(round(image_count * 0.1))

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


test_dummy_np = np.zeros(len(anchor_test))
test_dummy = tf.data.Dataset.from_tensor_slices(test_dummy_np)

anchor_test = tf.data.Dataset.from_tensor_slices(anchor_test)
positive_test = tf.data.Dataset.from_tensor_slices(positive_test)
negative_test = tf.data.Dataset.from_tensor_slices(negative_test)

test_dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
test_dataset = test_dataset.map(preprocess_triplets)

test_dataset = tf.data.Dataset.zip((test_dataset, test_dummy))



"""3) Setting up the embedding generator model """
base_cnn = EfficientNetB0(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

for layer in base_cnn.layers:
    layer.trainable = False

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(69, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
output = layers.Dense(13)(dense1)

embedding = Model(base_cnn.input, output, name="Embedding")

# trainable = False
# for layer in base_cnn.layers:
#     if layer.name == "conv5_block1_out":
#         trainable = True
#     layer.trainable = trainable 
    
class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(efficientnet.preprocess_input(anchor_input)),
    embedding(efficientnet.preprocess_input(positive_input)),
    embedding(efficientnet.preprocess_input(negative_input)),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.2)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
 
"""fitting our model"""

# cb = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', min_delta=0, patience=0, verbose=0,
#     mode='auto', baseline=None, restore_best_weights=False
# )

# cb_list = [cb, ...]¨



siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))

# print("data", train_dataset)
# print("model", siamese_model.input)

siamese_model.fit(train_dataset, epochs=1, validation_data=val_dataset)

"""testing our model, alternative"""


pre = siamese_model.predict(test_dataset)
#print(np.shape(pre))

"""testing our model"""
# anchor_t = layers.Input(name="at", shape=target_shape + (3,))
# positive_t = layers.Input(name="pt", shape=target_shape + (3,))
# negative_t = layers.Input(name="nt", shape=target_shape + (3,))

length = len(anchor_test)
t = int(length/ 3)

#dataset_test = tf.data.Dataset.zip((anchor_test, positive_test, negative_test))
#at, pt, nt = dataset_test
anchor_embedding = embedding.predict(anchor_test)
positive_embedding = embedding(positive_test)
negative_embedding = embedding(negative_test)

def cos_sim(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))

def similarity(A, B, C):
    x = len(A[:,0])
    result = np.zeros(x)
    for i in range(x):
        ab = cos_sim(A[i,:], B[i,:])
        ac = cos_sim(A[i,:], C[i,:])
        if ab > ac:
            result[i]=1
    return result



result = similarity(anchor_embedding, positive_embedding, negative_embedding)

np.savetxt("stupid_result.txt", result)