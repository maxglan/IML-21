"""
Title: Image similarity estimation using a Siamese Network with a triplet loss
Authors: [Hazem Essam](https://twitter.com/hazemessamm) and [Santiago L. Valdarrama](https://twitter.com/svpino)
Date created: 2021/03/25
Last modified: 2021/03/25
Description: Training a Siamese Network to compare the similarity of images using a triplet loss function.
"""

"""
## Introduction

A [Siamese Network](https://en.wikipedia.org/wiki/Siamese_neural_network) is a type of network architecture that
contains two or more identical subnetworks used to generate feature vectors for each input and compare them.

Siamese Networks can be applied to different use cases, like detecting duplicates, finding anomalies, and face recognition.

This example uses a Siamese Network with three identical subnetworks. We will provide three images to the model, where
two of them will be similar (_anchor_ and _positive_ samples), and the third will be unrelated (a _negative_ example.)
Our goal is for the model to learn to estimate the similarity between images.

For the network to learn, we use a triplet loss function. You can find an introduction to triplet loss in the
[FaceNet paper](https://arxiv.org/pdf/1503.03832.pdf) by Schroff et al,. 2015. In this example, we define the triplet
loss function as follows:

`L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)`

This example uses the [Totally Looks Like dataset](https://sites.google.com/view/totally-looks-like-dataset)
by [Rosenfeld et al., 2018](https://arxiv.org/pdf/1803.01485v3.pdf).
"""

"""
## Setup
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
from tensorflow.keras.applications import efficientnet

import pathlib

#required size for EfficientNetB0
target_shape = (224,224)

# max_image_count
max_train_count = 1000
max_val_count = 50
max_test_count = 50

batchsize = 10
epochs = 5

"""
## Load the dataset

We are going to load the *Totally Looks Like* dataset and unzip it inside the `~/.keras` directory
in the local environment.

The dataset consists of two separate files:

* `left.zip` contains the images that we will use as the anchor.
* `right.zip` contains the images that we will use as the positive sample (an image that looks like the anchor).
"""

# cache_dir = Path(Path.home()) / "Courses" / "keras-io" / "examples" / "vision" / "img" / "siamese_network" 
cache_dir = pathlib.Path().absolute()

images_path = cache_dir / "food"

#importing the txt with the image names
train_triplets = np.genfromtxt("train_triplets.txt", dtype='str')
test_triplets = np.genfromtxt("test_triplets.txt", dtype='str')


def path_to_dataset(triplets, start: int=0, max_image_count: int=1000): 
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
    triplets = triplets[start:(start+max_image_count),:]
    
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


"""
## Preparing the data

We are going to use a `tf.data` pipeline to load the data and generate the triplets that we
need to train the Siamese network.

We'll set up the pipeline using a zipped list with anchor, positive, and negative filenames as
the source. The pipeline will load and preprocess the corresponding images.
"""


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


"""
Let's setup our data pipeline using a zipped list with an anchor, positive,
and negative image filename as the source. The output of the pipeline
contains the same triplet with every image loaded and preprocessed.
"""

# Let's now split our dataset in train and validation.
train_dataset = path_to_dataset(train_triplets, 
                                start=0, 
                                max_image_count=max_train_count) 
val_dataset = path_to_dataset(train_triplets, 
                              start=max_train_count,
                              max_image_count=max_val_count) 
test_dataset = path_to_dataset(test_triplets, 
                              start=0,
                              max_image_count=max_test_count) 

train_dataset = train_dataset.batch(batchsize, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(batchsize, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(batchsize, drop_remainder=False)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

"""
Let's take a look at a few examples of triplets. Notice how the first two images
look alike while the third one is always different.
"""


def visualize(A, B, C):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], A[i])
        show(axs[i, 1], B[i])
        show(axs[i, 2], C[i])


#visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])

"""
## Setting up the embedding generator model

Our Siamese Network will generate embeddings for each of the images of the
triplet. To do this, we will use a ResNet50 model pretrained on ImageNet and
connect a few `Dense` layers to it so we can learn to separate these
embeddings.

We will freeze the weights of all the layers of the model up until the layer `conv5_block1_out`.
This is important to avoid affecting the weights that the model has already learned.
We are going to leave the bottom few layers trainable, so that we can fine-tune their weights
during training.
"""

""" Original version of CNN setup

base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable
    
    
"""


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


"""
## Setting up the Siamese Network model

The Siamese network will receive each of the triplet images as an input,
generate the embeddings, and output the distance between the anchor and the
positive embedding, as well as the distance between the anchor and the negative
embedding.

To compute the distance, we can use a custom layer `DistanceLayer` that
returns both values as a tuple.
"""


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


A_input = layers.Input(name="A", shape=target_shape + (3,))
B_input = layers.Input(name="B", shape=target_shape + (3,))
C_input = layers.Input(name="C", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(resnet.preprocess_input(A_input)),
    embedding(resnet.preprocess_input(B_input)),
    embedding(resnet.preprocess_input(C_input)),
)

siamese_network = Model(
    inputs=[A_input, B_input, C_input], outputs=distances
)

"""
## Putting everything together

We now need to implement a model with custom training loop so we can compute
the triplet loss using the three embeddings produced by the Siamese network.

Let's create a `Mean` metric instance to track the loss of the training process.
"""


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
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


print("Make predictions 1")

results1 = []
cosine_similarity = metrics.CosineSimilarity()

for i, test_dataset_item in enumerate(test_dataset): 
    
    A, B, C = test_dataset_item
    
    A_embedding = embedding(resnet.preprocess_input(A))
    B_embedding = embedding(resnet.preprocess_input(B))
    C_embedding =  embedding(resnet.preprocess_input(C))
    
    for j in range(A_embedding.shape[0]):
        
        AB_similarity = cosine_similarity(A_embedding[j], B_embedding[j])
        #print("Positive similarity:", AB_similarity.numpy())
        
        AC_similarity = cosine_similarity(A_embedding[j], C_embedding[j])
        #print("Negative similarity", AC_similarity.numpy())
        
        if AB_similarity.numpy() > AC_similarity.numpy(): 
            results1.append(1)
        else:
            results1.append(0)

np.savetxt("result1.txt", results1, fmt='%i') 


"""
## Training

We are now ready to train our model.
"""

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.fit(train_dataset, epochs=epochs, validation_data=val_dataset) # epochs=10

"""
## Inspecting what the network has learned

At this point, we can check how the network learned to separate the embeddings
depending on whether they belong to similar images.

We can use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to measure the
similarity between embeddings.

Let's pick a sample from the dataset to check the similarity between the
embeddings generated for each image.
"""

print("Make predictions 2")

results2 = []
cosine_similarity = metrics.CosineSimilarity()

for i, test_dataset_item in enumerate(test_dataset): 
    
    A, B, C = test_dataset_item
    
    A_embedding = embedding(resnet.preprocess_input(A))
    B_embedding = embedding(resnet.preprocess_input(B))
    C_embedding =  embedding(resnet.preprocess_input(C))
    
    for j in range(A_embedding.shape[0]):
        
        AB_similarity = cosine_similarity(A_embedding[j], B_embedding[j])
        #print("Positive similarity:", AB_similarity.numpy())
        
        AC_similarity = cosine_similarity(A_embedding[j], C_embedding[j])
        #print("Negative similarity", AC_similarity.numpy())
        
        if AB_similarity.numpy() > AC_similarity.numpy(): 
            results2.append(1)
        else:
            results2.append(0)

np.savetxt("result2.txt", results2, fmt='%i') 


"""
## Summary

1. The `tf.data` API enables you to build efficient input pipelines for your model. It is
particularly useful if you have a large dataset. You can learn more about `tf.data`
pipelines in [tf.data: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data).

2. In this example, we use a pre-trained ResNet50 as part of the subnetwork that generates
the feature embeddings. By using [transfer learning](https://www.tensorflow.org/guide/keras/transfer_learning?hl=en),
we can significantly reduce the training time and size of the dataset.

3. Notice how we are [fine-tuning](https://www.tensorflow.org/guide/keras/transfer_learning?hl=en#fine-tuning)
the weights of the final layers of the ResNet50 network but keeping the rest of the layers untouched.
Using the name assigned to each layer, we can freeze the weights to a certain point and keep the last few layers open.

4. We can create custom layers by creating a class that inherits from `tf.keras.layers.Layer`,
as we did in the `DistanceLayer` class.

5. We used a cosine similarity metric to measure how to 2 output embeddings are similar to each other.

6. You can implement a custom training loop by overriding the `train_step()` method. `train_step()` uses
[`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape),
which records every operation that you perform inside it. In this example, we use it to access the
gradients passed to the optimizer to update the model weights at every step. For more details, check out the
[Intro to Keras for researchers](https://keras.io/getting_started/intro_to_keras_for_researchers/)
and [Writing a training loop from scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch?hl=en).

"""