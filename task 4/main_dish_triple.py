# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:11:16 2021

@author: roman

Main dish triple
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import pathlib
import pickle

model = ResNet50(weights='imagenet')


""" Parameters """

# Dataset
number_of_images = 10000
get_only = 60000

# Preprocessing
do_load = True
do_extract = False
do_print = False

# Training
batch_size = 600
epochs = 30

# Fraction of data that is used for validation
train_frac = 0.9

# Optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=0.0001) # learning_rate = 1e-4, epsilon = 1e-7

sgd = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.2, nesterov=False, name='SGD')

opt = adam


""" Load data """

# Create paths
images_path = pathlib.Path().absolute() / "food"

#importing the txt with the image names
train_triplets = np.genfromtxt("train_triplets.txt", dtype='str')[:get_only]
test_triplets = np.genfromtxt("test_triplets.txt", dtype='str')[:get_only]


""" Extract features """



def get_features(do_load: bool=True, do_extract: bool=False, number_of_images: int=10): 
    
    prediction_list = []
    feature_set = set()
    
    if do_load: 
        try: 
            with open('feature_list.pkl', 'rb') as f:
                feature_list = pickle.load(f)
            with open('prediction_list.pkl', 'rb') as f:
                prediction_list = pickle.load(f)
                print("Loaded " + str(len(prediction_list)) + " images.")
            
        except:
            print("There are no old predictions that can be used.")
        
    if do_extract: 
        
        for i_img in range(len(prediction_list), number_of_images):
            
            print("Image number: " + str(i_img))
        
            img_path = images_path / (str(i_img).zfill(5) + ".jpg")
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
        
            preds = model.predict(x)
            
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            print('Predicted:', decode_predictions(preds, top=10)[0])
            prediction_list.append(decode_predictions(preds, top=3)[0])
            
            for prediction_item in decode_predictions(preds, top=3)[0]:
                nr, name, probability = prediction_item 
                feature_set.add(name)
                
        feature_list = list(feature_set)
        print(feature_list) 
    
        with open('feature_list.pkl', 'wb') as f:
            pickle.dump(feature_list, f)
        with open('prediction_list.pkl', 'wb') as f:
            pickle.dump(prediction_list, f)
        
            
    else:
        with open('feature_list.pkl', 'rb') as f:
            feature_list = pickle.load(f)
        with open('prediction_list.pkl', 'rb') as f:
            prediction_list = pickle.load(f)
            
    return feature_list, prediction_list

feature_list, prediction_list = get_features(do_load, do_extract, number_of_images)
    
if do_print: 
    print(20 * "*" + " feature_list " + 20 * "*")
    print(feature_list)
    
    print(20 * "*" + " prediction_list " + 20 * "*") 
    print(prediction_list)




""" Feature <-> index mapping """

# We enumerate the features to switch easily between feature and index

feature_space_dimension = len(feature_list)

feature_to_index = {}
for i_f, feature in enumerate(feature_list): 
    feature_to_index[feature] = i_f

index_to_feature = {value:key for key, value in feature_to_index.items()}

if do_print: 
    print(20 * "*" + " feature_to_index " + 20 * "*")
    print(feature_to_index)
    print(20 * "*" + " index_to_feature " + 20 * "*")
    print(index_to_feature)



""" Image (i_img) <-> prediction <-> vector mapping via indexing"""

def prediction_to_vector(prediction): 
    vector = np.zeros(feature_space_dimension)
    
    for prediction_item in prediction:
        
        nr, name, probability = prediction_item
        index = feature_to_index[name]
        vector[index] = probability
        
    return vector

vector_list = []

for prediction in prediction_list:
    
    vector = prediction_to_vector(prediction)
    vector_list.append(vector)
    
# print(vector_list) 
    

""" Generate dataset similar to the result of path_to_dataset() """

def triple_to_dataset(triplets):
    """ Make the dateset ready for binary classification """
    
    number = triplets.shape[0]
    XYZ = []
    y = []
    
    # Split triplets into tupels. Use symmetry between AB and BA. 
    for n in range(number):
            i_img_A = int(triplets[n,0])
            i_img_B = int(triplets[n,1])
            i_img_C = int(triplets[n,2])
            
            # Conversion to vector
            A = vector_list[i_img_A]
            B = vector_list[i_img_B]
            C = vector_list[i_img_C]
            
            # Similar taste
            XYZ.append([A,B,C])
            y.append([1])
            
            # Different taste
            XYZ.append([A,C,B])
            y.append([0])
        
    # Convert to np.array
    XYZ = np.array(XYZ)
    y = np.array(y)
           
    # Shuffle input and output the same way
    shuffle_indices = np.random.permutation(number*2)
    XYZ = XYZ[shuffle_indices]
    y = y[shuffle_indices]
    
    X_dataset = tf.constant(XYZ[:,0])
    Y_dataset = tf.constant(XYZ[:,1])
    Z_dataset = tf.constant(XYZ[:,2])
    y_dataset = tf.constant(y)

    # Combine datasets
    #dataset = tf.data.Dataset.zip((X_dataset, Y_dataset, y_dataset))
    
    return X_dataset, Y_dataset, Z_dataset, y_dataset #dataset

def triple_to_dataset_test(triplets):
    """ Make the dateset ready for binary classification """
    
    number = triplets.shape[0]
    XYZ = []
    
    # Split triplets into tupels. Use symmetry between AB and BA. 
    for n in range(number):
            i_img_A = int(triplets[n,0])
            i_img_B = int(triplets[n,1])
            i_img_C = int(triplets[n,2])
            
            # Conversion to vector
            A = vector_list[i_img_A]
            B = vector_list[i_img_B]
            C = vector_list[i_img_C]
            
            # Similar taste
            XYZ.append([A,B,C])
        
    # Convert to np.array
    XYZ = np.array(XYZ)
    
    X_dataset = tf.constant(XYZ[:,0])
    Y_dataset = tf.constant(XYZ[:,1])
    Z_dataset = tf.constant(XYZ[:,2])

    return X_dataset, Y_dataset, Z_dataset 

# Create datasets
#train_dataset = triple_to_dataset(train_triplets)


X, Y, Z, y = triple_to_dataset(train_triplets)

number = len(X)
X_train = X[:int(train_frac * number)]
X_val = X[int(train_frac * number):]
Y_train = Y[:int(train_frac * number)]
Y_val = Y[int(train_frac * number):]
Z_train = Z[:int(train_frac * number)]
Z_val = Z[int(train_frac * number):]
y_train = y[:int(train_frac * number)]
y_val = y[int(train_frac * number):]

X_test, Y_test, Z_test = triple_to_dataset_test(test_triplets)




""" Model """

X_input = layers.Input(name="X", shape=[len(feature_list)])
Y_input = layers.Input(name="Y", shape=[len(feature_list)])
Z_input = layers.Input(name="Z", shape=[len(feature_list)])

class ConcatenationLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, X, Y, Z):
        return layers.Concatenate()([X, Y, Z])

leaky_relu = layers.LeakyReLU(alpha=0.1)
dim_red = layers.Dense(400, activation=leaky_relu, input_shape=(len(feature_list),))

concat = ConcatenationLayer()(
    dim_red(X_input),
    dim_red(Y_input),
    dim_red(Z_input),
)

model = Model(
    inputs=[X_input, Y_input, Z_input], outputs=concat
)

flatten = layers.Flatten()(model.outputs[0]) 
dense1 = layers.Dense(50, activation=leaky_relu)(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(10, activation=leaky_relu)(dense1)
dense2 = layers.BatchNormalization()(dense2)
classifier_layer = layers.Dense(1, activation="sigmoid")(dense2)

model = Model(inputs=[X_input, Y_input, Z_input], outputs=classifier_layer)
model.summary()

                                                               
""" Train binary classifier """

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x=[X_train, Y_train, Z_train], 
          y=y_train, 
          validation_data=([X_val, Y_val, Z_val], y_val),
          batch_size=batch_size, 
          epochs=epochs)

""" Prediction """

pred = model.predict(x=[X_test, Y_test, Z_test])
print(pred)


def decide_it(array): 
    zeros = 0
    ones = 0
    
    result = np.zeros(len(array))
    
    for i, y in enumerate(array):
        if y > 0.5:
            result[i] = 1   
            ones += 1
        else: 
            zeros += 1
            
    print("Zeros: " + str(zeros)) 
    print("Ones: " + str(ones)) 
    
    return result

result = decide_it(pred)
    

""" Prediction to CSV """

np.savetxt("result_dish_triple.txt", result, fmt='%1.0i')


""" Validation """

pred_val = model.predict(x=[X_val, Y_val, Z_val])
result_val = decide_it(pred_val)

good = 0
bad = 0

for i in range(len(result_val)): 
    if result_val[i] == y_val[i]: 
        good += 1
    else: 
        bad += 1
        
        
print("Good: " + str(good)) 
print("Bad: " + str(bad)) 
print("Good / (Good + Bad): " + str((good - 0.5 * good - 0.5 * bad)/(good + bad))) 
    

""" Log """ 



