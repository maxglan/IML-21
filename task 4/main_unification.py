# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 22:37:15 2021

@author: roman

Unification
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
top = 50 # Only use the first "top" prediction of ResNet50
min_number = 2 # Minimum number of times that a feature should appear in total

# Training
batch_size = 65
epochs = 30
#epochs = 5
threshold = 6000

# Optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.00005, epsilon=0.0001) # learning_rate = 1e-4, epsilon = 1e-7

sgd = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.2, nesterov=False, name='SGD')

opt = adam


""" Load data """

# Create paths
images_path = pathlib.Path().absolute() / "food"

#importing the txt with the image names
train_triplets = np.genfromtxt("train_triplets.txt", dtype='str')[:get_only]
test_triplets = np.genfromtxt("test_triplets.txt", dtype='str')[:get_only]

train_indices = [i for i in range(len(train_triplets)) if (int(train_triplets[i][0]) < threshold and int(train_triplets[i][1]) < threshold and int(train_triplets[i][2]) < threshold)]
val_indices = [i for i in range(len(train_triplets)) if (int(train_triplets[i][0]) >= threshold and int(train_triplets[i][1]) >= threshold and int(train_triplets[i][2]) >= threshold)]

print("Number of train triples: ", len(train_indices))
print("Number of val triples: ", len(val_indices))


""" Extract features """

def get_features(do_load: bool=True, do_extract: bool=False, number_of_images: int=10): 
    
    prediction_list = []
    feature_set = set()
    
    if do_load: 
        try: 
            with open('feature_list_50.pkl', 'rb') as f:
                feature_list = pickle.load(f)
            with open('prediction_list_50.pkl', 'rb') as f:
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
            #print('Predicted:', decode_predictions(preds, top=50)[0])
            prediction_list.append(decode_predictions(preds, top=top)[0])
            
            for prediction_item in decode_predictions(preds, top=top)[0]:
                nr, name, probability = prediction_item                 
                feature_set.add(name)
                
        feature_list = list(feature_set)
        #print(feature_list) 
    
        with open('feature_list_50.pkl', 'wb') as f:
            pickle.dump(feature_list, f)
        with open('prediction_list_50.pkl', 'wb') as f:
            pickle.dump(prediction_list, f)
        
            
    else:
        with open('feature_list_50.pkl', 'rb') as f:
            feature_list = pickle.load(f)
        with open('prediction_list_50.pkl', 'rb') as f:
            prediction_list = pickle.load(f)
            
    return feature_list, prediction_list
feature_list, prediction_list = get_features(do_load, do_extract, number_of_images)
def reduced_feature_list(prediction_list):
    
    train_features = set()
    test_features = set() 
    
    for prediction in prediction_list[:4999]:
        for prediction_item in prediction: 
            nr, name, probability = prediction_item 
            train_features.add(name)
    
    for prediction in prediction_list[4999:]:
        for prediction_item in prediction: 
            nr, name, probability = prediction_item 
            test_features.add(name)
        
    #print("train_features: ", train_features)
    #print("test_features: ", test_features)
    
    feature_list = train_features.intersection(test_features)
    feature_list = list(feature_list)

    print("The reduced feature list has length ", len(feature_list))
    
    # Delete features that do not come often
    feature_number_dict = {}
    for prediction in prediction_list:
        for prediction_item in prediction: 
            nr, name, probability = prediction_item 
            if name in feature_list: 
                if name in feature_number_dict: 
                    feature_number_dict[name] += probability
                else: 
                    feature_number_dict[name] = probability
    
    final_feature_list = []
    
    for name in feature_list: 
        if feature_number_dict[name] >= min_number: 
            final_feature_list.append(name)
    
    print("The final feature list has length ", len(final_feature_list))

    return final_feature_list
feature_list = reduced_feature_list(prediction_list)


""" Feature <-> index mapping """

feature_space_dimension = len(feature_list)
feature_to_index = {}
for i_f, feature in enumerate(feature_list): 
    feature_to_index[feature] = i_f
index_to_feature = {value:key for key, value in feature_to_index.items()}


""" Image (i_img) <-> prediction <-> vector mapping via indexing"""

def prediction_to_vector(prediction): 
    vector = np.zeros(feature_space_dimension)
    
    for prediction_item in prediction:
        
        nr, name, probability = prediction_item
        
        if name in feature_to_index.keys(): 
            index = feature_to_index[name]
            vector[index] = probability
        
    return vector
vector_list = []
for prediction in prediction_list:
    vector = prediction_to_vector(prediction)
    vector_list.append(vector)


""" Generate triple dataset """

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

X_train_3, Y_train_3, Z_train_3, y_train_3 = triple_to_dataset(train_triplets[train_indices])
if len(val_indices) > 0: 
    X_val_3, Y_val_3, Z_val_3, y_val_3 = triple_to_dataset(train_triplets[val_indices])
else: 
    X_val_3, Y_val_3, Z_val_3, y_val_3 = triple_to_dataset(train_triplets[[0]])
X_test_3, Y_test_3, Z_test_3 = triple_to_dataset_test(test_triplets)


""" Generate tuple dataset """

def triple_to_dataset_tuple(triplets):
    """ Make the dateset ready for binary classification """
    
    number = triplets.shape[0]
    tupels = []
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
    
    X_dataset = tf.constant(tupels[:,0])
    Y_dataset = tf.constant(tupels[:,1])
    y_dataset = tf.constant(y)
    
    return X_dataset, Y_dataset, y_dataset #dataset

def triple_to_dataset_test_tuple(triplets):
    """ Make the dateset ready for binary classification """
    
    number = triplets.shape[0]
    tupels = []
    
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
            tupels.append([A,B])
            tupels.append([A,C])
        
    # Convert to np.array
    tupels = np.array(tupels)
    
    X_dataset = tf.constant(tupels[:,0])
    Y_dataset = tf.constant(tupels[:,1])

    return X_dataset, Y_dataset #dataset

X_train_2, Y_train_2, y_train_2 = triple_to_dataset_tuple(train_triplets[train_indices])
if len(val_indices) > 0:
    X_val_2, Y_val_2, y_val_2 = triple_to_dataset_tuple(train_triplets[val_indices])
else: 
    X_val_2, Y_val_2, y_val_2 = triple_to_dataset_tuple(train_triplets)

X_test_2, Y_test_2 = triple_to_dataset_test_tuple(test_triplets)



""" Model """

def create_triple_model(dim_red_nodes: int=400, 
                     dense0_nodes: int=300, 
                     dense1_nodes: int=50,
                     dense2_nodes: int=10): 
    """ Create the model that Max fine tuned """
    
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
    dim_red = layers.Dense(dim_red_nodes, activation=leaky_relu, input_shape=(len(feature_list),))
    
    concat = ConcatenationLayer()(
        dim_red(X_input),
        dim_red(Y_input),
        dim_red(Z_input),
    )
    
    model = Model(
        inputs=[X_input, Y_input, Z_input], outputs=concat
    )
    
    flatten = layers.Flatten()(model.outputs[0])
    drop = layers.Dropout(0.4, seed=3)(flatten)
    dense0 = layers.Dense(dense0_nodes, activation=leaky_relu)(drop)
    dense0 = layers.BatchNormalization()(dense0)
    drop = layers.Dropout(0.4, seed=5)(dense0)
    dense1 = layers.Dense(dense1_nodes, activation=leaky_relu)(drop)
    dense1 = layers.BatchNormalization()(dense1)
    drop = layers.Dropout(0.5, seed=3)(dense1)
    dense2 = layers.Dense(dense2_nodes, activation=leaky_relu)(drop)
    dense2 = layers.BatchNormalization()(dense2)
    classifier_layer = layers.Dense(1, activation="sigmoid")(dense2)
    
    model = Model(inputs=[X_input, Y_input, Z_input], outputs=classifier_layer)
    model.summary()

    return model

model_3_Max = create_triple_model()
model_3_Max_var = create_triple_model(dim_red_nodes=401, dense0_nodes=301, dense1_nodes=51, dense2_nodes=11) # Small variation
model_3_Roman = create_triple_model(dim_red_nodes=281, dense0_nodes=200) # Node according to food space dimension

model_3_list = [model_3_Max, model_3_Max_var, model_3_Roman]

def create_tuple_model(dim_red_nodes: int=400, 
                     dense1_nodes: int=50,
                     dense2_nodes: int=10):
    
    X_input = layers.Input(name="X", shape=[len(feature_list)])
    Y_input = layers.Input(name="Y", shape=[len(feature_list)])
    
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
    dim_red = layers.Dense(dim_red_nodes, activation=leaky_relu, input_shape=(len(feature_list),))
    
    concat = ConcatenationLayer()(
        dim_red(X_input),
        dim_red(Y_input),
    )
    
    model = Model(
        inputs=[X_input, Y_input], outputs=concat
    )
    
    flatten = layers.Flatten()(model.outputs[0]) 
    dense1 = layers.Dense(dense1_nodes, activation=leaky_relu)(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(dense2_nodes, activation=leaky_relu)(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    classifer_layer = layers.Dense(1, activation="sigmoid")(dense2)
    
    model = Model(inputs=[X_input, Y_input], outputs=classifer_layer)
    model.summary()
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

model_2 = create_tuple_model()
model_2_var = create_tuple_model(dim_red_nodes=401, dense1_nodes=51, dense2_nodes=11)

model_2_list = [model_2, model_2_var]


""" Arguments """

fit_args3 = {'x': [X_train_3, Y_train_3, Z_train_3], 
          'y': y_train_3, 
          'validation_data':([X_val_3, Y_val_3, Z_val_3], y_val_3),
          'batch_size': batch_size, 
          'epochs': epochs
          }

fit_args2 = {'x': [X_train_2, Y_train_2], 
          'y': y_train_2, 
          'validation_data':([X_val_2, Y_val_2], y_val_2),
          'batch_size': batch_size, 
          'epochs': epochs
          }

pred_args3 = {'x': [X_test_3, Y_test_3, Z_test_3]}
pred_args2 = {'x': [X_test_2, Y_test_2]}
    

""" Train binary classifier """


def train_and_predict(model, fit_args, pred_args): 
    """ Trains the model and makes prediction on another data set """
    
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(**fit_args)
    pred = model.predict(**pred_args)
    
    return pred

def decide_it_2(array): 
    result = np.zeros(int(len(array)/2))

    for i in range(int(len(array)/2)): 
        y_AB = array[2*i][0]
        y_AC = array[2*i+1][0]
        
        result[i] = 0.5 + 0.5 * (y_AB - y_AC)
            
    return result

def predict_on_test():
    """ Trains and runs multiple models and unifies the results into a list of predictions """
    
    # List to store the prediction with float values in [0,1]
    pred_list = []
    
    # Make predictions
    for model in model_3_list: 
        pred = train_and_predict(model, fit_args3, pred_args3)
        pred_list.append(pred)
    for model in model_2_list: 
        pred = train_and_predict(model, fit_args2, pred_args2)
        pred = decide_it_2(pred)
        pred_list.append(pred)    
    
    return pred_list
        
pred_list = predict_on_test()



""" Unification """

decision_threshold = 0.5

def decide_it_final(array_list): 
    zeros = 0
    ones = 0
    
    result = np.zeros(len(array_list[0]))
    
    for array in array_list: 
        for i in range(len(array)): 
            result[i] += array[i]
            
    for i in range(len(result)): 
        if result[i] >= decision_threshold * len(array_list): 
            result[i] = 1
            ones += 1
        else:
            result[i] = 0
            zeros += 1
    
    print("Zeros: " + str(zeros)) 
    print("Ones: " + str(ones)) 
    print("Fraction of zeros: ", str(zeros/(zeros+ones)))
    
    return result

result = decide_it_final(pred_list)

""" Prediction to CSV """

np.savetxt("result_dish_unification.txt", result, fmt='%1.0i')






