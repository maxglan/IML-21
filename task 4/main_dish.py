# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 09:25:17 2021

@author: roman

Word classification ansatz
"""


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

import pathlib
import pickle

model = ResNet50(weights='imagenet')


""" Parameters """

do_extract = True
get_only = 10000

""" Load data """

# Create paths
images_path = pathlib.Path().absolute() / "food"

#importing the txt with the image names
train_triplets = np.genfromtxt("train_triplets.txt", dtype='str')[:get_only]
test_triplets = np.genfromtxt("test_triplets.txt", dtype='str')[:get_only]


""" Extract features """

def get_features(do_extract: bool=do_extract, number_of_images=10000): 
    
    prediction_list = []
    feature_set = set()
    
    try: 
        with open('feature_list.pkl', 'rb') as f:
            feature_list = pickle.load(f)
        with open('prediction_list.pkl', 'rb') as f:
            prediction_list = pickle.load(f)
        
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
            print('Predicted:', decode_predictions(preds, top=3)[0])
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

for i_img in [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
    feature_list, prediction_list = get_features(do_extract, number_of_images=i_img)
    
print(feature_list)
print(prediction_list)


""" Feature <-> index mapping """

# We enumerate the features to switch easily between feature and index

feature_space_dimension = len(feature_list)

feature_to_index = {}
for i_f, feature in enumerate(feature_list): 
    feature_to_index[feature] = i_f
    
print(feature_to_index)

index_to_feature = {value:key for key, value in feature_to_index.items()}
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
    
    
    # Convert path to datasets
    X_dataset = tf.data.Dataset.from_tensor_slices(tupels[:,0])
    Y_dataset = tf.data.Dataset.from_tensor_slices(tupels[:,1])

    # Combine datasets
    dataset = tf.data.Dataset.zip((X_dataset, Y_dataset, y))
    
    return dataset

training_dataset = triple_to_dataset(train_triplets)


print(training_dataset)

""" Model """



""" Training """


""" Prediction """



""" Prediction to CSV """

    
    









    