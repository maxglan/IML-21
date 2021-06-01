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

do_extract = False
number_of_images = 10

""" Load data """

# Create paths
images_path = pathlib.Path().absolute() / "food"

#importing the txt with the image names
train_triplets = np.genfromtxt("train_triplets.txt", dtype='str')[:10]
test_triplets = np.genfromtxt("test_triplets.txt", dtype='str')[:10]


""" Extract features """

def get_features(do_extract: bool=do_extract): 

    if do_extract: 
    
        feature_set = set()
        prediction_list = []
        
        for i in range(number_of_images):
        
            img_path = images_path / (str(i).zfill(5) + ".jpg")
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
        
feature_list, prediction_list = get_features(do_extract)
print(feature_list)
print(prediction_list)


""" Feature <-> index mapping """

# We enumerate the features to switch easily between feature and index

feature_space_dimension = len(feature_list)

feature_to_index = {}
for i, feature in enumerate(feature_list): 
    feature_to_index[feature] = i
    
print(feature_to_index)

index_to_feature = {value:key for key, value in feature_to_index.items()}
print(index_to_feature)



""" Prediction to vector conversion """

prediction_vector_list = []

for prediction in prediction_list:
    
    prediction_vector = np.zeros(feature_space_dimension)
    
    for prediction_item in prediction:
        
        nr, name, probability = prediction_item
        index = feature_to_index[name]
        prediction_vector[index] = probability
        
    prediction_vector_list.append(prediction_vector)
    
print(prediction_vector_list) 


""" Generate dataset similar to the result of path_to_dataset() """

def triplets_to_dataset(): 
    
    dataset = 0 
    
    return dataset


""" Model """



""" Training """


""" Prediction """



""" Prediction to CSV """

    
    









    