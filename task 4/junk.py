#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:45:03 2021

parts removed from previous main.py
"""
def feature_trafo(anchor, positive, negative):
    """takes all three lists of images and returns the output in the form (n_images, anchor_output + positive_output + negative_output)
    here each x_output is a reshaped array of the oroginal multi dimensional output array of the used NN """
    """Pretraining anchor"""
    features_anchor = base_model.predict(anchor)
    """Pretraining positive"""
    features_positive = base_model.predict(positive)
    """Pretraining negative"""
    features_negative = base_model.predict(negative)
    
    shape= np.shape(features_anchor)
    
    f_a = np.reshape(features_anchor, (shape[0], -1))
    f_p = np.reshape(features_positive, (shape[0], -1))
    f_n = np.reshape(features_negative, (shape[0], -1))
    
    return np.concatenate((f_a, f_p, f_n), axis=1)
cat = base_model.predict(image_set)
cats = feature_trafo(image_set, image_set,image_set)

"""calculating the feature transformation using the pretrained NN and saving them for future use"""
training_data = feature_trafo(anchor_dataset, positive_dataset, negative_dataset)
test_data = feature_trafo(anchor_test, positive_test, negative_test)

np.savetxt("train_features.csv", training_data)
np.savetxt("test_features.csv", test_data)


"""defining our new NN"""
length = len(training_data[0,:])
t = int(length/ 3)

def triplet_loss(true, A):
    margin=0.4
    a = A[:t]
    b = A[t: 2*t]
    c = A[2*t:]
    return max(0, np.linalg.norm(a, b) - np.linalg.norm(a, c) + margin)

def NN(input_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(420,input_dim=input_size, activation="relu"))
    model.add(tf.keras.layers.Dense(13, activation="relu"))
    model.compile()
    return model

single_input = tf.keras.Input(np.shape(training_data[0,:]))

single_model = NN(training_data[0,:])

concat = tf.keras.concatenate([single_model, single_model, single_model])

model = tf.keras.Model([single_input, single_input, single_input], concat)
model.compile( optimizer = 'adam', loss = triplet_loss)