# -*- coding: utf-8 -*-

"""
@author: romanwixinger
"""

""" Sources """

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html


""" Imports """

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import svm


""" Resources """
 
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    svm.SVC(kernel='sigmoid'),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    svm.SVR()]


""" Functions """

def subtask1(trainf: np.array, 
             trainl: pd.DataFrame, 
             test, 
             clf=classifiers[3]): 
    """
    Arguments: 
    - trainf: Dataframe with train_features
    - trainl: Dataframe with train_labels
    
    Description of the function: 
    - 
    
    Return value: 
    - df_submission: Dataframe with the predicted values. 
    
    """
    
    labels = ["LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", "LABEL_Alkalinephos", 
          "LABEL_Bilirubin_total", "LABEL_Lactate", "LABEL_TroponinI", "LABEL_SaO2", 
          "LABEL_Bilirubin_direct", "LABEL_EtCO2"]

    model={}

    prediction = np.zeros((len(test), len(labels)))

    for l, i in zip(labels, range(len(labels))):
        model[l] = clf()
        model[l].fit(trainf, trainl[l])
        
        prediction[:,i] = model[l].predict(test)
    
    return prediction