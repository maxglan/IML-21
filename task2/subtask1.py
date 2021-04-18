# -*- coding: utf-8 -*-

"""
@author: romanwixinger
"""

""" Sources """

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html


""" Imports """

import numpy as np
import pandas as pd

from scipy.stats import logistic # sigmoid function

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
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
             clf=classifiers[-1]): 
    """
    Arguments: 
    - trainf: Dataframe with train_features
    - trainl: Dataframe with train_labels
    
    Description of the function: 
    - 
    
    Return value: 
    - df_submission: Dataframe with the predicted values. 
    
    """
    
    # Reshape the dataset if necessary
    
    
    # Train 
    clf.fit(X=trainf, y=trainl)
    print(clf.get_param([]))
    
    
    # Test
    
    
    # Predict

    prediction = clf.predict(test)
    
    
    return prediction