# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 03:06:26 2018

@author: sudhe
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
X_train = train_dataset.iloc[:, [2,4,5,6,7,9,11]].values
y_train = train_dataset.iloc[:, 1].values
X_test = test_dataset.iloc[:,[1,3,4,5,6,8,10]].values

#Missing Values
from sklearn.preprocessing import Imputer                               
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)  
imputer = imputer.fit(X_train[:,2])                                       
X_train[:,2] = imputer.transform(X_train[:,2])                               


#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder           
labelencoder_country = LabelEncoder()                                   
X[:,0] = labelencoder_country.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])               
X = onehotencoder.fit_transform(X).toarray()

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)