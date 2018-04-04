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
X = train_dataset.iloc[:,[2,4,5,6,7,9]]
y = train_dataset.iloc[:,1]
test = test_dataset.iloc[:,[1,3,4,5,6,8]]


# Missing Data [Training Set]
X_missing = X.isnull()
X_missing_sum = X.isnull().sum()
avg_Age = X.Age.mean()
X.Age = X.Age.fillna(value = avg_Age)
X_missing_dealtwith = X.isnull().sum()


#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder           
labelencoder_sex = LabelEncoder()                                   
X.Sex = labelencoder_sex.fit_transform(X.Sex)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# PREPROCESSING TEST SET

# Missing Data [Test Set]
test_missing = test.isnull()
test_missing_sum = test.isnull().sum()
avg_Age = test.Age.mean()
test.Age = test.Age.fillna(value = avg_Age)
avg_Fare = test.Fare.mean()
test.Fare = test.Fare.fillna(value = avg_Fare)
test_missing_dealtwith = test.isnull().sum()


#Encoding Categorical data for test set        
labelencoder_testdata_sex = LabelEncoder()                                   
test.Sex = labelencoder_testdata_sex.fit_transform(test.Sex)

# Predicting the model on real test set results
Test_pred = classifier.predict(test)