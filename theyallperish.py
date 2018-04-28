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

train_missing_sum = train_dataset.isnull().sum()
test_missing_sum = test_dataset.isnull().sum()

#Missing age values
from sklearn.ensemble import RandomForestRegressor

def missing_ages(X):
    
    age = X[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    
    age_known = age[age.Age.notnull()].as_matrix()
    age_unknown = age[age.Age.isnull()].as_matrix()
    
    m = age_known[:,1:]
    n = age_known[:,0]
    
    regressor = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    regressor.fit(m,n)
    
    predict_ages = regressor.predict(age_unknown[:,1::])
    
    X.loc[(X.Age.isnull(), 'Age')] = predict_ages
    
    return X, regressor

train_dataset, regressor = missing_ages(train_dataset)

#Missing Cabin Values
def missing_cabin(X):
    X.loc[(X.Cabin.notnull()), 'Cabin'] = 'Yes'
    X.loc[(X.Cabin.isnull()), 'Cabin'] = 'No'
    return X

train_dataset = missing_cabin(train_dataset)


#Encoding Categorical Variables
encode_cabin = pd.get_dummies(train_dataset['Cabin'], prefix='Cabin')
encode_embarked = pd.get_dummies(train_dataset['Embarked'], prefix='Embarked')
encode_Sex = pd.get_dummies(train_dataset['Sex'], prefix='Sex')
encode_Pclass = pd.get_dummies(train_dataset['Pclass'], prefix='Pclass')

#Appending all the variables to dataframe
X = pd.concat([train_dataset, encode_cabin, encode_embarked, encode_Sex, encode_Pclass], axis=1)

#Dropping the duplicates
X.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler_param = scaler.fit(X[['Age', 'Fare']])
X['Age_scaled'] = scaler.fit_transform(X[['Age', 'Fare']], scaler_param)[:,0]
X['Fare_scaled'] = scaler.fit_transform(X[['Age', 'Fare']], scaler_param)[:,1]

#Filtering the duplicates
train_df = X.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.as_matrix()

#Matrix of independent features and dependent vector
features = train_np[:,1:]
vector =train_np[:,0]

#Using google's tpot library to make a pipeline
from tpot import TPOTClassifier
pipeline_optimizer = TPOTClassifier(generations=5, random_state=0, verbosity=2)
pipeline_optimizer.fit(features,vector)
pipeline_optimizer.export("some.py")

#Using Bagging regression with logistic regression
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0, penalty ='l1',tol=1e-6,random_state = 0)
bagging_r = BaggingRegressor(lr, n_estimators=100, max_samples =0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_r.fit(features, vector)

#Using cross validation to train the algorithm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

#Using support vector machines to train the algorithm
from sklearn.svm import SVC
clf_svc = SVC()
scoring = 'accuracy'
score_svc = cross_val_score(clf_svc, features, vector, cv=k_fold, n_jobs=1, scoring=scoring)
print_score_svc = round(np.mean(score_svc)*100,2)

#Using Random forest classifier to train the algorithm
from sklearn.ensemble import RandomForestClassifier
clf_rfc = RandomForestClassifier(n_estimators=100)
scoring = 'accuracy'
score_rfc = cross_val_score(clf_rfc, features, vector, cv=k_fold, n_jobs=1, scoring=scoring)
print_score_rfc = round(np.mean(score_rfc)*100,2)


#Fitting SVC
clf = SVC()
clf.fit(features, vector)


#Working on Test data
test_dataset.loc[(test_dataset.Fare.isnull()), 'Fare'] = 0
X_temp = test_dataset[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
age_test_missing = X_temp[test_dataset.Age.isnull()].as_matrix()

k = age_test_missing[:, 1:]
predict_ages = regressor.predict(k)

test_dataset.loc[(test_dataset.Age.isnull(), 'Age')] = predict_ages
test_dataset = missing_cabin(test_dataset)
encode_cabin = pd.get_dummies(test_dataset['Cabin'], prefix='Cabin')
encode_Embarked_test = pd.get_dummies(test_dataset['Embarked'], prefix='Embarked')
encode_Sex_test = pd.get_dummies(test_dataset['Sex'], prefix='Sex')
encode_Pclass_test = pd.get_dummies(test_dataset['Pclass'], prefix='Pclass')

y = pd.concat([test_dataset, encode_cabin, encode_Embarked_test, encode_Sex_test, encode_Pclass_test], axis=1)
y.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

y['Age_scaled'] = scaler.fit_transform(y[['Age', 'Fare']], scaler_param)[:,0]
y['Fare_scaled'] = scaler.fit_transform(y[['Age', 'Fare']], scaler_param)[:,1]

test_df = y.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
test = test_df.as_matrix()
y_predict_bagging = bagging_r.predict(test)
y_predict_randomforest = clf_rfc.predict(test)
y_predict_svc = clf.predict(test)
y_predict_tpot = pipeline_optimizer.predict(test)

baggingregressor_df = pd.DataFrame({'PassengerId':test_dataset['PassengerId'].as_matrix(), 'Survived':y_predict_bagging.astype(np.int32)})
randomforest_df = pd.DataFrame({'PassengerId':test_dataset['PassengerId'].as_matrix(), 'Survived':y_predict_randomforest})
svc_df = pd.DataFrame({'PassengerId':test_dataset['PassengerId'].as_matrix(), 'Survived':y_predict_svc.astype(np.int32)})
baggingregressor_df.to_csv("bagging_predictions.csv", index=False)
randomforest_df.to_csv("randomforest_predictions.csv", index=False)
svc_df.to_csv("svc_predictions.csv", index=False)


