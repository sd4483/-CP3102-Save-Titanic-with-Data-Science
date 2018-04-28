# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 18:32:53 2018

@author: sudhe
"""



# Missing Data [Training Set]
X_missing = X.isnull()
X_missing_sum = X.isnull().sum()
group_sex_class = X.groupby(['Sex','Pclass'])
X.Age = group_sex_class.Age.fillna(X.Age.mean())
#X.Embarked = X.Embarked.fillna('0')
#X = X.dropna(how='any', axis = 0)
X_missing_dealtwith = X.isnull().sum()


#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder           
labelencoder_sex = LabelEncoder()                                   
X.Sex = labelencoder_sex.fit_transform(X.Sex)
#labelencoder_embarked = LabelEncoder()                                   
#X.Embarked = labelencoder_embarked.fit_transform(X.Embarked)
#onehotencoder = OneHotEncoder(categorical_features = [6])
#X = onehotencoder.fit_transform(X).toarray()
#dummies = pd.get_dummies(X['Embarked'], drop_first = True)
#X = pd.concat([X,dummies], axis = 1)
#X = X.drop(['Embarked'], axis = 1)


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
labelencoder_embarked_test = LabelEncoder()                                   
test.Embarked = labelencoder_embarked_test.fit_transform(test.Embarked)
onehotencoder_test = OneHotEncoder(categorical_features = [6])
test = onehotencoder_test.fit_transform(test).toarray()


# Predicting the model on real test set results
Test_pred = classifier.predict(test)