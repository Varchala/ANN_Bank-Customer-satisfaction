# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as py
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

#import the data
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#Encoding Categorical Data
x.iloc[:,2] = LabelEncoder().fit_transform(x.iloc[:,2])
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(columnTransformer.fit_transform(x), dtype = np.str)
#x = x[:, 1:]

#splitting of the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2,   random_state=0)

#Feature Scaling using StandardScaler	
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

#Building ANN
model = keras.Sequential()
model.add(keras.layers.Dense(6, activation = 'relu', input_shape = (12,)))
model.add(keras.layers.Dense(6, activation = 'relu'))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

#compile an evaluate model
model.compile(loss = 'binary_crossentropy' ,
              optimizer = 'adam', 
              metrics = ['accuracy'])

model.summary()

#fit the training model to the dataset
model.fit(x_train, y_train, epochs = 100, batch_size = 32)

#predict the outcome using the existing model
print(model.predict(StandardScaler().fit_transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))>0.5)

#Predict the test set results
y_pred = model.predict(x_test)
y_pred = ( y_pred > 0.5 )
print ( np.concatenate ( ( y_pred.reshape ( len( y_pred ) , 1 )  , y_pred.reshape ( len( y_pred ), 1 ) ), 1))

#Print confusion matrix and overall accuracy
cm = confusion_matrix(y_test, y_pred)
print( cm )
print(accuracy_score(y_test, y_pred))
