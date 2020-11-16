# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:35:21 2020

@author: goury
"""

import tensorflow
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import models, layers
import numpy as np

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
# EXPLORE DATA
print(X_train[0])
print(f'Training data : {X_train.shape}')
print(f'Test data : {X_test.shape}')
print(f'Training sample : {X_train[0]}')
print(f'Training target sample : {y_train[0]}')


print(X_train[:2])
print(y_train[:5])
# Data should be normalaized because of the wide diffrence between the values of X_train and y_train

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# first we fit the scaler on the training dataset
scaler.fit(X_train)

# then we call the transform method to scale both the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# a sample output
print(X_train_scaled[0])

# BUILDING MODEL
def baseline_model():

    model = models.Sequential()
    model.add(layers.Dense(8, activation='relu', input_shape=[X_train.shape[1]]))
    model.add(layers.Dense(16, activation='relu'))
    
    # output layer
    model.add(layers.Dense(1))
    
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
"""
# TRAINING 
mod = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100)

model.evaluate(X_test_scaled, y_test)

# we get a sample data (the first 2 inputs from the training data)
to_predict = X_train_scaled[:2]
# we call the predict method
predictions = model.predict(to_predict)

print(predictions)

print(y_train[:2])
"""
seed = 7
np.random.seed(seed)
estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))