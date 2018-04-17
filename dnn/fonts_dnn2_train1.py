#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:05:52 2018

@author: gubo
"""

from keras.layers import Dense
from keras.models import Sequential

# Create the model
hidden_layer = Dense(100, input_dim=X_train.shape[1])
output_layer = Dense(y_train.shape[1])
model = Sequential(layers=[hidden_layer, output_layer])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# In[53]:

model.fit(X_train, y_train, epochs=100, verbose=False)
y_pred = model.predict(X_test)