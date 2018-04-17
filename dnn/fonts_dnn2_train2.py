#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:05:52 2018

@author: gubo
"""

from keras.layers import Dense
from keras.models import Sequential
hidden_layer = Dense(100, input_dim=X_train.shape[1])
output_layer = Dense(y_train.shape[1])
# Create the model
model = Sequential(layers=[hidden_layer, output_layer])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# In[53]:

from sklearn.metrics import classification_report

# train stage 1
print('train stage 1')
model.fit(X_train, y_train, epochs=50, verbose=False)

print('w/o noise')
y_pred = model.predict(X_test)
print(classification_report(y_pred=y_pred.argmax(axis=1),
y_true=y_test.argmax(axis=1)))

print('with noise')
y_pred_n = model.predict(X_test_n)
print(classification_report(y_pred=y_pred_n.argmax(axis=1),
y_true=y_test_n.argmax(axis=1)))


# train stage 1
print('train stage 2')
model.fit(X_train_n, y_train_n, epochs=200, verbose=False)

print('w/o noise')
y_pred = model.predict(X_test)
print(classification_report(y_pred=y_pred.argmax(axis=1),
y_true=y_test.argmax(axis=1)))

print('with noise')
y_pred_n = model.predict(X_test_n)
print(classification_report(y_pred=y_pred_n.argmax(axis=1),
y_true=y_test_n.argmax(axis=1)))