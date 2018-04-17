#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:05:52 2018

@author: gubo
"""

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential

# Create the model, CNN
output_layer = Dense(y_train.shape[1], activation='softmax')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=(20,20,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(output_layer)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# In[53]:

from sklearn.metrics import classification_report

# train stage 1
print('train stage 1')
model.fit(X_train, y_train, epochs=10, verbose=False)

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
model.fit(X_train_n, y_train_n, epochs=30, verbose=False)

print('w/o noise')
y_pred = model.predict(X_test)
print(classification_report(y_pred=y_pred.argmax(axis=1),
y_true=y_test.argmax(axis=1)))

print('with noise')
y_pred_n = model.predict(X_test_n)
print(classification_report(y_pred=y_pred_n.argmax(axis=1),
y_true=y_test_n.argmax(axis=1)))