
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.datasets import load_iris 
iris = load_iris() 
X = iris.data.astype(np.float32) 
y_true = iris.target.astype(np.int32)


# In[2]:

from sklearn.preprocessing import OneHotEncoder

y_onehot = OneHotEncoder().fit_transform(y_true.reshape(-1, 1))
y_onehot = y_onehot.astype(np.int64).todense()


# In[4]:
# divide data into test set and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, random_state=14)


# In[5]:

input_layer_size, hidden_layer_size, output_layer_size = 4, 6, 3


# In[7]:

from keras.layers import Dense
hidden_layer = Dense(output_dim=hidden_layer_size, input_dim=input_layer_size, activation='relu')
output_layer = Dense(output_layer_size, activation='sigmoid')


# In[8]:

from keras.models import Sequential
model = Sequential(layers=[hidden_layer, output_layer])


# In[9]:

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])


# In[10]:

history = model.fit(X_train, y_train, epochs = 200)


# In[18]:

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
plt.plot(history.epoch, history.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")


# use the neural network to predict the test set and run a classification report
## In[15]:
#
from sklearn.metrics import classification_report
y_pred = model.predict_classes(X_test)
print(classification_report(y_true=y_test.argmax(axis=1), y_pred=y_pred))
#
#
# change to 1000 epochs to see better prediction results
## In[17]:
#
#history = model.fit(X_train, y_train, epochs=1000, verbose=False)
#
#
## In[19]:
#
#y_pred = model.predict_classes(X_test)
#print(classification_report(y_true=y_test.argmax(axis=1), y_pred=y_pred))
#
