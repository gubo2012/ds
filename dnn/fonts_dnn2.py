
# coding: utf-8

# In[6]:

import numpy as np 
from PIL import Image, ImageDraw, ImageFont 
import fonts_dnn_func
from sklearn.utils import check_random_state

from matplotlib import pyplot as plt

random_state = check_random_state(17) 

image, target = fonts_dnn_func.generate_sample(random_state, 1) 
plt.imshow(image, cmap="Greys")

reshape_flag=0
# image w/o noise
dataset, targets = fonts_dnn_func.generate_dataset_target(random_state, 1000)
X, y = fonts_dnn_func.generate_x_y(dataset, targets, reshape_flag)

# image with noise
dataset_n, targets_n = fonts_dnn_func.generate_dataset_target(random_state, 1000, noise_flag=1)
X_n, y_n = fonts_dnn_func.generate_x_y(dataset_n, targets_n, reshape_flag)

from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_n, y_n, train_size=0.9)
