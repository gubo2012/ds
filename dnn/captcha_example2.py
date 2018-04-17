
# coding: utf-8

# In[6]:

import numpy as np 
from PIL import Image, ImageDraw, ImageFont 
from skimage import transform as tf
from skimage import io
from skimage.util import random_noise 
import fonts_dnn_func
from matplotlib import pyplot as plt

#image = io.imread('img1.gif', as_grey=True)
image = io.imread('img2_2.png', as_grey=True)

image = 1 - image
plt.figure(0)
plt.imshow(image, cmap='Greys')

subimages = fonts_dnn_func.segment_image(image)

# temporarily manually re-order them
subimages_filtered=[]
subimages_filtered.append(subimages[1])
subimages_filtered.append(subimages[5])
subimages_filtered.append(subimages[2])
subimages_filtered.append(subimages[3])
subimages_filtered.append(subimages[0])
subimages_filtered.append(subimages[4])


f, axes = plt.subplots(1, len(subimages_filtered), figsize=(10, 3)) 
for i in range(len(subimages_filtered)): 
    axes[i].imshow(subimages_filtered[i], cmap="gray")