
# coding: utf-8

# In[6]:

import numpy as np 
from PIL import Image, ImageDraw, ImageFont 
from skimage import transform as tf
from skimage.util import random_noise
from sklearn.utils import check_random_state

# In[32]:


letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") 
shear_values = np.arange(0, 0.5, 0.05)
scale_values = np.arange(0.5, 1.5, 0.1)

def create_captcha(text, shear=0, size=(50, 30), scale=1, noise_flag=0):
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(r"bretan/Coval-Black.otf", 22) 
    draw.text((0, 0), text, fill=1, font=font)
    image = np.array(im)
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    image = image / image.max()
    # Apply scale
    shape = image.shape
    shapex, shapey = (int(shape[0] * scale), int(shape[1] * scale))
    image = tf.resize(image, (shapex, shapey))
    if noise_flag==1:
        image = random_noise(image)
    return image

from skimage.measure import label, regionprops

def segment_image(image):
    # label will find subimages of connected non-black pixels
    labeled_image = label(image>0.2, connectivity=1, background=0)
    subimages = []
    # regionprops splits up the subimages
    for region in regionprops(labeled_image):
        # Extract the subimage
        start_x, start_y, end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x,start_y:end_y])
        if len(subimages) == 0:
            # No subimages found, so return the entire image
            return [image,]
    return subimages


def generate_sample(random_state=None, noise_flag=0): 
    random_state = check_random_state(random_state) 
    letter = random_state.choice(letters) 
    shear = random_state.choice(shear_values)
    scale = random_state.choice(scale_values)
    # We use 30,30 as the image size to ensure we get all the text in the image
    return create_captcha(letter, shear=shear, size=(30, 30), scale=scale, noise_flag=noise_flag), letters.index(letter)


def generate_dataset_target(random_state, size, noise_flag=0):
    dataset, targets = zip(*(generate_sample(random_state, noise_flag) for i in range(size))) 
#    dataset = np.array([tf.resize(segment_image(sample)[0], (20, 20)) for sample in dataset])
    dataset = np.array([tf.resize(sample, (20, 20)) for sample in dataset])
    dataset = np.array(dataset, dtype='float') 
    targets = np.array(targets)
    return dataset, targets

# In[48]:

from sklearn.preprocessing import OneHotEncoder 
onehot = OneHotEncoder() 

def generate_x_y(dataset, targets, reshape_flag=1):
    y = onehot.fit_transform(targets.reshape(targets.shape[0],1))
    y = y.todense()
    if reshape_flag==1:
        X = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
    else:
        X = dataset.reshape((dataset.shape[0], dataset.shape[1], dataset.shape[2], 1))
    return X,y





