
# coding: utf-8

# In[6]:

import numpy as np 
from PIL import Image, ImageDraw, ImageFont 
from skimage import transform as tf
from skimage.util import random_noise 
import cv2
# In[32]:

def create_captcha(text, shear=0, size=(150, 30), scale=1):
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
    return image


# In[39]:

from matplotlib import pyplot as plt
#image = create_captcha("GENE", shear=0.25, scale=0.8)
#plt.imshow(image, cmap='Greys')
image = create_captcha("THRONES", shear=0.35, scale=1.2)
image = random_noise(image)
plt.figure(0)
plt.imshow(image, cmap='Greys')


image_threshold = 0.25
image = image * (image > image_threshold)

# In[41]:

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


# In[42]:

subimages = segment_image(image)


# In[43]:

f, axes = plt.subplots(1, len(subimages), figsize=(10, 3)) 
for i in range(len(subimages)): 
    axes[i].imshow(subimages[i], cmap="gray")



