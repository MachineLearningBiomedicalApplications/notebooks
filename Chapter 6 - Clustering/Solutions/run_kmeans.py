#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:57:06 2018

@author: emma
"""

import matplotlib.pyplot as plt
import numpy as np
import kmeans


# load image using matplotlib
I = plt.imread('datasets/Cells.tif')

print(I)
#display image

plt.imshow(I)
plt.show()

plt.hist(np.reshape(I,(I.shape[0]*I.shape[1])),25)
plt.title('Intensity distribution')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()

# make guesses for cluster intensities (pick cluster centres from histogram)
m1=150
m2=240

print('initial guesses for cluster centroids:', m1,m2,I.shape)
if m1 is not None and m2 is not None:
    # apply k_means function 
    seg=kmeans.k_means_segmentation(I,m1,m2)


