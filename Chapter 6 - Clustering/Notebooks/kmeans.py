#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:29:48 2018

@author: emma
"""
import numpy as np 
import matplotlib.pyplot as plt

def k_means_segmentation(image, m1, m2):
    
    '''
      1.1. create doc string
    '''
   
    # 1.2. TO DO create seg image same size as input image but initialised as zeros
    seg=None
    
    # cast data as 16 bit int
    image=image.astype(np.uint16)
    seg=seg.astype(np.uint16)
                       
    # 1.3.  TO DO - Create a loop to run the agorithm for 10 iterations.
    for None:
        
        # 1.4. TO DO -calculate the squared i﻿ntensity distance (d_k)
        # between each pixel in the image and each cluster centroid m_k: d_k(i)=(image(i)-m_k)^2
        
        d_1=None
        d_2=None

        # 1.5. TO DO - Calculate the segmentation by assigning each index in 'seg' to the cluster with closer centroid:
        # Hint two lines - one for each cluster
        
        None
        None
        
        # 1.6. TO DO - Update the centroids:
   
        m1=None
        m2=None
        
        # display segmentation for this iteration
        plt.imshow(seg)
        plt.show()
        
        print('for iter {} the new cluster means are {} and {}'.format(i,m1,m2))
       
    return seg