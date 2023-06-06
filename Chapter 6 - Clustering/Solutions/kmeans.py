#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:29:48 2018

@author: emma
"""
import numpy as np 
import matplotlib.pyplot as plt

def k_means_segmentation(image, m1, m2):
    
    '''k-means segmentation:
     
     Performs segmentation of an image into 2 classes. The
     input parameters is the image to be segmented and the initial values the
     centroids of the two clusters m1 and m2. The output parameters are the
     segmented image (seg) with labels 1 and 2, and  final cluster centroids.
    
    input:
        image: image to be segmented
        m1:  initial centroid 1 guess
        m2:  initial centroid 2 guess    
    
    output:
        seg: segmented image
    '''
   
    #create seg image same size as image but initialised as zeros
    seg=np.zeros(image.shape)
    
    # cast data as 16 bit int
    image=image.astype(np.uint16)
    seg=seg.astype(np.uint16)
                       
    # Run the agorithm in a for loop for 10 iterations.
    for i in range(10):
        # For each pixel in the image calculate intensity distance 
        #d_k to each cluster centroid m_k: d_k(i)=(image(i)-m_k)^2
        
        d_1=(image-m1)**2;
        d_2=(image-m2)**2;
        # Calculate the segmentation seg by assigning each pixel to a cluster 
        #with closer centroid:
        
        seg[d_1<d_2]=1
        seg[d_2<d_1]=2
        
        # Update the centroids:
   
        m1=np.mean(image[seg==1]);
        m2=np.mean(image[seg==2]);
        
        # display segmentation for this iteration
        plt.imshow(seg)
        plt.show()
        
        print('for iter {} the new cluster means are {} and {}'.format(i,m1,m2))
       
    return seg