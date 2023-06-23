#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 13:47:34 2020

@author: Emma C. Robinson (adapted from James Clough 2018)
Machine Learning for Biomedical Applications
"""

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D # for 3D plotting with matplotlib
#import plot3Darrows

from sklearn.decomposition import PCA


# a function for creating the swiss roll data set
def create_spiral(X, num_rotations):
    """ Take 2D manifold X and output 3D spiral made from curling up X in 3D space """
    N = X.shape[0]
    r = np.exp(X[:,1] * num_rotations) * 0.5
    theta = X[:,1] * (2 * np.pi) * num_rotations
    Y = np.zeros((N, 3))
    Y[:,0] = X[:,0] * 6
    Y[:,1] = r * np.cos(theta)
    Y[:,2] = r * np.sin(theta)
    return Y


# call create_spiral to create the data set

N = 2000                       # number of datapoints
num_rotations = 1.2
X_m = np.random.random((N, 2)) # random points scattered in [0,1]^2

Y = create_spiral(X_m, num_rotations)

# Plot the manifold in 2D

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(121)
ax.set_xticks([])
ax.set_yticks([])
_ = ax.scatter(X_m[:,0], X_m[:,1], c=X_m[:,1], marker='o')

# Plot the manifold curled up into a 3D swiss-roll

ax = fig.add_subplot(122, projection='3d')
_ = ax.scatter(Y[:,0], Y[:,1], Y[:,2], c=X_m[:,1], marker='o')
#ax.view_init(azim=0,elev=0) # uncomment this line to change the view of the plot to the Y-Z plane


# Ex 1 TO DO - fit a PCA model to this data
model=None # only want the first 2 components

# 1.2 project data onto first two principal components
PCA_transform=None


# Uncomment to - Plot the data projected onto the PCA components

# =============================================================================
# fig = plt.figure(figsize=(16,8))
# ax = fig.add_subplot(121)
# ax.set_xticks([])
# ax.set_yticks([])
# _ = ax.scatter(PCA_transform[:,0], PCA_transform[:,1], c=X_m[:,1], marker='o')
#
#
# =============================================================================
