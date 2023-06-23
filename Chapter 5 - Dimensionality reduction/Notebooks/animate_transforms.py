#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:30:08 2018

Functions to animate linear transformations, taken from
https://dododas.github.io/linear-algebra-with-python/posts/16-12-29-2d-transformations.html
and https://notgnoshi.github.io/linear-transformations/

@author: Emma C. Robinson
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc


# Function to compute intermediate transforms to be plotted during animation
def stepwise_transform(a, points, nsteps=30):
    '''
    Generate a series of intermediate transform for the matrix multiplication
      np.dot(a, points) # matrix multiplication
    starting with the identity matrix, where
      a: 2-by-2 matrix
      points: 2-by-n array of coordinates in x-y space 

    Returns a (nsteps + 1)-by-2-by-n array
    '''
    # create empty array of the right size
    transgrid = np.zeros((nsteps+1,) + np.shape(points))
    # compute intermediate transforms
    for j in range(nsteps+1):
        intermediate = np.eye(2) + j/float(nsteps)*(a - np.eye(2)) 
        transgrid[j] = np.dot(intermediate, points) # apply intermediate matrix transformation
        
    return transgrid


def animate_transform(A, grid=None, num_steps=50, repeat=False):
    """
        Animates the effect a transform has on a given grid.
        If no grid is given, one will be generated.

        You can expect a small delay while the function generates the animation.
    """

    if grid is None:
        # Create a grid of points in x-y space
        xvals = np.linspace(-3, 3, 18)
        yvals = np.linspace(-3, 3, 14)
        grid = np.column_stack([[x, y] for x in xvals for y in yvals])

    intermediate_transforms = stepwise_transform(A, grid, num_steps)
    fig = plt.figure(figsize=(6, 6))

    grid_range_min=min(min(min(grid[0])-2, min(intermediate_transforms[-1][0])-2),\
                       min(min(grid[1])-2, min(intermediate_transforms[-1][1])-2) )
    
    grid_range_max=max(max(max(grid[0])+2, max(intermediate_transforms[-1][0])+2),\
                       max(max(grid[1])+2, max(intermediate_transforms[-1][1])+2))
    
    xmin = grid_range_min
    xmax = grid_range_max
    ymin = grid_range_min
    ymax = grid_range_max

    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
    plt.grid(True)
    scatter = ax.scatter([], [], c='r')
    # Prevent `%matplotlib inline` from displaying the above scatter plot.
    plt.close()

    def update(i):
        """Draws the ith intermediate transform"""
        scatter.set_offsets(intermediate_transforms[i].T)
        return scatter,

    return animation.FuncAnimation(fig,
                                   update,
                                   interval=50,
                                   frames=num_steps,
                                   blit=True,
                                   repeat=repeat)
    
    
# =============================================================================
# A = np.column_stack([[2, 0], [0, 1]])
# anim = animate_transform(A, repeat=True)
# #anim.save('/Users/emma/Documents/TEACHING/LECTURES/Machine Learning/code/LinearManifoldLearning/shear2.html')
# anim
# =============================================================================

