#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 13:38:29 2018

Applying Random Forests to the GA regression problem

@author: Emma Robinson
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


# load data

DATAMAT=pd.read_csv('GA-structure-volumes-preterm.csv',header=None)

# separate out data from labals

DATA = DATAMAT.loc[:,1:] # volumes - we have 86 features and 164 samples
LABELS = DATAMAT[0] # GA - 164 

# split data into test and train
X_train, X_test, y_train, y_test = train_test_split(DATA, LABELS, test_size=.4, random_state=42)

# get baseline prediction from decision tree (no param optimisation)

clf=DecisionTreeRegressor()
clf.fit(X_train, y_train)
score_DT = clf.score(X_test, y_test)
print('Decision Tree Score', score_DT)

# try random forest (no param optimisation)

clf=RandomForestRegressor(random_state=42)
clf.fit(X_train, y_train)
score_RF1 = clf.score(X_test, y_test)
print('Random Forest initial Score', score_RF1)


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, 5, 10, 20, 50],
              "max_features": np.linspace(10,DATA.shape[1],5).astype(int),
              "n_estimators": [10,20,50,100]
              }

grid = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_dist)
grid.fit(DATA, LABELS)

# summarize the results of the grid search
print('Best classification score achieved using grid search:', grid.best_score_)
print('The parameters resulting in the best score are depth: {},max_f {} and n_estimators {} '.format(
        grid.best_estimator_.max_depth,grid.best_estimator_.max_features,grid.best_estimator_.n_estimators))
      

