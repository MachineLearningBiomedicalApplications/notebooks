# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.ensemble import RandomForestRegressor # import ML method from scikit learn
import pandas as pd # library for reading dataframes (spreadsheets)
from sklearn.model_selection import train_test_split

# load data
DATAMAT=pd.read_csv('GA-structure-volumes-preterm.csv', header=None, )

# separate out data from labals

DATA = DATAMAT.loc[:,1:] # volumes - we have 86 features and 164 samples
LABELS = DATAMAT[0] # GA - 164 


# create a test and train data set using scikit learn method train_test_split
X_train, X_test, y_train, y_test =train_test_split(DATA,LABELS,test_size=0.2)

model=RandomForestRegressor(n_estimators=100)

model.fit(X_train,y_train)

train_performance=model.score(X_train,y_train)

test_performance=model.score(X_test,y_test)

print('train score is {} test score is {} '.format(train_performance,test_performance))
