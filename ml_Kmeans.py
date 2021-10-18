#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 16:57:17 2018

@author: xuetan
"""

##k nearest neighbor
##ucl dataset
##breast cancer
import numpy as np
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd

df = pd.read_csv('/Users/xuetan/breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'],1,inplace=True)

X =np.array(df.drop(['Class'],1)) 
y = np.array(df['Class'])

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)

###prediction
example_measure = np.array([4,2,1,1,1,2,3,2,3])
###use len(example_measure) to automatically reshape
example_measure = example_measure.reshape(len(example_measure),-1)
prediction = clf.predict(example_measure)
print(prediction)

##########