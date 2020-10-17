#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:24:59 2020

@author: robin
"""
import pandas as pd 
import random
import matplotlib.pyplot as plt
import numpy as np


iristemp = pd.read_csv('iris.csv', sep=",", header=None) 
iris=pd.DataFrame({'Sepal Length':iristemp[0], 'Sepal Width':iristemp[1], 
                 'Petal Length':iristemp[2], 'Petal Width':iristemp[3], 
                 'Species':iristemp[4]})

def sample_rectangles(k,n): #sample k points from n rectangles. Rectangles are formed by x-coordinates and y-coordinates
   left_boundary = 0
   sampled_regions=[]
   bottom_boundary = iris['Sepal Width'].quantile(.15)
   top_boundary = iris['Sepal Width'].quantile(.85)
   poly_regions=np.zeros((k*n, 3))
   for i in range(n):
       right_boundary = iris['Sepal Length'].quantile((1+i)/n)
       region = []
       for j in range(150):
           if left_boundary<iris['Sepal Length'][j]<right_boundary and bottom_boundary<iris['Sepal Width'][j]<top_boundary:
               region.append((iris['Sepal Length'][j],iris['Sepal Width'][j]))
       sampled_regions.append(random.sample(region, k))
       left_boundary = right_boundary 
       x,y = zip(*sampled_regions[i])
       for m in range(len(x)): #organize regions into an array
           poly_regions[m+k*i][0]=i+1
           poly_regions[m+k*i][1]=x[m]
           poly_regions[m+k*i][2]=y[m]        
   return poly_regions


# plot results
poly_points = sample_rectangles(15, 3)
plt.plot(poly_points[0:14, 1], poly_points[0:14, 2], 'ro', 
         poly_points[15:29, 1], poly_points[15:29, 2], 'bo', 
         poly_points[30:44, 1], poly_points[30:44, 2], 'go' )
plt.show()