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

def sample_rectangles(k,n): #sample k points from n rectangles. Rectangles are formed by x-coordinates
   left_boundary = 0
   sampled_regions=[]
   for i in range(n):
       right_boundary = iris['Sepal Length'].quantile((1+i)/n)
       region = []
       for j in range(150):
           if left_boundary<iris['Sepal Length'][j]<right_boundary:
               region.append((iris['Sepal Length'][j],iris['Sepal Width'][j]))
       sampled_regions.append(random.sample(region, k))
       left_boundary = right_boundary 
   return sampled_regions


# plot results
sampled_points = sample_rectangles(15, 3)
x1,y1 = zip(*sampled_points[0])
x2,y2 = zip(*sampled_points[1])
x3,y3 = zip(*sampled_points[2])
plt.plot(x1, y1, 'ro', x2, y2, 'bo', x3, y3, 'go')
plt.show()