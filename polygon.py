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
   bottom_boundary = iris['Sepal Width'].quantile(0)
   top_boundary = iris['Sepal Width'].quantile(1)
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
           poly_regions[m+k*i][2]=i
           poly_regions[m+k*i][0]=x[m]
           poly_regions[m+k*i][1]=y[m]        
   return poly_regions

def sample_rectangles_noise(k,n,r): #sample k points from n rectangles, with r noise points Rectangles are formed by x-coordinates and y-coordinates
   left_boundary = 0
   sampled_regions=[]
   bottom_boundary = iris['Sepal Width'].quantile(0)
   top_boundary = iris['Sepal Width'].quantile(1)
   poly_regions=np.zeros((k*n+r, 3))
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
           poly_regions[m+k*i][2]=i
           poly_regions[m+k*i][0]=x[m]
           poly_regions[m+k*i][1]=y[m]
   xmin = min(iris['Sepal Length'])
   xmax = max(iris['Sepal Length'])
   for i in range(r):
       if i%2 == 0:
           poly_regions[k*n+i][0]=random.uniform(xmin-2, xmin)
           poly_regions[k*n+i][1]=random.uniform(bottom_boundary-2, bottom_boundary)
           poly_regions[k*n+i][2]=3
       else:
           poly_regions[k*n+i][0]=random.uniform(xmax, xmax+2)
           poly_regions[k*n+i][1]=random.uniform(top_boundary, top_boundary+2)
           poly_regions[k*n+i][2]=3
   return poly_regions

# example with plot results
poly_points = sample_rectangles(30, 3)
plt.plot(poly_points[0:29, 0], poly_points[0:29, 1], 'ro', 
         poly_points[30:59, 0], poly_points[30:59, 1], 'bo', 
         poly_points[60:89, 0], poly_points[60:89, 1], 'go' )
plt.show()

# example with noise
poly_points_noise = sample_rectangles_noise(30, 3, 10)
plt.plot(poly_points_noise[0:29, 0], poly_points_noise[0:29, 1], 'ro', 
         poly_points_noise[30:59, 0], poly_points_noise[30:59, 1], 'bo', 
         poly_points_noise[60:89, 0], poly_points_noise[60:89, 1], 'go',
         poly_points_noise[90:99,0], poly_points_noise[90:99,1], 'yo')
plt.show()

#define purity function - this assumes we have 3 clusters. Could generalize this more to take in an arbitrary 
# amount of clusters and applying some sort of permutation function on sets
def purity(D, C, k, r): #input is a dataframe, D and a dataframe with the clustered data, C, sample k points
    ground_truth = sample_rectangles(k,3,r)
    Cmat = C.as_matrix()
    
    A = {(ground_truth[i,0],ground_truth[i,1]) for i in range(k)}
    B = {(ground_truth[i,0],ground_truth[i,1]) for i in range(k,2*k)}
    C = {(ground_truth[i,0],ground_truth[i,1]) for i in range(2*k,3*k)}
    
    Q = {(Cmat[i,0],Cmat[i,1]) for i in range(len(Cmat[:,1])) if Cmat[i,2] == 0}
    R = {(Cmat[i,0],Cmat[i,1]) for i in range(len(Cmat[:,1])) if Cmat[i,2] == 1}
    S = {(Cmat[i,0],Cmat[i,1]) for i in range(len(Cmat[:,1])) if Cmat[i,2] == 2}
    
    bestAmatch = max(len(A.intersection(Q)), len(A.intersection(R)), len(A.intersection(S)))
    bestBmatch = max(len(B.intersection(Q)), len(B.intersection(R)), len(B.intersection(S)))
    bestCmatch = max(len(C.intersection(Q)), len(C.intersection(R)), len(C.intersection(S)))
    
    #Don't think I need what is below but am just commenting it out for now in case I do
    #score=[]
    #score.append(len(A.intersection(Q))+len(B.intersection(R))+len(C.intersection(S)))
    #score.append(len(A.intersection(Q))+len(B.intersection(S))+len(C.intersection(R)))
    #score.append(len(A.intersection(R))+len(B.intersection(Q))+len(C.intersection(S)))
    #score.append(len(A.intersection(R))+len(B.intersection(S))+len(C.intersection(R)))
    #score.append(len(A.intersection(S))+len(B.intersection(Q))+len(C.intersection(R)))
    #score.append(len(A.intersection(S))+len(B.intersection(R))+len(C.intersection(Q)))
    
    return((bestAmatch+bestBmatch+bestCmatch)/(150+r))
    
    