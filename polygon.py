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

def SyntheticRectData(k, n, r): 
    '''
    Samples k points from n non-overlapping rectangles, and adds r noise points to the data
    :param k: an integer denoting the number of points to be sampled from each rectangular region
    :param n: an integer denoting the number of non-overlapping rectangles to form
    :param r: an integer denoting the number of noise points to add
    :return: a data frame with X, Y, and Label columns. The label 
             denotes the rectangle the point was sampled from. A label of 0 
             means the point is a noise point
    
    '''
    left_boundary = 0
    bottom_boundary = 0
    top_boundary = 7
    poly_regions=np.zeros((k*n+r, 3))
    for i in range(n):
        right_boundary = (7/n)*(i+1)
        for j in range(k):
            poly_regions[j+k*i][2]=i+1
            poly_regions[j+k*i][0]=random.uniform(left_boundary,right_boundary-.05)
            poly_regions[j+k*i][1]=random.uniform(bottom_boundary,top_boundary)
        left_boundary = right_boundary
    xmin = 0
    xmax = 7
    for i in range(r): #Add noise to data
       if i%2 == 0:
           poly_regions[k*n+i][0]=random.uniform(xmin-2, xmin)
           poly_regions[k*n+i][1]=random.uniform(bottom_boundary-2, bottom_boundary)
           poly_regions[k*n+i][2]=0
       else:
           poly_regions[k*n+i][0]=random.uniform(xmax, xmax+2)
           poly_regions[k*n+i][1]=random.uniform(top_boundary, top_boundary+2)
           poly_regions[k*n+i][2]=0
    poly_data = pd.DataFrame({'X':poly_regions[:,0], 'Y':poly_regions[:,1], #convert matrix into a data frame
                 'Label':poly_regions[:,2]})
    return poly_data       
       
# Example of generating synthetic data
synth = SyntheticRectData(30,3,10)
plt.plot(synth['X'][0:29], synth['Y'][0:29], 'ro', #plot results
         synth['X'][30:59], synth['Y'][30:59], 'bo', 
         synth['X'][60:89], synth['Y'][60:89], 'go',
         synth['X'][90:99], synth['Y'][90:99], 'yo')
plt.show()       

def GenerateRectanglePartition(D,k,n): #not needed for iris data since we know ground truth, but could be useful for other data
    '''
    Samples k points from n non-overlapping rectangles of a data set. 
    :param D: A 2D data frame
    :param k: an integer denoting the number of points to be sampled
    :param n: an integer denoting the number of rectangles to form
    :return: a data frame with X, Y, and Label columns. The label 
             denotes the rectangle the point was sampled from.     
    '''
    left_boundary = 0
    sampled_regions=[]
    poly_regions=np.zeros((k*n, 3))
    for i in range(n):
       right_boundary = D.iloc[:,0].quantile((1+i)/n)
       region = []
       for j in range(len(D.iloc[:,0])): #partition points into rectangular regions
           if left_boundary<D.iloc[j,0]<right_boundary: 
               region.append((D.iloc[j,0],D.iloc[j,1]))
       sampled_regions.append(random.sample(region, k)) #sample points from each rectangle
       left_boundary = right_boundary 
       x,y = zip(*sampled_regions[i])
       for m in range(len(x)): #organize regions into a matrix
           poly_regions[m+k*i][2]=i+1
           poly_regions[m+k*i][0]=x[m]
           poly_regions[m+k*i][1]=y[m]
    poly_data = pd.DataFrame({'X':poly_regions[:,0], 'Y':poly_regions[:,1], #convert matrix into a data frame
                 'Label':poly_regions[:,2]})
    return poly_data

# Example of generating rectangular partitions of iris data
D = iris[['Sepal Length', 'Sepal Width']]
rect_points= GenerateRectanglePartition(D, 30, 3)
plt.plot(rect_points['X'][0:29], rect_points['Y'][0:29], 'ro', #plot results
         rect_points['X'][30:59], rect_points['Y'][30:59], 'bo', 
         rect_points['X'][60:89], rect_points['Y'][60:89], 'go')
plt.show()


def Purity(D, C, n): 
    '''
    Computes purity based from 2D data clustered from 3 regions compared to given partition data
    :param D: A data frame with attributes, X, Y, Labels. The labels are taken to be the ground truth
    :param C: A data frame of the clustered data with attributes, X, Y, Labels
    :param n: number of points in Data set
    :return: The purity of C given that D is the ground truth.
    
    '''
    ground_truth = D.as_matrix()
    Cmat = C.as_matrix()
    
    A = {(ground_truth[i,0],ground_truth[i,1]) for i in range(len(ground_truth[:,1])) if ground_truth[i,2] == 1}
    B = {(ground_truth[i,0],ground_truth[i,1]) for i in range(len(ground_truth[:,1])) if ground_truth[i,2] == 2}
    C = {(ground_truth[i,0],ground_truth[i,1]) for i in range(len(ground_truth[:,1])) if ground_truth[i,2] == 3}
    
    Q = {(Cmat[i,0],Cmat[i,1]) for i in range(len(Cmat[:,1])) if Cmat[i,2] == 1}
    R = {(Cmat[i,0],Cmat[i,1]) for i in range(len(Cmat[:,1])) if Cmat[i,2] == 2}
    S = {(Cmat[i,0],Cmat[i,1]) for i in range(len(Cmat[:,1])) if Cmat[i,2] == 3}
    
    bestAmatch = max(len(A.intersection(Q)), len(A.intersection(R)), len(A.intersection(S)))
    bestBmatch = max(len(B.intersection(Q)), len(B.intersection(R)), len(B.intersection(S)))
    bestCmatch = max(len(C.intersection(Q)), len(C.intersection(R)), len(C.intersection(S)))
    
    return((bestAmatch+bestBmatch+bestCmatch)/(n))
 
# Make ground truth data of Iris Data
D = iris[['Sepal Length', 'Sepal Width', 'Species']]
ground_truth = np.zeros((len(D.iloc[:,0]),3))
ground_truth[:,0] = D.iloc[:,0]
ground_truth[:,1] = D.iloc[:,1]
for i in range(len(D.iloc[:,0])):
    if D.iloc[i,2]=='Iris-setosa':
        ground_truth[i,2] = 1
    elif D.iloc[i,2]=='Iris-versicolor':
        ground_truth[i,2] = 2
    else:
        ground_truth[i,2] = 3
groundtruth = pd.DataFrame({'X':ground_truth[:,0], 'Y':ground_truth[:,1], 
                 'Label':ground_truth[:,2]})       
# Compute purity with Kmeans output, km (need to run k-means clustering function to get km)
Purity(groundtruth, km, 150)
    