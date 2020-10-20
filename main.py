#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
from scipy.spatial import distance
import math
import random
import matplotlib.pyplot as plt
from numpy import linalg as LA
from datetime import datetime
import sys


# random.seed(datetime.now())

## Iris data 
iristemp = pd.read_csv('iris.csv', sep=",", header=None) 
iris=pd.DataFrame({'Sepal Length':iristemp[0], 'Sepal Width':iristemp[1], 
                 'Petal Length':iristemp[2], 'Petal Width':iristemp[3], 
                 'Species':iristemp[4]})
temp=iris[["Sepal Length","Sepal Width"]]
temp["label"]=np.nan

###########################################################################
# Main functions

######### K-mean clustering algorithm

def centroiddistance(T,a,b):
	d=[]
	for i in range(0,k):
		d.append(np.sqrt(abs(T[a][i][0]-T[b][i][0])+abs(T[a][i][1]-T[b][i][1])))
	return sum(d)

def kmean(D,k,e):
	### initialize the first centroids
	
	maxi0=max(D.iloc[:,0])
	maxi1=max(D.iloc[:,1])
	mini0=min(D.iloc[:,0])
	mini1=min(D.iloc[:,1])
	
	T={}
	t=0
	centroids=[]
	for i in range(k):
		centroids.append([random.uniform(mini0,maxi0),random.uniform(mini1,maxi1)])
	T[0]=centroids

	### Assign points to a cluster
	distance=100
	while distance>e:
		C=[]
		tminus1=t
		t=t+1
		for i in range(0,k): 
				i=[] 
				C.append(i)

		for j in range(len(D)):
			d=[]
			for i in range(len(T[tminus1])):
				d.append(math.sqrt(abs(T[tminus1][i][0]-D.iloc[j,0])+abs(T[tminus1][i][1]-D.iloc[j,1])))
			minarg = min(d)
			for i in range(len(d)):
				if minarg==d[i]:
					D.iloc[j,2]=i
		for i in range(k):
			for j in range(len(D)):
				if D.iloc[j,2]==i:
					C[i].append(D.iloc[j,:])

		### Centroid update
		newcentroid=[]
		for i in range(len(C)):
			x=[]
			y=[]
			for j in range(len(C[i])):
				x.append(C[i][j].iloc[0])
				y.append(C[i][j].iloc[1])
			xcenter=sum(x)/len(C[i])
			ycenter=sum(y)/len(C[i])
			r = [xcenter,ycenter]
			newcentroid.append(r)
		T[t]=newcentroid
		distance=centroiddistance(T,t, tminus1)
	#### convert the labels to start from 1
	for i in range(len(D)):
		D.iloc[i,2]=D.iloc[i,2].copy() +1
	print(D)

	
	# for i in C:
	# 	print("new cluster")
	# 	print("the length is" + str(len(i)))

### Internal assessment
def internalassessment(D,k):
	internal=D.copy()
	######## create a cluster list to store points in each cluster
	cluster=[]
	for i in range(0,k): 
		i=[] 
		cluster.append(i)

	for i in range(0,k):
		for j in range(len(internal)):
			if internal.iloc[j,2]==i+1:
				cluster[i].append(internal.iloc[j,:])

	######## mean distance to all points in closest cluster: Uout
	Uout=[]
	currentclu=[]
	for k in range(len(cluster)):
		clu=cluster.copy()
		currentclu=clu.pop(k) ## length is the number of points
		subclu=clu ## length is 2
		for i in currentclu:
			Dout=[]
			for j in subclu:
				Doutsub=[]
				for y in j:
					t=math.sqrt(abs(i.iloc[0]-y.iloc[0])+abs(i.iloc[1]-y.iloc[1]))
					Doutsub.append(t)
				average= sum(Doutsub)/len(Doutsub)
				Dout.append(average)
			i["Uout"]=min(Dout)
		Uout.append(currentclu)

	######## mean distance to all points in the cluster: Uin
	for k in Uout:
		for i in range(len(k)):
			clus=k.copy()
			currenti=clus.pop(i)
			subclus=clus
			Dinsub=[]
			for j in subclus:
				w=math.sqrt(abs(currenti.iloc[0]-j.iloc[0])+abs(currenti.iloc[1]-j.iloc[1]))
				Dinsub.append(w)
			average=sum(Dinsub)/len(Dinsub)
			k[i]["Uin"]=average

	######## Silhouette coeff for a point
	for k in Uout:
		for i in k:
			Si=(i.iloc[3]-i.iloc[4])/(max(i.iloc[3],i.iloc[4]))
			i["Si"]=Si
	sc=[]
	for k in Uout:
		for i in k:
			sc.append(i.iloc[5])
	SC=sum(sc)/len(sc)
	print("The Silhouette Coeff is ")
	print(SC)

"""
Created on Fri Oct 16 14:24:59 2020
@author: robin
"""

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

def Purity(D, C): 
    '''
    Computes purity based from 2D data clustered from 3 regions compared to given partition data
    :param D: A data frame with attributes, X, Y, Labels. The labels are taken to be the ground truth
    :param C: A data frame of the clustered data with attributes, X, Y, Labels
    :param n: number of points in Data set
    :return: The purity of C given that D is the ground truth.
    
    '''
    
    A = {(D.iloc[i,0],D.iloc[i,1]) for i in range(len(D.iloc[:,1])) if D.iloc[i,2] == 1}
    B = {(D.iloc[i,0],D.iloc[i,1]) for i in range(len(D.iloc[:,1])) if D.iloc[i,2] == 2}
    E = {(D.iloc[i,0],D.iloc[i,1]) for i in range(len(D.iloc[:,1])) if D.iloc[i,2] == 3}
    
    Q = {(C.iloc[i,0],C.iloc[i,1]) for i in range(len(C.iloc[:,1])) if C.iloc[i,2] == 1}
    R = {(C.iloc[i,0],C.iloc[i,1]) for i in range(len(C.iloc[:,1])) if C.iloc[i,2] == 2}
    S = {(C.iloc[i,0],C.iloc[i,1]) for i in range(len(C.iloc[:,1])) if C.iloc[i,2] == 3}
    
    bestAmatch = max(len(A.intersection(Q)), len(A.intersection(R)), len(A.intersection(S)))
    bestBmatch = max(len(B.intersection(Q)), len(B.intersection(R)), len(B.intersection(S)))
    bestEmatch = max(len(E.intersection(Q)), len(E.intersection(R)), len(E.intersection(S)))
    
    return((bestAmatch+bestBmatch+bestEmatch)/(len(D.iloc[:,0])))


####################
#
# author: Daniel Laden
# @: dthomasladen@gmail.com
#
####################

def FindPointsInRange(data, p, eps):
	neighbours = []
	attri_len = len(p)-1
	#print(attri_len)
	for index, q in data.iterrows():
		dist = LA.norm(p[:attri_len]-q[:attri_len])#NOTE change for dataframe
		# print(dist)
		if dist <= eps and dist != 0: #if it is below our threshold distance eps and isn't itself add it to the list.
			neighbours.append(q)
	# 		print("Here")
	# print(len(neighbours))
	# print(neighbours)
	return neighbours

def dbscan(data, eps, minPts):
	Cluster_num = 0
	for index, point in data.iterrows():
		if not math.isnan(point[2]): #point already has a label from explorative steps
			#print("point")
			continue
		neighbours = FindPointsInRange(data, point, eps) #find all closest neighbours
		if len(neighbours) < minPts: #if point doesn't have enough neighours label it as noise
			point[2] = -1 #noise
			continue
		Cluster_num += 1
		point[2] = Cluster_num
		set_of_neighbours = neighbours
		#print(len(set_of_neighbours))
		for n_point in set_of_neighbours:
			if n_point[2] == -1: #point isn't a noise point, it's part of a cluster
				n_point[2] = Cluster_num
			if not math.isnan(n_point[2]): #point already has a label from explorative steps
				continue
			n_point[2] = Cluster_num
			neighbours = FindPointsInRange(data, n_point, eps)
			if len(neighbours) >= minPts:
				set_of_neighbours = set_of_neighbours+neighbours #This might be the union of the two but I'm unsure, it seems to work but this might cause larger sets to take forever
			#print(len(set_of_neighbours))
	return data

# End of Functions
###########################################################################


###########################################################################
# Start of executable 


if len(sys.argv) > 1: #meaning the user ran it with additional arguments
	if sys.argv[1] == "kmean":
		### apply k-mean algorithm 
		try:
			k=int(sys.argv[2])
			e=float(sys.argv[3])
			km=temp.copy()
			kmean(km,k,e)
			internalassessment(km,k)
		except:
			print("Not enough arguments please try main.py kmean 3 0.00001 as a test")
	elif sys.argv[1] == "dbscan":
		#DBSCAN
		try:
			#			db, e value, minpts
			e=float(sys.argv[2])
			minPts=int(sys.argv[3])
			sn = temp.copy()
			db = dbscan(sn, e, minPts)
			print(db)
		except:
			print("Not enough arguments please try main.py dbscan 0.3 10 as a test")
	else:
		print("Improper input: try main.py kmean 3 0.00001 or main.py dbscan 0.3 10")

else: #combined showings of our test code
	### apply k-mean algorithm 
	k=3
	e=0.00001
	km=temp.copy()
	kmean(km,k,e)
	internalassessment(km,k)


	# Example of generating synthetic data
	synth = SyntheticRectData(30,3,10)
	plt.plot(synth['X'][0:29], synth['Y'][0:29], 'ro', #plot results
	         synth['X'][30:59], synth['Y'][30:59], 'bo', 
	         synth['X'][60:89], synth['Y'][60:89], 'go',
	         synth['X'][90:99], synth['Y'][90:99], 'yo')
	plt.savefig('synth.png')  
	plt.show()     

	# Example of generating rectangular partitions of iris data
	D = iris[['Sepal Length', 'Sepal Width']]
	rect_points= GenerateRectanglePartition(D, 30, 3)
	plt.plot(rect_points['X'][0:29], rect_points['Y'][0:29], 'ro', #plot results
	         rect_points['X'][30:59], rect_points['Y'][30:59], 'bo', 
	         rect_points['X'][60:89], rect_points['Y'][60:89], 'go')
	plt.show()

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
	Purity(groundtruth, km)



	#DBSCAN
	#			db, e value, minpts
	sn = temp.copy()
	db = dbscan(sn, 0.3, 15)

	print(db)


# End of executable code
###########################################################################

###################################
# Code References for DBSCAN
#
# https://stackoverflow.com/a/3519314
# https://stackoverflow.com/a/20894023
# https://stackoverflow.com/a/1401828
# https://stackoverflow.com/a/944733
# https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf?source=post_page---------------------------
# https://en.wikipedia.org/wiki/DBSCAN
#
####################################