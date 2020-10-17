import numpy as np
import pandas as pd 
from scipy.spatial import distance
import math
import random
import matplotlib.pyplot as plt

## Iris data 
iristemp = pd.read_csv('iris.csv', sep=",", header=None) 
iris=pd.DataFrame({'Sepal Length':iristemp[0], 'Sepal Width':iristemp[1], 
                 'Petal Length':iristemp[2], 'Petal Width':iristemp[3], 
                 'Species':iristemp[4]})
temp=iris[["Sepal Length","Sepal Width"]]
temp["label"]=np.nan


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
	# for i in C:
	# 	print("new cluster")
	# 	print("the length is" + str(len(i)))
	
	
### apply k-mean algorithm 
k=3
e=2
kmean(temp,k,e)

print(temp)