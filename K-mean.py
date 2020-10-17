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
	return(D)
	
	# for i in C:
	# 	print("new cluster")
	# 	print("the length is" + str(len(i)))


### Internal assessment
def internalassessment(D,k):
	internalkmean=D.copy()
	######## create a cluster list to store points in each cluster
	cluster=[]
	for i in range(0,k): 
		i=[] 
		cluster.append(i)

	for i in range(0,k):
		for j in range(len(internalkmean)):
			if internalkmean.iloc[j,2]==i:
				cluster[i].append(internalkmean.iloc[j,:])

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

### apply k-mean algorithm 
k=3
e=0.05
km=temp.copy()
kmean(km,k,e)
internalassessment(km,k)