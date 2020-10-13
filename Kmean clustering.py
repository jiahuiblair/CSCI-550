import numpy as np
import pandas as pd 
from scipy.spatial import distance
import math
import random
import matplotlib.pyplot as plt
## Iris data 
Iris = pd.read_table("iris.data")
iristemp = Iris["col"].str.split(",")
data = iristemp.to_list()
names =['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width','Species']
iris = pd.DataFrame(data,columns=names)
irisclu = iris[['Petal Length','Petal Width']] # clusering on Petal Length and Sepal Length
iris=[]

for i in range(len(irisclu['Petal Length'])):
	clu=[]
	a = irisclu.iloc[i,:]
	clu.append(float(a.iloc[0]))
	clu.append(float(a.iloc[1]))
	iris.append(clu)

######### K-mean clustering algorithm

def centroiddistance(T,a,b):
		d=[]
		for i in range(0,k):
			d.append(np.sqrt(abs(T[a][i][0]-T[b][i][0])+abs(T[a][i][1]-T[b][i][1])))
		return sum(d)

def kmean(D,k,e):
	### initialize the first centroids
	opos=[]
	pos=[]
	for i in D:
		opos.append(i[0])
		pos.append(i[1])
	maxi0=max(opos)
	maxi1=max(pos)
	mini0=min(opos)
	mini1=min(pos)
	
	T={}
	t=0
	centroids=[]
	for i in range(k):
		centroids.append([random.uniform(mini0,maxi0),random.uniform(mini1,maxi1)])
	T[0]=centroids

	### Assign points to a cluster
	distance=100
	while distance>e:
		temp = iris
		C=[]
		tminus1=t
		t=t+1
		for i in range(0,k): 
				i=[] 
				C.append(i)

		for j in range(len(temp)):
			d=[]
			for i in range(len(T[tminus1])):
				d.append(math.sqrt(abs(T[tminus1][i][0]-temp[j][0])+abs(T[tminus1][i][1]-temp[j][1])))
			minarg = min(d)
			for i in range(len(d)):
				if minarg==d[i]:
					temp[j].insert(2,i)
		for i in range(k):
			for j in range(len(temp)):
				if temp[j][2]==i:
					C[i].append(temp[j])

		### Centroid update
		newcentroid=[]
		for i in range(len(C)):
			x=[]
			y=[]
			for j in range(len(C[i])):
				x.append(C[i][j][0])
				y.append(C[i][j][1])
			xcenter=sum(x)/len(C[i])
			ycenter=sum(y)/len(C[i])
			r = [xcenter,ycenter]
			newcentroid.append(r)
		T[t]=newcentroid
		distance=centroiddistance(T,t, tminus1)
	
	### Final clustermatrix
	clustermatrix=[]
	for i in range(0,k): 
				i=[] 
				clustermatrix.append(i)

	print("[x,y,the final clustering label, all previous labels]")
	for i in range(len(C)):
		print("The " + str(i+1) +" clustering list")
		print(C[i])

### apply k-mean algorithm 
k=3
e=2
kmean(iris,k,e)


