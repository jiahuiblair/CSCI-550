####################
#
# author: Daniel Laden
# @: dthomasladen@gmail.com
#
####################

import numpy as np 
import pandas as pd
import math
from numpy import genfromtxt
from numpy import linalg as LA

## Iris data 
iristemp = pd.read_csv('iris.csv', sep=",", header=None) 
iris=pd.DataFrame({'Sepal Length':iristemp[0], 'Sepal Width':iristemp[1], 
                 'Petal Length':iristemp[2], 'Petal Width':iristemp[3], 
                 'Species':iristemp[4]})
temp=iris[["Sepal Length","Sepal Width"]]
temp["label"]=np.nan

# for index, row in temp.iterrows():
# 	print(index)
# 	print(math.isnan(row[2]))


def FindPointsInRange(data, p, eps):
	neighbours = []
	for q in data:
		dist = LA.norm(p-q)#NOTE change for dataframe
		if dist <= eps and dist != 0: #if it is below our threshold distance eps and isn't itself add it to the list.
			neighbours.append(q)
	return neighbours

def dbscan(data, eps, minPts):
	Cluster_num = 0
	for index, point in data.iterrows():
		if not math.isnan(point[2]): #point already has a label from explorative steps
			continue
		neighbours = FindPointsInRange(data, point, eps) #find all closest neighbours
		if len(neighbours) < minPts: #if point doesn't have enough neighours label it as noise
			point[2] = -1 #noise
			continue
		Cluster_num += 1
		point[2] = Cluster_num
		set_of_neighbours = neighbours
		for n_point in neighbours:
			if n_point[2] == -1: #point isn't a noise point, it's part of a cluster
				n_point[2] = Cluster_num
			if point == 0: #point already has a label from explorative steps
				continue
			n_point[2] = Cluster_num
			neighbours = FindPointsInRange(data, n_point, eps)
			if len(neighbours) >= minPts:
				set_of_neighbours = list(set().union(set_of_neighbours, neighbours)) #add any additional points that meet the distance requirement








###################################
# Code References
#
# https://stackoverflow.com/a/3519314
# https://stackoverflow.com/a/20894023
# https://stackoverflow.com/a/1401828
# https://stackoverflow.com/a/944733
# https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf?source=post_page---------------------------
# https://en.wikipedia.org/wiki/DBSCAN
#
####################################