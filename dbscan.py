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
# 	print(type(row[1]))
# 	print(math.isnan(row[2]))


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

			#db, e value, minpts
db = dbscan(temp, 0.3, 10)

print(db)


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
