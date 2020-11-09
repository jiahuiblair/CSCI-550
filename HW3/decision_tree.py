####################
#
# author: Daniel Laden
# @: dthomasladen@gmail.com
#
####################

#imports
import pandas as pd
import numpy as np
import scipy.stats
from math import log2

#Load dataset
iristemp = pd.read_csv('iris.csv', sep=",", header=None) 
iris=pd.DataFrame({'Sepal Length':iristemp[0], 'Sepal Width':iristemp[1], 
                 'Petal Length':iristemp[2], 'Petal Width':iristemp[3], 
                 'Species':iristemp[4]}).to_numpy()

# D is the dataset, and Xj is a list of values for an attribute
def DTnumerical_score(D, Xj, attribute):
	n = len(Xj)
	D.sort_values(by=attribute)#attribute of Xj
	Xj.sort()

	M = [] #midpoints of Xj
	index = 1
	for x in Xj[:-1]: #Xj needs to be ordered by value too
		if x != Xj[index]: #not the same value
			midpoint = x + ((Xj[index]-x)/2)#calculate midpoint
			M.append(midpoint)
		index +=1

	Dy = None
	Dn = None
	max_Entropy = -1
	index = 1
	for point in M:

		data_in = lessThan(Xj, point)#all the data on the midpoint or left of it
		left = data_in/n #the amount our categorical value covers the dataset
		right = (n-data_in)/n #the inverse of that
		entropy = -(left * log2(left) + right * log2(right)) #calculate entropy
		if entropy < max_Entropy or max_Entropy == -1: #better value for splitting found
			max_Entropy = 0 + entropy
			try:
				Dy = Xj[:index]
				Dn = Xj[index:]
			except:
				print("This error should not exist, if so that means your dataset's last value has more varience then the rest of the data. I'm unsure how this would happen though, so check the code.")

	return (Dy, Dn, max_Entropy) # we get the value to split on.


# D is the dataset, and Xj is a list of values for an attribute
#This scoring will use Entropy or Information gain
def DTcategorical_score(D, Xj):
	n = len(Xj)
	unique, counts = numpy.unique(a, return_counts=True)
	unique_categories = dict(zip(unique, counts))

	max_Entropy = -1
	max_Entropy_value = None
	for uni in unique_categories:
		num_in = unique_categories[uni]/n #the amount our categorical value covers the dataset
		num_out = (n-unique_categories[uni])/n #the inverse of that
		entropy = -(num_in * log2(num_in) + num_out * log2(num_out)) #calculate entropy
		if entropy < max_Entropy or max_Entropy == -1: #better value for splitting found
			max_Entropy = 0 + entropy
			max_Entropy_value = uni

	Dy = None
	Dn = None
	for x in Xj:
		if x == max_Entropy_value: #match for the split point added it to Dy
			Dy.append(x)
		else: #Not a match for the split point add it to Dn
			Dn.append(x)

	return (Dy, Dn, max_Entropy)# we get the value to split on.

def lessThan(Xj, midpoint):
	n = 0
	for x in Xj:
		if x <= midpoint:
			n += 1
	return n

#D is the dataframe, q is the smallest group size threshold, p is the purity score
def DecisionTree(D, q, p):
	n = len(D)
	for 
	newD = []
	if n <= q:
		#label all points in leaf with class C
		for point in D:
			#assign them a leaf node class
		return

	#newD = (Dy, Dn, score)
	best_score = -1
	best_D = None
	for name, col in D.iteritems():
		if col[0].isInt() or col[0].isFloat():
			newD = DTnumerical_score(D, col, name)
		else:
			newD = DTcategorical_score(D, col)
		if best_score > newD[2]: #new best score
			best_score = 0 + newD[2]
			best_D = newD

	DecisionTree(best_D[0], q, p) #Dy | points in D that satisfies split point
	DecisionTree(best_D[1], q, p) #Dn | points in D that do not satisfy split point




######################
# Code Resources
#
# https://stackoverflow.com/a/39210767
#
######################
