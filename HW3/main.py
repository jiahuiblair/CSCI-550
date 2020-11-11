#main.py file
#imports
import pandas as pd
import numpy as np
import scipy.stats
from math import log2
from random import randrange
from random import seed

#Load dataset
iristemp = pd.read_csv('iris.csv', sep=",", header=None) 
iris=pd.DataFrame({'Sepal Length':iristemp[0], 'Sepal Width':iristemp[1], 
                 'Petal Length':iristemp[2], 'Petal Width':iristemp[3], 
                 'Species':iristemp[4]}).to_numpy()
### Find training and testing data
Numberofpointsintraining=50
indices = np.random.permutation(iris.shape[0])
training_idx, test_idx = indices[:Numberofpointsintraining], indices[Numberofpointsintraining:]
training, test = iris[training_idx,:], iris[test_idx,:]

### subset training and testing data
training_data=training[:,[0,1,2,3]]
labels=training[:,4]
testing=test[:,[0,1,2,3]]


###################################
# Main functions

#Authour Jiahui Ma

# given 2 1-D numpy array, calculate the distance and return
def cal_distance(dist_measure, X, Y):
	dist = 0
	if dist_measure == "euclidean":
		dist = np.sqrt(np.sum(np.square(np.subtract(X,Y)), axis=0))
	return dist


def ranking(training_data, labels, eval_instance):
	dist_ranking = {}
	for i, row in enumerate(training_data):
		cur_dist = cal_distance("euclidean", row, eval_instance)
		dist_ranking[str(i)] = cur_dist

	sorted_dist = sorted(dist_ranking.items(), key=lambda x:x[1])

	# print(sorted_dist)
	return sorted_dist

def knn_classify(training_data, labels, eval_instance, k):

	# ranking distance from eval instance to all training data points
		# sort the data points from closest distance to fartherest distance
	sorted_points = ranking(training_data, labels, eval_instance)

	# save the first k data points
	
	label_count = {}
	for item in sorted_points[:k]:
		cur_label = labels[int(item[0])]
		if cur_label not in label_count.keys():
			# label_count[]
			label_count[cur_label] = 1
		else:
			label_count[cur_label] += 1

	# print(label_count)
	max_count = 0
	max_count_label = 0

	for key in label_count.keys():
		if label_count[key] > max_count:
			max_count = label_count[key]
			max_count_label = key
	# print(max_count_label)
	return max_count_label

def k_nearest(training_data,labels,testing,k):
	Finallabels=pd.DataFrame(data=test, columns=['Sepal Length', 'Sepal Width', 
                 'Petal Length', 'Petal Width', 
                 'Species'])

	newlabel=[]
	for i in testing:
		newlabel.append(knn_classify(training_data,labels,i,k)) 

	Finallabels["New label"]=newlabel
	print(Finallabels)
	return Finallabels



####################
#
# author: Daniel Laden
# @: dthomasladen@gmail.com
#
####################
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
				Dy = D[:index]
				Dn = D[index:]
			except:
				print("This error should not exist, if so that means your dataset's last value has more varience then the rest of the data. I'm unsure how this would happen though, so check the code.")

	return (Dy, Dn, max_Entropy) # we get the value to split on.


# D is the dataset, and Xj is a list of values for an attribute
#This scoring will use Entropy or Information gain
def DTcategorical_score(D, Xj, attribute):
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

	Dy = D[D[attribute] == max_Entropy_value]
	Dn = D[D[attribute] != max_Entropy_value]
	# for x in Xj:
	# 	if x == max_Entropy_value: #match for the split point added it to Dy
	# 		Dy.append(x)
	# 	else: #Not a match for the split point add it to Dn
	# 		Dn.append(x)

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
	newD = []
	if n <= q:
		#label all points in leaf with class C
		for point in D:
			print(D)
			#assign them a leaf node class
		return

	#newD = (Dy, Dn, score)
	best_score = -1
	best_D = None
	for name, col in D.iteritems():
		if col[0].isInt() or col[0].isFloat():
			newD = DTnumerical_score(D, col, name)
		else:
			newD = DTcategorical_score(D, col, name)
		if best_score > newD[2]: #new best score
			best_score = 0 + newD[2]
			best_D = newD

	DecisionTree(best_D[0], q, p) #Dy | points in D that satisfies split point
	DecisionTree(best_D[1], q, p) #Dn | points in D that do not satisfy split point


#"Author Badr Zerktouni"

#K-Fold Cross Validation Split Code
def xvalidation_test_split(dataset, folds=6):
    dataset_split = [] 
    dataset_copy = []+ dataset
    fold_size = int(len(dataset)/folds)
    for i in range(folds):
        fold = list()
        index = 0
        while index < fold_size:
            index = randrange(len(dataset_copy))
            fold = dataset_copy.pop(index)
            index += 1
        dataset_split.append(fold)
    return dataset_split, dataset_copy

#Author Robin Belton

#Implement F-measure
def Fmeasure(D):
    '''
    Computes F measure on clustered Iris Data into 3 clusters compared to true species identification
    :param D: A data frame with attributes, Sepal Length, Sepal Width, Petal Length, Petal Width,
              Species, and New Label, where the New Label is from the cluster identification.
    :return: The F measure of the clustered data (Defined on page 429 of "Data Mining and Machine Learning" by Zaki and Meira)
    '''
    ClassifiedVersicolor = {(D.iloc[i,0],D.iloc[i,1],D.iloc[i,2],D.iloc[i,3]) for i in range(len(D.iloc[:,1])) if D.iloc[i,5] == "Iris-versicolor"}
    ClassifiedVirginica  = {(D.iloc[i,0],D.iloc[i,1],D.iloc[i,2],D.iloc[i,3]) for i in range(len(D.iloc[:,1])) if D.iloc[i,5] == "Iris-virginica"}
    ClassifiedSetosa = {(D.iloc[i,0],D.iloc[i,1],D.iloc[i,2],D.iloc[i,3]) for i in range(len(D.iloc[:,1])) if D.iloc[i,5] == "Iris-setosa"}

    TrueVersicolor = {(D.iloc[i,0],D.iloc[i,1],D.iloc[i,2],D.iloc[i,3]) for i in range(len(D.iloc[:,1])) if D.iloc[i,4] == "Iris-versicolor"}
    TrueVirginica  = {(D.iloc[i,0],D.iloc[i,1],D.iloc[i,2],D.iloc[i,3]) for i in range(len(D.iloc[:,1])) if D.iloc[i,4] == "Iris-virginica"}
    TrueSetosa = {(D.iloc[i,0],D.iloc[i,1],D.iloc[i,2],D.iloc[i,3]) for i in range(len(D.iloc[:,1])) if D.iloc[i,4] == "Iris-setosa"}

    prec1 = (max(len(ClassifiedVersicolor.intersection(TrueVersicolor)), len(ClassifiedVersicolor.intersection(TrueVirginica)),
            len(ClassifiedVersicolor.intersection(TrueSetosa))))/len(ClassifiedVersicolor)
    prec2 = (max(len(ClassifiedVirginica.intersection(TrueVersicolor)), len(ClassifiedVirginica.intersection(TrueVirginica)),
            len(ClassifiedVirginica.intersection(TrueSetosa))))/len(ClassifiedVirginica)
    prec3 = (max(len(ClassifiedSetosa.intersection(TrueVersicolor)), len(ClassifiedSetosa.intersection(TrueVirginica)),
            len(ClassifiedSetosa.intersection(TrueSetosa))))/len(ClassifiedSetosa)

    recall1 = (max(len(ClassifiedVersicolor.intersection(TrueVersicolor)), len(ClassifiedVersicolor.intersection(TrueVirginica)),
            len(ClassifiedVersicolor.intersection(TrueSetosa))))/len(TrueVersicolor)
    recall2 = (max(len(ClassifiedVirginica.intersection(TrueVersicolor)), len(ClassifiedVirginica.intersection(TrueVirginica)),
            len(ClassifiedVirginica.intersection(TrueSetosa))))/len(TrueVirginica)
    recall3 = (max(len(ClassifiedSetosa.intersection(TrueVersicolor)), len(ClassifiedSetosa.intersection(TrueVirginica)),
            len(ClassifiedSetosa.intersection(TrueSetosa))))/len(TrueSetosa)

    F1 = (2*prec1*recall1)/(prec1+recall1)
    F2 = (2*prec2*recall2)/(prec2+recall2)
    F3 = (2*prec3*recall3)/(prec3+recall3)
    F = (F1+F2+F3)/3
    return F


# End of Functions
################################


################################
# Main Code
#K-Fold Cross Validation Split Test
seed(1)
example = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train, test = xvalidation_test_split(example, 6)
print(train)
print(test)


#test on KNN
D = k_nearest(training_data,labels,testing,4) #Using J's KNN function
Fmeasure(D)

#Example output
#[2, 9, 8, 3, 5, 6]
#[1, 4, 7, 10]

# End of Main Code
################################


######################
# Code Resources
#
# https://stackoverflow.com/a/39210767
# https://machinelearningmastery.com/implement-resampling-methods-scratch-python/
# https://scikit-learn.org/stable/modules/cross_validation.html
#
######################