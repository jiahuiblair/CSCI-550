import numpy as np
import pandas as pd 
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

k_nearest(training_data,labels,testing,4)

