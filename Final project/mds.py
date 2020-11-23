#####################
#
#author: Daniel Laden
# @dthomasladen@gmail.com
#
#####################

#imports
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Load dataset
temp = pd.read_csv('trustworthinesspca.csv', sep=",", header=None) 
scores=pd.DataFrame({'Mental':temp[6], 'Temporal':temp[7], 
                 'Task Complexity':temp[8], 'Stress':temp[9], 
                 'Distractions':temp[10], 'Performance':temp[11],
                 'Performance':temp[12], 'Effort':temp[13], 'Frustration':temp[14]}).to_numpy()


##############################
# Functions

def mean(data):
    count = len(data)
    total = sum(data)
    return total / count
def square_deviation(data):
    c = mean(data)
    count = len(data)
    total = sum((x-c)**2 for x in data)
    total2 = sum((x-c) for x in data)
    
    total -= total2**2/count
    assert not total < 0 , 'negative sum of square deviation: %f' % total
    return total
def variance(data):
    n = len(data)
    if n < 2:
        raise Exception('variance requires at least two data points')
    ss = square_deviation(data)
    return ss/(n-1)

def printVariance(data):
	for column in range(data.shape[1]):
		dataCol = data[:,column]
		b_var = variance(dataCol)
		print("b_var = {}".format(b_var))
		#b_var=plt.b_var()


# End of Functions
##############################

##############################
# Main Code

dataset = scores.copy()

# print(dataset)
dataset = np.delete(dataset, 0, 0) #remove row of column names
# print(dataset)


mds_embedding_2 = MDS(n_components=2, metric=False)
mds_embedding_3 = MDS(n_components=3, metric=False)
mds_embedding_5 = MDS(n_components=5, metric=False)
mds_embedding_8 = MDS(n_components=8, metric=False)

dataset_normalized = StandardScaler().fit(dataset).transform(dataset)
# dataset_normalized_T=dataset_normalized.T


db_transform_2 = mds_embedding_2.fit_transform(dataset_normalized)
db_transform_3 = mds_embedding_3.fit_transform(dataset_normalized)
db_transform_5 = mds_embedding_5.fit_transform(dataset_normalized)
db_transform_8 = mds_embedding_8.fit_transform(dataset_normalized)


# plt.scatter(db_transform_8[:,0],db_transform_8[:,1], c='r', s=15)
# plt.scatter(db_transform_8[:,0],db_transform_8[:,2], c='b', s=15)


# plt.show()


print(db_transform_2.shape)

#np.savetxt("mds8.csv", db_transform_8, delimiter=",", fmt="%f")
np.savetxt("mds2.csv", db_transform_2, delimiter=",", fmt="%f")


print(type(db_transform_2))

# printVariance(db_transform_2)
# printVariance(db_transform_3)
# printVariance(db_transform_5)
# printVariance(db_transform_8)

print(np.var(db_transform_2))
print(np.var(db_transform_3))
print(np.var(db_transform_5))
print(np.var(db_transform_8))




#########################
# Ref code
#
# https://stackoverflow.com/questions/42689003/save-numpy-array-to-csv-without-scientific-notation
# https://scikit-learn.org/0.15/auto_examples/manifold/plot_mds.html#example-manifold-plot-mds-py
#
#########################