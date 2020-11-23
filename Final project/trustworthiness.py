import numpy as np
import pandas as pd 
from scipy.spatial import distance
import math
import random
import matplotlib.pyplot as plt
from datetime import datetime

datapca = pd.read_csv('trustworthinesspca.csv') 

k_range=[20,30,40,50]


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

	#print(sorted_dist)
	return sorted_dist 
def label(label,list):
	L=0
	for i in range(len(list)):
		if label==list[i][0]:
			L=i+1
	return L

M1pca=[]
M2pca=[]
M1mds=[]
M2mds=[]
for k in k_range:
	or_in_pca=[]
	pca_in_or=[]
	or_in_mds=[]
	mds_in_or=[]
	for i in range(len(datapca)):
		point=datapca.iloc[i] 
		eval_instance_orig=point[["Mental","Physical","Temporal","Task Complexity","Stress","Distractions","Performance","Effort","Frustration"]].to_numpy()
		eval_instance_pca=point[["Component1","Component2"]].to_numpy()
		eval_instance_mds=point[["mds1","mds2"]].to_numpy()
		label_p=point[["Label"]].to_numpy()

		sub_data=datapca.drop(i)
		original=sub_data[["Mental","Physical","Temporal","Task Complexity","Stress","Distractions","Performance","Effort","Frustration"]].to_numpy()
		pca2_d= sub_data[["Component1","Component2"]].to_numpy()
		mds2_d=sub_data[["mds1","mds2"]].to_numpy()
		labels=sub_data[["Label"]].to_numpy()

		sorted_original=ranking(original,labels,eval_instance_orig)
		sorted_pca=ranking(pca2_d,labels,eval_instance_pca)
		sorted_mds=ranking(mds2_d,labels,eval_instance_mds)

		k_original = sorted_original[0:k]
		k_pca=sorted_pca[0:k]
		k_mds=sorted_mds[0:k]
		
		
		orset=[]
		pcaset=[]
		mdsset=[]
		orsum=0
		pcasum=0
		mdssum=0
		orsummds=0

		for item in k_original:
			orset.append(item[0])

		for item in k_pca:
			pcaset.append(item[0])

		for item in k_mds:
			mdsset.append(item[0])
		if i==0:
			print("#######original######")
			print(orset)
			print("########pca#########")
			print(pcaset)
			print("########mds######")
			print(mdsset)
#######pca
		# print("######################pca1########################")
		for i in pcaset:
			if i not in orset:
				temp=label(i,sorted_original)
				# print(temp)
				pcasum+=(temp-k)
		pca_in_or.append(pcasum)
		# print("#######################pca2#######################")
		for i in orset:
			if i not in pcaset:
				temp=label(i,sorted_pca)
				# print(temp)
				orsum+=(temp-k)
		or_in_pca.append(orsum)
		
########mds
		# print("######################MDS1########################")
		for i in mdsset:
			if i not in orset:
				temp=label(i,sorted_original)
				# print(temp)
				mdssum+=(temp-k)
		mds_in_or.append(mdssum)
		# print("######################MDS2########################")
		for i in orset:
			if i not in mdsset:
				temp=label(i,sorted_mds)
				# print(temp)
				orsummds+=(temp-k)
		or_in_mds.append(orsummds)

		

	m2pca=1-2/(200*k*(2*200-3*k-1))*sum(or_in_pca)
	m1pca=1-2/(200*k*(2*200-3*k-1))*sum(pca_in_or)

	m2mds=1-2/(200*k*(2*200-3*k-1))*sum(or_in_mds)
	m1mds=1-2/(200*k*(2*200-3*k-1))*sum(mds_in_or)
	M1pca.append(m1pca)
	M2pca.append(m2pca)
	M1mds.append(m1mds)
	M2mds.append(m2mds)

print(M1pca)

print(M2pca)

print(M1mds)

print(M2mds)

k=np.array(k_range)
M1plotpca=np.array(M1pca)
M2plotpca=np.array(M2pca)
M1plotmds=np.array(M1mds)
M2plotmds=np.array(M2mds)
plt.plot(k, M1plotpca, 'r--', k, M2plotpca, 'bs')
plt.show()

plt.plot(k, M1plotmds, 'r--', k, M2plotmds, 'bs')
plt.show()


