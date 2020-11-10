#"Author Badr Zerktouni"

#Import Libraries

from random import randrange
from random import seed



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

  
   
#K-Fold Cross Validation Split Test
seed(1)
dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train, test = xvalidation_test_split(dataset, 6)
print(train)
print(test)

#Example output
#[2, 9, 8, 3, 5, 6]
#[1, 4, 7, 10]

# coding references
# https://machinelearningmastery.com/implement-resampling-methods-scratch-python/
# https://scikit-learn.org/stable/modules/cross_validation.html
