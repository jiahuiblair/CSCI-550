#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:23:49 2020

@author: zerktounibadr
"""
import numpy as np
import math
import matplotlib.pyplot as plt



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

for column in data:
    b_var = variance(data)
    print("b_var = {}".format(b_var))
    b_var=plt.b_var()
    
   