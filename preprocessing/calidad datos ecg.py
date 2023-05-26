#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 23:22:58 2023

@author: victorrincon
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

import random


"""
Programa para verificar la calidad de los datos leidos de las bases de datos 
"""



# load dataset
pickleIn = open('data/ready/pickles/CYBHi-290.pickle', 'rb')
#x = pickle.load(pickleIn)
people, y, x = pickle.load(pickleIn)

unique, counts = np.unique(y, return_counts=True)

result = np.column_stack((unique, counts)) 
print (result)


why = []
for i in range(len(y)):
    if y[i] == "PMA":
        why.append(y[i])
        plt.plot(x[i])
        
        plt.title("Signal: " + y[i] + '_' + str(i))
        plt.show()
        
        
print(why)
plt.show()

for i in range(len(x)):
    plt.plot(x[i])
    plt.show()