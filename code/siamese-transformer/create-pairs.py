#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 03:26:25 2023

@author: victor-rincon
"""

import utils
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt 
import time
from datetime import datetime
import random
path = 'data/ready/pickles/CYBHi-1500.pickle'

y, x, people = utils.load_data(path)
y, x = utils.shuffle(y, x)



x = np.array(x)

print('Before Normalization\n')
print('Shape:', x.shape)
print('Min:', x.min(), 'Max:', x.max())
print(x.dtype)

print('\nAfter Normalization\n')

x = (x - x.min()) / (x.max() - x.min())
y = np.array(y)

print('Shape:', x.shape)
print('Min:', x.min(), 'Max:', x.max())
print(x.dtype)



# shuffle data
length = len(y)
data = []
for i in range(length):
  data.append([y[i], x[i]])

print(y)
num = random.randint(0, length)
random.seed(num)
random.shuffle(data)

y, x = [], []
for k in range(length):
  y.append(data[k][0])
  x.append(data[k][1])

data, k, length, num = [], 0, 0, 0 # just for memory management

x = np.array(x)
y = np.array(y)

print(y)
print(len(y))
print(x.shape)




from tqdm.auto import tqdm

def create_pairs(x, y):
  yy, xx = [], []

  indices = [np.where(y == i)[0] for i in people]
  dic = {people[j]: indices[j] for j in range(len(people))}
  for i in tqdm(range(len(x))):
    current_image = x[i]
    label = y[i]
    ia = np.random.choice(dic[label], replace=False)
    positive_image = x[ia]
    xx.append([current_image, positive_image])
    yy.append(1)
    
    choices = np.where(y != label)[0]
    ib = np.random.choice(choices, replace=False)
    negative_image = x[ib]
    xx.append([current_image, negative_image])
    yy.append(0)

  xx = np.array(xx)
  yy = np.array(yy)

  print(len(yy), len(xx))
  print(yy[:5])
  print(xx[5][:5])
  return xx, yy

xxx, yyy = create_pairs(x[:5000], y[:5000])
print(xxx.shape)



yy, xx = yyy, xxx


s = [i for i in range(len(yy))]
for i in range(10):
  w = np.random.choice(s)
  print(yy[w])
  plt.plot(xx[w][1])
  plt.plot(xx[w][0])
  plt.show()



from sklearn.preprocessing import LabelBinarizer
SIG_DIMS = (xx.shape[2], 1)

# train 70%, test is 30%
x_train, x_test, y_train, y_test = train_test_split(
    xx, yy, test_size=0.3, shuffle=True, random_state=42)

x_valid, x_test, y_valid, y_test = train_test_split(
    x_test, y_test, test_size=0.333333, shuffle=True, random_state=42)

xx, yy = [], []  # just for memory management

print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)
print(SIG_DIMS)
