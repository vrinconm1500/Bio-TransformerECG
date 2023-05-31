# Author: Victor Rincon 
# ECG Biometric Indetification & Authentication using Transformer and Transformer-Siamese 

# import the necessary packages

import pickle
import random
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import  load_model

def load_data(where):
    with open(where, "rb") as pickleIn:
        ps, yy, xx = pickle.load(pickleIn)
    yy = np.array(yy)
    xx = np.array(xx)

    # Normalize data
    print('Before Normalizing: Min:', xx.min(), 'Max:', xx.max(), "\n")
    xx = (xx - xx.min()) / (xx.max() - xx.min())
    print('After Normalizing: Min:', xx.min(), 'Max:', xx.max(), "\n")

    return yy, xx, ps



# Shuffle data
def shuffle(yy, xx):
    length = len(yy)
    data = []
    for i in range(length):
        data.append([yy[i], xx[i]])

    num = random.randint(0, length)
    random.seed(num) 
    random.shuffle(data)

    yy, xx = [], []
    for k in range(length):
        yy.append(data[k][0])
        xx.append(data[k][1])

    xx = np.array(xx)
    yy = np.array(yy)

    return yy, xx

def shuffle_data(yy, xx):
    length = len(yy)
    data = list(zip(yy, xx))
    random.shuffle(data)
    yy, xx = zip(*data)
    yy = np.array(yy)
    xx = np.array(xx)
    return yy, xx


def splits(yy, xx, sig_dims):
    # binarize the labels
    lbb = LabelBinarizer()
    yy = lbb.fit_transform(yy)

    # train 70%, test is 30%
    xx_train, xx_test, yy_train, yy_test = train_test_split(
        xx, yy, test_size=0.4, shuffle=True, random_state=42)

    xx_valid, xx_test, yy_valid, yy_test = train_test_split(
        xx_test, yy_test, test_size=0.5, shuffle=True, random_state=42)

    print("X train shape:", xx_train.shape)
    xx_train = xx_train.reshape(xx_train.shape[0], sig_dims[0], sig_dims[1])
    xx_valid = xx_valid.reshape(xx_valid.shape[0], sig_dims[0], sig_dims[1])
    xx_test = xx_test.reshape(xx_test.shape[0], sig_dims[0], sig_dims[1])
    print("X train shape:", xx_train.shape)
    print("X valid shape:", xx_valid.shape)
    print("X test shape:", xx_test.shape, "\n")

    return xx_train, yy_train, xx_valid, yy_valid, xx_test, yy_test, lbb

def split_data(yy, xx, sig_dims):
    # binarize the labels
    lbb = LabelBinarizer()
    yy = lbb.fit_transform(yy)

    # train 70%, validation 20%, test 10%
    xx_train, xx_test, yy_train, yy_test = train_test_split(
        xx, yy, test_size=0.3, random_state=42)
    xx_valid, xx_test, yy_valid, yy_test = train_test_split(
        xx_test, yy_test, test_size=0.33, random_state=42)

    print("X train shape:", xx_train.shape)
    xx_train = xx_train.reshape(xx_train.shape[0], *sig_dims)
    xx_valid = xx_valid.reshape(xx_valid.shape[0], *sig_dims)
    xx_test = xx_test.reshape(xx_test.shape[0], *sig_dims)
    print("X train shape:", xx_train.shape)
    print("X valid shape:", xx_valid.shape)
    print("X test shape:", xx_test.shape, "\n")

    return xx_train, yy_train, xx_valid, yy_valid, xx_test, yy_test, lbb