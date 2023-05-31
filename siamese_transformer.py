# -*- coding: utf-8 -*-
"""
Created on Thu May 25 03:14:23 2023

@author: john
"""

from tensorflow import keras
from tensorflow.keras import layers


import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from tensorflow.keras.layers import Lambda
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt 
import time
from datetime import datetime
import random
from tensorflow.keras.optimizers import Adam
path = 'datasets/ptb-sanos_290.pickle'

pickleIn = open(path, 'rb')
people, y, x = pickle.load(pickleIn)




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




def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_siamese_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs_a = keras.Input(shape=input_shape)
    inputs_b = keras.Input(shape=input_shape)

    # Shared Transformer Encoder for both inputs
    shared_encoder = transformer_encoder(inputs_a, head_size, num_heads, ff_dim, dropout)
    for _ in range(1, num_transformer_blocks):
        shared_encoder = transformer_encoder(shared_encoder, head_size, num_heads, ff_dim, dropout)

    # Global Average Pooling and MLP Layers
    pooled_output = layers.GlobalAveragePooling1D(data_format="channels_first")(shared_encoder)
    for dim in mlp_units:
        pooled_output = layers.Dense(dim, activation="relu")(pooled_output)
        pooled_output = layers.Dropout(mlp_dropout)(pooled_output)

    # Siamese Branches
    output = Lambda(euclidean_distance)([pooled_output, pooled_output])
    # output_a = layers.Dense(1, activation="sigmoid")(pooled_output)
    # output_b = layers.Dense(1, activation="sigmoid")(pooled_output)
    
    output =  layers.Dense(1)(output)
    output =  layers.Activation("sigmoid")(output)
    
    
    # Siamese Model
    siamese_model = keras.Model([inputs_a, inputs_b], [output])

    return siamese_model

SIG_DIMS = (x_train.shape[2], 1)


import tensorflow as tf
from tensorflow.keras import backend as K






def euclidean_distance(vectors):
  (featsA, featsB) = vectors
  sum_squared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
  return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def contrastive_loss(y, preds, margin=1):
 y = tf.cast(y, preds.dtype)
 squared_preds = K.square(preds)
 squared_margin = K.square(K.maximum(margin - preds, 0))
 loss = 1 - K.mean(y * squared_preds + (1 - y) * squared_margin) 
 return loss

# Model parameters
input_shape = SIG_DIMS  # Define input shape based on your data
head_size = 256
num_heads = 6
ff_dim = 6
num_transformer_blocks = 6
mlp_units = [128]
mlp_dropout = 0.3
dropout = 0.25

# Build Siamese Model
siamese_model = build_siamese_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout)
siamese_model.compile(loss=contrastive_loss, optimizer = "adam", metrics=["accuracy"])
siamese_model.summary()


STEPS_PER_EPOCH = len(x_train) // 64
VAL_STEPS_PER_EPOCH = len(x_valid) // 64



#Learning Rate Scheduler
def step_decay(epoch):
    x=0.0001
    decay_rate =0.5
    if epoch >=30: x=x*decay_rate
    if epoch >=60: x=x*decay_rate
    
    
    #if epoch >=45: x=x*(decay_rate*decay_rate)

    
    # drop = 0.6
    # epochs_drop = 15
    # lr = x * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return x

lr_decay = LearningRateScheduler(step_decay)


LR = 0.01
decay = LR/100
adam = Adam(learning_rate=LR,decay=decay)

archivo_modelo = 'modelos/CYBHi_siamese_1500.h5'

model_checkpoint = ModelCheckpoint(filepath=archivo_modelo, monitor='val_accuracy',verbose=1,
                                   save_best_only=True, save_weights_only=False,mode='auto',
                                   period=1)


# Train the Siamese Model
history = siamese_model.fit([x_train[:, 0], x_train[:, 1]], y_train[:], validation_data=([x_valid[:, 0],
      x_valid[:, 1]], y_valid[:]), batch_size=64,steps_per_epoch=STEPS_PER_EPOCH, verbose=1,
              validation_steps=VAL_STEPS_PER_EPOCH,epochs=50)




# evaluate model
accuracy = siamese_model.evaluate([x_test[:, 0], x_test[:, 1]], y_test[:])#, batch_size=BS, verbose=1)
print('\n', 'Test accuracy:', accuracy, '\n')


# plot the loss
def plot_history(h):
    loss = h.history['loss']
    val_loss = h.history['val_loss']

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    #name = "media/plots/siamese_history_" + datetime.now().strftime("%Y%m%d-%H%M%S")
   # plt.savefig(name, dpi=300, bbox_inches='tight')

    plt.show()
    
def plot_predictions(m, xx_test):
    predictions = m.predict([xx_test[:, 0], xx_test[:, 1]])
    print(len(predictions))

    up, down = [], []
    for i in predictions:
        pred = max(i)
        if pred >= 0.00099999:
            up.append(pred)
        else:
            down.append(pred)

    print("Number of predicted Positive Pairs:", len(up))
    print(up, "\n")
    print("Number of predicted Negative Pairs:", len(down))
    print(down, "\n")

    fig = plt.figure(figsize=(64, 54))
    for i, idx in enumerate(np.random.choice(xx_test.shape[0], size=225, replace=False)):
        pred = max(predictions[idx])
        ax = fig.add_subplot(15, 15, i + 1, xticks=[], yticks=[])
        ax.plot(xx_test[idx][0])
        ax.plot(xx_test[idx][1])
        ax.set_title("{:.6f}".format(pred), color=("green" if pred > 0.00099999 else "red"))

    name = "siamese_predictions_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    fig.savefig(name, dpi=800, bbox_inches='tight')
    plt.show()


H = history
plot_history(H)

plot_predictions(siamese_model, x_test)
predictions = siamese_model.predict([x_test[:, 0], x_test[:, 1]])

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report

y_pred = siamese_model.predict([x_test[:, 0], x_test[:, 1]])
y_pred = np.round(y_pred).flatten()
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Generar reporte de clasificación
print(classification_report(y_test, y_pred))
