# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 00:47:16 2023

@author: Victor Rincon
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

# load dataset
pickleIn = open('datasets/CYBHi-1500.pickle', 'rb')

people, y, x = pickle.load(pickleIn)

n_classes = len(np.unique(y))


# shuffle data
from collections import Counter

# print(Counter(y))
length = len(y)
#print(length)
data = []
for i in range(length):
  data.append([y[i], x[i]])

num = random.randint(0, length)
random.seed(num)
random.shuffle(data)

y, x = [], []
for k in range(length):
  y.append(data[k][0])
  x.append(data[k][1])

data, k, length, num = [], 0, 0, 0 # just for memory management

x = np.array(x)

x = (x - x.min()) / (x.max() - x.min())
y = np.array(y)


from sklearn.preprocessing import LabelBinarizer

SIG_DIMS = (x.shape[1], 1)

lb = LabelBinarizer()
y = lb.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, shuffle=True, random_state=42, stratify=y)

x_valid, x_test, y_valid, y_test = train_test_split(
    x_test, y_test, test_size=0.3, shuffle=True, random_state=42, stratify=(y_test))

x, y = [], []  # just for memory management

# print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], SIG_DIMS[0], SIG_DIMS[1])
x_valid = x_valid.reshape(x_valid.shape[0], SIG_DIMS[0], SIG_DIMS[1])
x_test = x_test.reshape(x_test.shape[0], SIG_DIMS[0], SIG_DIMS[1])
# print(x_train.shape)
# print(x_valid.shape)
# print(x_test.shape)


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


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


input_shape = x_train.shape[1:]


#Learning Rate Scheduler
def step_decay(epoch):
    x=0.0038
    decay_rate =0.5
    if epoch >=30: x=x*decay_rate
    if epoch >=60: x=x*decay_rate
    
    
    #if epoch >=45: x=x*(decay_rate*decay_rate)

    
    # drop = 0.6
    # epochs_drop = 15
    # lr = x * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return x


model = build_model(
input_shape,
head_size=256,
num_heads=6,
ff_dim=512,
num_transformer_blocks=6,
mlp_units=[1024, 512],
mlp_dropout=0.3,
dropout=0.25,
)

model.compile(
loss="categorical_crossentropy",
optimizer=keras.optimizers.Adam(learning_rate=0.0038),
metrics=["accuracy"],
)

model.summary()

lr_decay = LearningRateScheduler(step_decay)

archivo_modelo = 'modelos/CYBHi_1500.h5'

model_checkpoint = ModelCheckpoint(filepath=archivo_modelo, monitor='val_accuracy',verbose=1,
                                   save_best_only=True, save_weights_only=False,mode='auto',
                                   period=1)



callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

H = model.fit(
    x_train,
    y_train,
    validation_data=(x_valid,y_valid),
    epochs=200,
    batch_size=64,
    callbacks=[model_checkpoint, lr_decay, callbacks]
)

# Guardar el histórico de métricas en un archivo CSV
import pandas as pd
df = pd.DataFrame(H.history)
df.to_csv('CYBHi-training/historico.csv', index=False)

# evaluate model
_, accuracy = model.evaluate(x_test, y_test)
print('\n', 'Test accuracy:', accuracy, '\n')

# plot the training loss and accuracy
acc = H.history['accuracy']
val_acc = H.history['val_accuracy']

loss = H.history['loss']
val_loss = H.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
#plt.savefig("Training and Validation Accuracy.jpg")

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,5.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig("CYBHi-training/Training and Validation Loss  ACC.jpg", dpi = 600)
plt.show()


from sklearn.metrics import classification_report

lbb = LabelBinarizer()

predictions = model.predict(x_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(predictions, axis=1)
y_pred_bool = lbb.fit_transform(y_pred_bool)
print(classification_report(y_test, y_pred_bool))



def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.RdPu):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

test = np.argmax(y_test, axis=1)
preds = np.argmax(predictions, axis=1)
cm = confusion_matrix(y_true=test, y_pred=preds)
plot_confusion_matrix(cm=cm, classes=lb.classes_, title='Confusion Matrix')
plt.savefig("CYBHi-training/cm.jpg", dpi = 600)




x_ = x.reshape(x.shape[0], SIG_DIMS[0], SIG_DIMS[1])



predictions = model.predict(x_, batch_size=64, verbose=1)
y_pred_bool = np.argmax(predictions, axis=1)
y_pred_bool = lbb.fit_transform(y_pred_bool)
print(classification_report(y, y_pred_bool))


test = np.argmax(y, axis=1)
preds = np.argmax(predictions, axis=1)
cm = confusion_matrix(y_true=test, y_pred=preds)
plot_confusion_matrix(cm=cm, classes=lb.classes_, title='Confusion Matrix-total')
plt.savefig("CYBHi-training/cm-total.jpg", dpi = 600)