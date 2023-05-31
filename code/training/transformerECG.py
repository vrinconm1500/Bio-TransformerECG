# Author: Victor Rincon 
# ECG Biometric Indetification & Authentication using Transformer and Transformer-Siamese 
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import tensorflow as tf
import tensorflow.keras as keras
import time
from datetime import datetime


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
    n_classes=0,
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

def step_decay(epoch, initial_lr, decay_rate, decay_epochs):
    lr = initial_lr
    for decay_epoch in decay_epochs:
        if epoch >= decay_epoch:
            lr *= decay_rate
    return lr


def train(folder, sig_dims, data, epoch, initial_lr, decay_rate, decay_epochs):
    x_train, y_train, x_valid, y_valid, x_test, y_test, lb = data
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save the label binarizer to disk
    with open(folder + "lb.pickle", "wb") as f:
        pickle.dump(lb, f)

    # Model
    model = build_model(
        sig_dims,
        head_size=256,
        num_heads=6,
        ff_dim=512,
        num_transformer_blocks=6,
        mlp_units=[1024],
        mlp_dropout=0.3,
        dropout=0.25,
        n_classes=len(lb.classes_)
        )

    
    
    model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.002),
    metrics=["accuracy"],
    )

    print(model.summary())

    lr_decay = LearningRateScheduler(lambda epoch: step_decay(epoch, initial_lr, decay_rate, decay_epochs))
    best_model = folder + "Transformer-ID"
    model_checkpoint = ModelCheckpoint(filepath=best_model, monitor='val_accuracy',verbose=1,
                                   save_best_only=True, save_weights_only=False,mode='auto',
                                   period=1)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Define the Keras TensorBoard callback.
    log_dir = folder + "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir)


    # fit network
    t = time.time()


    H = model.fit(
    x_train,
    y_train,
    validation_data=(x_valid,y_valid),
    epochs=epoch,
    batch_size=64,
    callbacks=[model_checkpoint,tensorboard_callback, lr_decay]
    )

    print('\nTraining time: ', time.time() - t)

    # save the model to disk
    model.save(best_model)

    return model, H