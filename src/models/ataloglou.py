import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (Input, Conv2D, ReLU, 
                                     Concatenate, Average, 
                                     BatchNormalization,
                                     Add)

from tensorflow.keras.metrics import BinaryAccuracy, MeanIoU
from tensorflow.keras.initializers import RandomNormal, RandomUniform, GlorotUniform
from tensorflow.keras.optimizers import Adam, RMSprop

def CNN(in_layer):
    w_init0 = RandomUniform(minval=-1.0, maxval=1.0, seed=42)
    w_init1 = RandomUniform(minval=-1.0, maxval=1.0, seed=42)

    x = Conv2D(128, 3, padding="same", kernel_initializer=w_init0)(in_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, 3, padding="same", kernel_initializer=w_init1)(x)
    x = BatchNormalization()(x)
    return Add()([x, in_layer])

def AtaloglouCNN(input_layer):
    x1 = Conv2D(128, 3, padding="same")(input_layer)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x2 = CNN(x1)
    x3 = CNN(x2)
    x4 = CNN(x3)
    x5 = CNN(x4)
    x6 = CNN(x5)
    x7 = CNN(x6)

    x8 = Conv2D(64, 3, padding="same")(x7)
    x8 = BatchNormalization()(x8)
    x8 = ReLU()(x8)

    x9 = Conv2D(1, 3, padding="same", activation="linear")(x8)
    return x9

def AtaloglouSeg(input_shape=(120, 120, 1)):
    inputs = Input(shape=input_shape)

    out = AtaloglouCNN(inputs)

    out = (tf.nn.tanh(out) + 1)/2

    model = keras.Model(inputs=inputs, outputs = out) 

    model.compile(optimizer = "Adam",
                  loss = "mse",
                  metrics = BinaryAccuracy())
    return model

def AtaloglouCorr(in_shape=(100, 100, 1)):
    input_raw = Input(shape=in_shape)
    input_mask = Input(shape=in_shape)
    x1 = Concatenate()([input_raw, input_mask])

    x2 = AtaloglouCNN(x1)
    replace_out = (tf.nn.tanh(x2) + 1)/2

    x3 = Concatenate()([input_raw, input_mask, replace_out])
    refine_out = AtaloglouCNN(x3)

    out = Add()([refine_out, replace_out])

    model = keras.Model(inputs=[input_raw, input_mask], outputs=out)

    model.compile(optimizer = "Adam",
                  loss = "mse",
                  metrics = BinaryAccuracy())

    return model

def AtaloglouSeg3D(mri_in, sagit, coron, axial):

    X = tf.expand_dims(mri_in, -1)
    X_sagit = X
    X_coron = tf.transpose(X, perm=[1, 0, 2, 3])
    X_axial = tf.transpose(X, perm=[2, 0, 1, 3])

    X_sagit = sagit.predict(X_sagit)
    X_coron = tf.transpose(coron.predict(X_coron), perm=[1, 0, 2, 3])
    X_axial = tf.transpose(axial.predict(X_axial), perm=[1, 2, 0, 3])

    return Average()([X_sagit, X_coron, X_axial])


if __name__ == "__main__":
    axial_cor = AtaloglouCorr()
    plot_model(axial_cor, "model.png", show_dtype=False, rankdir="TB",
               show_shapes=False, show_layer_names=False)
