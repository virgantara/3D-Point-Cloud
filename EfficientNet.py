import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
import tensorflow.keras.backend as K
import tensorflow.keras.regularizers as rg
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
from keras.layers import *
from DOConv import *
import h5py
from imblearn.over_sampling import SMOTE
import numpy as np
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import gc
import os
import scipy
import random

VOXEL_SIZE = 16


NUM_CLASSES = 10
oversample = SMOTE()
with h5py.File("data_voxel_"+str(NUM_CLASSES)+"_"+str(VOXEL_SIZE)+".h5", "r") as hf:
    X_train = hf["X_train"][:]
    X_train = np.array(X_train)

    targets_train = hf["y_train"][:]

    X_test = hf["X_test"][:]
    X_test = np.array(X_test)

    targets_test = hf["y_test"][:]
    test_y = targets_test
    # Determine sample shape
    sample_shape = (VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)

    X_train, targets_train = oversample.fit_resample(X_train, targets_train)
    X_train = np.array(X_train)

    X_test, targets_test = oversample.fit_resample(X_test, targets_test)

    X_train = X_train.reshape(X_train.shape[0], VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
    X_test = X_test.reshape(X_test.shape[0], VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)

    targets_train = to_categorical(targets_train).astype(np.int32)
    targets_test = to_categorical(targets_test).astype(np.int32)

NUM_EPOCH = 50


def get_top(x_input):
    """Block top operations
    This functions apply Batch Normalization and Leaky ReLU activation to the input.
    # Arguments:
        x_input: Tensor, input to apply BN and activation  to.
    # Returns:
        Output tensor
    """

    x = tf.keras.layers.BatchNormalization()(x_input)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def get_block(x_input, input_channels, output_channels):
    """MBConv block
    This function defines a mobile Inverted Residual Bottleneck block with BN and Leaky ReLU
    # Arguments
        x_input: Tensor, input tensor of conv layer.
        input_channels: Integer, the dimentionality of the input space.
        output_channels: Integer, the dimensionality of the output space.

    # Returns
        Output tensor.
    """

    x = tf.keras.layers.Conv2D(input_channels, kernel_size=(1, 1), padding='same', use_bias=False)(x_input)
    x = get_top(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 3), padding='same', use_bias=False)(x)
    x = get_top(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 1), padding='same', use_bias=False)(x)
    x = get_top(x)
    x = tf.keras.layers.Conv2D(output_channels, kernel_size=(2, 1), strides=(1, 2), padding='same', use_bias=False)(x)
    return x


def EffNet(input_shape, num_classes, plot_model=False):
    """EffNet
    This function defines a EfficientNet architecture.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        num_classes: Integer, number of classes.
        plot_model: Boolean, whether to plot model architecture or not
    # Returns
        EfficientNet model.
    """
    x_input = tf.keras.layers.Input(shape=input_shape)
    x = get_block(x_input, 32, 64)
    x = get_block(x, 64, 128)
    x = get_block(x, 128, 256)
    # x = GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    # print(x.shape)

    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=x_input, outputs=x)

    if plot_model:
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    return model


def get_model(input_shape, nclasses=10):
    model = EffNet(input_shape, num_classes=nclasses)
    return model

# input_shape = (16,16,16)


input_shape = VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE

model = get_model(input_shape=input_shape, nclasses=NUM_CLASSES)
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(X_train, targets_train, epochs=NUM_EPOCH, verbose=1,
                    validation_split=0.3)
#
#
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'history_efficientnet_modelnet'+str(NUM_CLASSES)+'.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

loss, accuracy = model.evaluate(X_test, targets_test)

print(loss, accuracy)
#
# plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
# plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
# plt.title('Model performance for 3D Voxel Keras Conv2D (Loss)')
# plt.ylabel('Loss value')
# plt.xlabel('No. epoch')
# plt.legend(['train', 'test'], loc="upper left")
# plt.show()
#
# # # Plot history: Categorical Accuracy
# plt.plot(history.history['accuracy'], label='Accuracy (training data)')
# plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
# plt.title('Model performance for 3D Voxel Keras Conv2D (Accuracy)')
# plt.ylabel('Accuracy value')
# plt.xlabel('No. epoch')
# plt.legend(['train', 'test'], loc="upper left")
# plt.show()
