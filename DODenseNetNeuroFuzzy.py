import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
import tensorflow.keras.backend as K
from DOConv import *
from keras.layers import *
from itertools import product
from tqdm import tqdm
import gc
import os
import scipy
import random
import h5py
from imblearn.over_sampling import SMOTE
import numpy as np
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt

# number of neurons as number of Rule will be produce
n_neurons = 100

# number of features feed to fuzzy Inference Layer
n_feature = 9

# based of article
batch_size = 70
# to get all permutaion
fRules = list(product([-1.0,0.0,1.0], repeat=n_feature))

# based on article just 100 of them are needed
out_fRules = random.sample(fRules, n_neurons)

fRules_sigma = K.transpose(out_fRules)

# Creating Densenet121
def densenet(input_shape, n_classes, filters = 32):

    #batch norm + relu + conv
    def bn_rl_conv(x, filters, kernel=1, strides=1):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = DOConv2D(filters, kernel, strides=strides, padding='same')(x)
        return x


    def dense_block(x, repetition):
        for _ in range(repetition):
            y = bn_rl_conv(x, 4 * filters)
            y = bn_rl_conv(y, filters, 3)
            x = concatenate([y, x])
        return x


    def transition_layer(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x


    input = Input(input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    for repetition in [6, 12, 24, 16]:
        d = dense_block(x, repetition)
        x = transition_layer(d)

    feature_maps = Flatten()(x)
    fuzzy_inference = []
    n_fmaps = 32
    mu = 3.0
    sigma = 1.0
    for i in tqdm(range(n_fmaps)):
        f_block = fuzzy_inference_block(output_dim=n_neurons, i_fmap=i, mu=mu, sigma=sigma)(feature_maps)
        fuzzy_inference.append(f_block)

    merged = concatenate(fuzzy_inference, axis=1)

    output = Dense(n_classes, activation='softmax')(merged)

    # x = GlobalAveragePooling2D()(d)
    # output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output)
    return model



class fuzzy_inference_block(tf.keras.layers.Layer):
    def __init__(self, output_dim, i_fmap, mu, sigma):
        self.output_dim = output_dim
        self.index = i_fmap
        self.mu = mu
        self.sigma = sigma

        super(fuzzy_inference_block, self).__init__()

    def build(self, input_shape):
        self.mu_map = fRules_sigma * self.mu
        self.sigma_map = tf.ones((n_feature, self.output_dim)) * self.sigma

        super().build(input_shape)

    def call(self, inputs):
        fMap = inputs[:, n_feature * (self.index):n_feature * (self.index + 1)]
        # create variables for processing
        aligned_x = K.repeat_elements(K.expand_dims(fMap, axis=-1), self.output_dim, -1)
        aligned_c = self.mu_map
        aligned_s = self.sigma_map

        # calculate output of each neuron (fuzzy rule)
        phi = K.exp(-K.sum(K.square(aligned_x - aligned_c) / (2 * K.square(aligned_s)),
                           axis=-2, keepdims=False))
        return phi


if __name__ == "__main__":

    NUM_CLASSES = 10

    input_shape = 16, 16, 16

    model = densenet(input_shape=input_shape, n_classes=NUM_CLASSES)
    model.summary()

    oversample = SMOTE()
    with h5py.File("data_voxel_"+str(NUM_CLASSES)+".h5", "r") as hf:
        X_train = hf["X_train"][:]
        X_train = np.array(X_train)

        targets_train = hf["y_train"][:]

        X_test = hf["X_test"][:]
        X_test = np.array(X_test)

        targets_test = hf["y_test"][:]
        test_y = targets_test
        # Determine sample shape
        sample_shape = (16, 16, 16)

        X_train, targets_train = oversample.fit_resample(X_train, targets_train)
        X_train = np.array(X_train)

        X_test, targets_test = oversample.fit_resample(X_test, targets_test)

        X_train = X_train.reshape(X_train.shape[0], 16, 16, 16)
        X_test = X_test.reshape(X_test.shape[0], 16, 16, 16)

        targets_train = to_categorical(targets_train).astype(np.int32)
        targets_test = to_categorical(targets_test).astype(np.int32)

    NUM_EPOCH = 50

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, targets_train, epochs=NUM_EPOCH, verbose=1,
                        validation_split=0.2)

    # model.save('densenet121_modelnet'+str(NUM_CLASSES)+'.h5', save_format='h5')
    hist_df = pd.DataFrame(history.history)

    # or save to csv:
    hist_csv_file = 'history_do_neuro_fuzzy_densenet121_modelnet'+str(NUM_CLASSES)+'.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    loss, accuracy = model.evaluate(X_test, targets_test)

    print(loss, accuracy)

    plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
    plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
    plt.title('Model performance for 3D Voxel Keras Conv2D (Loss)')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(['train', 'test'], loc="upper left")
    plt.show()

    # # Plot history: Categorical Accuracy
    plt.plot(history.history['accuracy'], label='Accuracy (training data)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
    plt.title('Model performance for 3D Voxel Keras Conv2D (Accuracy)')
    plt.ylabel('Accuracy value')
    plt.xlabel('No. epoch')
    plt.legend(['train', 'test'], loc="upper left")
    plt.show()
