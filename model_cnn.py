import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import warnings
import tensorflow.keras.backend as K
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import *
from keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from Data_ModelNET10_PCD_to_Voxel_HDF5 import *
from itertools import product
from tqdm import tqdm
import gc
import os
import scipy
import random
from imblearn.over_sampling import SMOTE
from DOConv import *
import h5py
from sklearn.metrics import confusion_matrix, accuracy_score
# import seaborn as sns
# sns.set_style('white')
import uuid

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Set the seed value for experiment reproducibility.
seed = 1842
tf.random.set_seed(seed)
np.random.seed(seed)
# Turn off warnings for cleaner looking notebook
warnings.simplefilter('ignore')
oversample = SMOTE()


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

NUM_CLASSES = 10

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
def get_model(input_shape, nclasses=10):
    x_input = tf.keras.layers.Input(shape=input_shape)
    x = DOConv2D(32, (3, 3), activation='relu')(x_input)
    x = DOConv2D(64, (2, 2), activation='relu')(x)
    x = DOConv2D(64, (2, 2), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)

    # neuro fuzzy inference
    mu = 3.0
    sigma = 1.0
    n_femap = 64
    #     feature_maps = Flatten()(x)
    fuzzy_inference = []
    for i in tqdm(range(n_femap)):
        f_block = fuzzy_inference_block(output_dim=n_neurons, i_fmap=i, mu=mu, sigma=sigma)(x)
        fuzzy_inference.append(f_block)

    merged = concatenate(fuzzy_inference, axis=1)
    x = tf.keras.layers.Dense(nclasses, activation='softmax')(merged)
    model = tf.keras.models.Model(inputs=x_input, outputs=x)

    return model

model = get_model(input_shape=(16,16,16),nclasses=NUM_CLASSES)
model.summary()
model.compile(optimizer='adam',
             loss = 'categorical_crossentropy',
             metrics=['accuracy'])

history = model.fit(X_train, targets_train, epochs=NUM_EPOCH,verbose=1,
                validation_split=0.2)
# model.save('cnn_modelnet40.h5',save_format='h5')
hist_df = pd.DataFrame(history.history)

# or save to csv:
hist_csv_file = 'history/history_cnn_modelnet'+str(NUM_CLASSES)+'.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

loss, accuracy = model.evaluate(X_test, targets_test)


print("Loss: ", loss)
print("Accuracy: ", accuracy)


pred = model.predict(X_test)
# pred = np.argmax(pred, axis=1)
res = confusion_matrix(np.argmax(targets_test, axis=1), np.argmax(pred, axis=1))
print(res)
# cm = pd.DataFrame(res, index = range(NUM_CLASSES), columns = range(NUM_CLASSES))
# plt.figure(figsize=(20,20))
# sns.heatmap(cm, annot=True)
# plt.show()
#
# plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
# plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
# plt.title('Model performance for 3D Voxel Keras Conv2D (Loss)')
# plt.ylabel('Loss value')
# plt.xlabel('No. epoch')
# plt.legend(['train','test'],loc="upper left")
# plt.show()
#
# # # Plot history: Categorical Accuracy
# plt.plot(history.history['accuracy'], label='Accuracy (training data)')
# plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
# plt.title('Model performance for 3D Voxel Keras Conv2D (Accuracy)')
# plt.ylabel('Accuracy value')
# plt.xlabel('No. epoch')
# plt.legend(['train','test'],loc="upper left")
# plt.show()

