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
from imblearn.metrics import *
from DOConv import *
import h5py
from sklearn.metrics import confusion_matrix, accuracy_score
# import seaborn as sns
# sns.set_style('white')
import uuid

from utils import *
from path import Path
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import classification_report_imbalanced


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Set the seed value for experiment reproducibility.
seed = 1842
tf.random.set_seed(seed)
np.random.seed(seed)
# Turn off warnings for cleaner looking notebook
warnings.simplefilter('ignore')
oversample = SMOTE()

BASEDATA_PATH = "/media/virgantara/DATA1/Penelitian/Datasets/"
# DATA_DIR = ""
# DATA_DIR = os.path.join(BASEDATA_PATH, "ModelNet10")
DATA_DIR = "dataset/45Deg_merged"
path = Path(DATA_DIR)
folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)};

NUM_CLASSES = np.array(folders).shape[0]
from tqdm import tqdm

# with h5py.File("data_voxel_"+str(NUM_CLASSES)+".h5", "r") as hf:
#     X_train = hf["X_train"][:]
#     X_train = np.array(X_train)
#
#     targets_train = hf["y_train"][:]
#
#     X_test = hf["X_test"][:]
#     X_test = np.array(X_test)
#
#     targets_test = hf["y_test"][:]
#     test_y = targets_test
#     # Determine sample shape
#     sample_shape = (16, 16, 16)
#
#     X_train, targets_train = oversample.fit_resample(X_train, targets_train)
#     X_train = np.array(X_train)
#
#     X_test, targets_test = oversample.fit_resample(X_test, targets_test)
#
#     X_train = X_train.reshape(X_train.shape[0], 16, 16, 16)
#     X_test = X_test.reshape(X_test.shape[0], 16, 16, 16)
#
#     targets_train = to_categorical(targets_train).astype(np.int32)
#     targets_test = to_categorical(targets_test).astype(np.int32)
VOXEL_X = 16
VOXEL_Y = 16
VOXEL_Z = 16
X_train, X_test, y_train, y_test = read_voxel_our(vx=VOXEL_X, vy=VOXEL_Y, vz=VOXEL_Z, is_oversampled=False)
# X, y = get_prepared_cross_validation(is_oversampled=False)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
#
# from sklearn.model_selection import StratifiedKFold
# kf = StratifiedKFold(n_splits=10)

NUM_EPOCH = 50
def get_model(input_shape, nclasses=10):
    x_input = tf.keras.layers.Input(shape=input_shape)
    x = DOConv2D(32, (3, 3), activation='relu')(x_input)
    x = DOConv2D(64, (2, 2), activation='relu')(x)
    x = DOConv2D(64, (2, 2), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(nclasses, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=x_input, outputs=x)

    return model

model = get_model(input_shape=(16,16,16),nclasses=NUM_CLASSES)
model.summary()
model.compile(optimizer='adam',
             loss = 'categorical_crossentropy',
             metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
history = model.fit(X_train, y_train, epochs=NUM_EPOCH, verbose=1,
                            validation_split=0.2, callbacks=[callback])
# skf = StratifiedKFold(n_splits=10)

VALIDATION_ACCURACY = []
VALIDATION_LOSS = []
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
# for i, (train_index,test_index) in tqdm(enumerate(skf.split(X,y))):
#     print(f"Fold {i}:")
#     print(np.array(X.shape))
#     print(np.array(y.shape))
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     X_train = X_train.reshape(X_train.shape[0], VOXEL_X, VOXEL_Y, VOXEL_Z)
#     X_test = X_test.reshape(X_test.shape[0], VOXEL_X, VOXEL_Y, VOXEL_Z)
#
#     y_train = to_categorical(y_train).astype(np.int32)
#     y_test = to_categorical(y_test).astype(np.int32)
#     history = model.fit(
#         X_train, y_train,
#         epochs=NUM_EPOCH,
#         verbose=1,
#         validation_split=0.2,
#         callbacks=[callback]
#     )
# # model.save('cnn_modelnet40.h5',save_format='h5')
# hist_df = pd.DataFrame(history.history)
#
# # or save to csv:
# hist_csv_file = 'history/history_cnn_modelnet'+str(NUM_CLASSES)+'.csv'
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)

loss, accuracy = model.evaluate(X_test, y_test)

VALIDATION_ACCURACY.append(accuracy)
VALIDATION_LOSS.append(loss)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


pred = model.predict(X_test)
# pred = np.argmax(pred, axis=1)
res = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))
print(res)

labels = folders
y_pred = np.argmax(pred, axis=1)
y_test = np.argmax(y_test, axis=1)
#
report = classification_report_imbalanced(y_test, y_pred,target_names=labels)
# report = metrics.classification_report(y_test, y_pred,target_names=labels)
print(report)

print("ACC",VALIDATION_ACCURACY)
print("LOSS",VALIDATION_LOSS)
# cm = pd.DataFrame(res, index = range(NUM_CLASSES), columns = range(NUM_CLASSES))
# plt.figure(figsize=(20,20))
# sns.heatmap(cm, annot=True)
# plt.show()
#
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