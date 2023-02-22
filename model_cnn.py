import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import warnings

import tensorflow as tf
from keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import *
from keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from Data_ModelNET10_PCD_to_Voxel_HDF5 import *
from imblearn.over_sampling import SMOTE
# from DOConv import *
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
with h5py.File("data_voxel_40.h5", "r") as hf:
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

NUM_CLASSES = 40
NUM_EPOCH = 50
model = keras.models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape = [16, 16, 16]),
    keras.layers.MaxPooling2D(),
    Conv2D(64, (2, 2), activation='relu'),
    keras.layers.MaxPooling2D(),
    Conv2D(64, (2, 2), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu',name=str(uuid.uuid4())),
    keras.layers.Dense(NUM_CLASSES, activation ='softmax',name=str(uuid.uuid4()))
])
model.add_weight(name="name")
model.summary()
model.compile(optimizer='adam',
             loss = 'categorical_crossentropy',
             metrics=['accuracy'])

history = model.fit(X_train, targets_train, epochs=NUM_EPOCH,verbose=1,
                validation_split=0.2)
model.save('cnn_modelnet40.h5',save_format='h5')
hist_df = pd.DataFrame(history.history)

# or save to csv:
hist_csv_file = 'history/history_cnn_modelnet40.csv'
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

