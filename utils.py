import h5py
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical

def read_voxel_our(vx, vy, vz, is_oversampled=True):
    oversample = SMOTE()
    with h5py.File("data_voxel_45deg_merged_half_16_16_16.h5", "r") as hf:
        X = hf["data_points"][:]
        X = np.array(X)

        y = hf["data_labels"][:]
        if is_oversampled:
            X, y = oversample.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

        X_train = np.array(X_train)

        X_train = X_train.reshape(X_train.shape[0], vx, vy, vz)
        X_test = X_test.reshape(X_test.shape[0], vx, vy, vz)

        y_train = to_categorical(y_train).astype(np.int32)
        y_test = to_categorical(y_test).astype(np.int32)

    return X_train, X_test, y_train, y_test

def get_prepared_cross_validation(is_oversampled=True):
    oversample = SMOTE()
    with h5py.File("data_voxel_45deg_merged_half_16_16_16.h5", "r") as hf:
        X = hf["data_points"][:]
        X = np.array(X)

        y = hf["data_labels"][:]
        if is_oversampled:
            X, y = oversample.fit_resample(X, y)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
        #
        # X_train = np.array(X_train)
        #
        # X_train = X_train.reshape(X_train.shape[0], vx, vy, vz)
        # X_test = X_test.reshape(X_test.shape[0], vx, vy, vz)

        # y = to_categorical(y).astype(np.int32)


    return X, y