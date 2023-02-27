
import h5py
from imblearn.over_sampling import SMOTE
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

with h5py.File("data_voxel_10.h5", "r") as hf:
    X_train = hf["X_train"][:]
    X_train = np.array(X_train)

    y_train = hf["y_train"][:]

    X_test = hf["X_test"][:]
    X_test = np.array(X_test)

    y_test = hf["y_test"][:]
    # test_y = targets_test
    # Determine sample shape
    # X_train, targets_train = oversample.fit_resample(X_train, targets_train)
    # X_train = np.array(X_train)
    X = np.vstack([X_train, X_test])
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    y = np.vstack([np.array(y_train), np.array(y_test)])
    # print(y.shape)
    # X_test, targets_test = oversample.fit_resample(X_test, targets_test)
