import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import h5py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
import os, sys

np.set_printoptions(threshold=sys.maxsize)
from VoxelGrid import VoxelGrid

from pathlib import Path


def normalize_pc_range(pcd_np):
    scaler = MinMaxScaler()
    scaler.fit(pcd_np)
    pcd_np = scaler.transform(pcd_np)
    return pcd_np


def count_plot(array):
    cm = plt.cm.get_cmap('gist_rainbow')
    n, bins, patches = plt.hist(array, bins=64)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    plt.show()


def pcd_to_voxel(path):
    voxel_2d = []
    # LOAD PCD
    pcd = o3d.io.read_point_cloud(path)
    R = pcd.get_rotation_matrix_from_axis_angle([0, 0, 1.571])
    pcd = pcd.rotate(R, center=(0, 0, 0))
    R = pcd.get_rotation_matrix_from_axis_angle([-1.571, 0, 0])
    pcd = pcd.rotate(R, center=(0, 0, 0))
    pcd_np = np.asarray(pcd.points)
    pcd_np = np.array(normalize_pc_range(pcd_np))
    # VOXELIZATION
    voxel_grid = VoxelGrid(pcd_np, x_y_z=[16, 16, 16])
    voxel_2d = np.array(voxel_grid.vector[:, :, :])
    voxel_2d = voxel_2d.reshape(-1)
    voxel_2d = np.where(voxel_2d > 0.0, 1, 0)
    voxel_final = voxel_2d.astype('float64')
    return voxel_final


def generate_hdf5(DIR, filename):
    # def generate_hdf5(DIR):
    folders = glob.glob(os.path.join(DIR, "*"))

    with h5py.File(filename, 'w') as f:

        train_points = []
        train_labels = []
        test_points = []
        test_labels = []
        # class_map = {}
        folders = sorted(glob.glob(os.path.join(DIR, "*")))
        # print(folders)

        for i, folder in enumerate(folders):
            print("processing class: {}".format(os.path.basename(folder)))
            # store folder name with ID so we can retrieve later
            # class_map[i] = folder.split("/")[-1]
            # class_map[i] = folder.split("\\")[-1]
            fname = Path(folder).stem
            # gather all files
            train_files = glob.glob(os.path.join(folder, "train/*"))
            test_files = glob.glob(os.path.join(folder, "test/*"))
            print("Idx", i, fname)
            for fdata in train_files:
                train_points.append(pcd_to_voxel(fdata))
                train_labels.append(i)

            for fdata in test_files:
                test_points.append(pcd_to_voxel(fdata))
                test_labels.append(i)

        f.create_dataset("X_train", data=np.asarray(train_points))
        f.create_dataset("y_train", data=np.asarray(train_labels))
        f.create_dataset("X_test", data=np.asarray(test_points))
        f.create_dataset("y_test", data=np.asarray(test_labels))

        print("Done!")
    # return (
    #     np.asarray(train_points),
    #     np.asarray(test_points),
    #     np.asarray(train_labels),
    #     np.asarray(test_labels),
    #     class_map,
    # )


def read_hdf5(path):
    with h5py.File(path, 'r') as h5:
        X_train, y_train = h5["X_train"][:], h5["y_train"][:]
        X_test, y_test = h5["X_test"][:], h5["y_test"][:]

        return X_train, y_train, X_test, y_test


def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:, :-1]


#
DATA_DIR = "dataset/modelnet10_pcd"
#
#
generate_hdf5(DATA_DIR, "data_voxel_10.h5")
# X_train, y_train, X_test, y_test= read_hdf5("data_voxel_10.h5")
#
# #
# # X_train = rgb_data_transform(X_train)
# # print(X_train[0])
# X_train = X_train.reshape(X_train.shape[0], 16, 16, 16)
# print(X_train.shape)
# print(y_train.shape)
# print(y_train[0])
# data = X_train[0]
# print(data)
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.voxels(data, facecolors='g', edgecolor='k')
# plt.show()

# print_grid(X_train[0], 16)
# print(y_train)
# print(X_test)
# print(y_test)
