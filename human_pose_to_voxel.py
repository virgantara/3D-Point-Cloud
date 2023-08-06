import os

from VoxelGrid import VoxelGrid
import glob
import h5py
import numpy as np
import tqdm
import open3d as o3d



data_path = 'dataset/45Deg_merged'

def points3d_to_voxel(points, voxel_size=16):
    voxel_grid = VoxelGrid(points, x_y_z=[voxel_size, voxel_size, voxel_size])
    voxel_2d = np.array(voxel_grid.vector[:, :, :])
    # voxel_2d = voxel_2d.reshape(-1)
    voxel_final = voxel_2d.astype('float64')
    return voxel_final



def convert_to_voxel():
    all_data = []
    all_label = []

    def add_matrix_to_list(matrix_data):
        all_data.append(np.array(matrix_data))
    indeks = 0

    for folder in sorted(os.listdir(data_path)):
        for f in os.listdir(os.path.join(data_path, folder)):
            pcd = o3d.io.read_point_cloud(os.path.join(data_path, folder, f))
            data = np.asarray(pcd.points).astype('float64')
            # label = np.asarray(folder).astype('int32')
            dt = np.zeros((3500, 3), dtype='float64')
            for i,val in enumerate(data):
                dt[i] = val
            # for i, d in enumerate(dt):
            #     tgt = data[i, 0]
            #     print(d)
            all_data.append(dt)
            all_label.append(indeks)

        indeks += 1

    with h5py.File('voxels/human_pose_raw.h5', 'w') as f:
        f.create_dataset("data", data=np.asarray(all_data),dtype='float64')
        f.create_dataset("label", data=np.asarray(all_label))

convert_to_voxel()