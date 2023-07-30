import os.path

import matplotlib.pyplot as plt
from VoxelGrid import VoxelGrid

from helper import *

VOXEL_X = 16
VOXEL_Y = 16
VOXEL_Z = 16

def pcd_to_voxelv2(path):
    voxel_2d = []
    # LOAD PCD
    pcd = o3d.io.read_point_cloud(path)
    R = pcd.get_rotation_matrix_from_axis_angle([0, 0, 1.571])
    pcd = pcd.rotate(R, center=(0, 0, 0))
    R = pcd.get_rotation_matrix_from_axis_angle([-1.571, 0, 0])
    pcd = pcd.rotate(R, center=(0, 0, 0))
    pcd_np = np.asarray(pcd.points)
    voxel_grid = VoxelGrid(pcd_np, x_y_z=[VOXEL_X, VOXEL_Y, VOXEL_Z])
    voxel_2d = np.array(voxel_grid.vector[:, :, :])
    voxel_2d = voxel_2d.reshape(-1)
    # voxel_2d = np.where(voxel_2d > 0.0, 1, 0)
    voxel_final = voxel_2d.astype('float64')
    return voxel_final

dataset_path = "dataset/ReducedNoise/"
# dataset_path = ""
pcd_file_path = "dataset/ReducedNoise/duduk/duduk_belakang_cropped_obj_000030.pcd"

vx = pcd_to_voxelv2(pcd_file_path)

vx = vx.reshape(VOXEL_X, VOXEL_Y, VOXEL_Z)
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(vx[:,:,:], facecolors='r', edgecolor='k')

plt.show()