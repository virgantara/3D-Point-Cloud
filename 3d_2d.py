
import numpy as np
import matplotlib.pyplot as plt
from helper import *

base_path = 'dataset/2011_09_26_drive_0005_extract/2011_09_26/2011_09_26_drive_0005_extract/velodyne_points/data/'
file_path = base_path+'0000000000.txt'

path_velo2cam = 'dataset/2011_09_26_calib/2011_09_26/calib_velo_to_cam.txt'

Tr_velo_to_cam = load_calibration_rigid(path_velo2cam)
print(Tr_velo_to_cam)

path_cam2cam = 'dataset/2011_09_26_calib/2011_09_26/calib_cam_to_cam.txt'
calib = load_calibration_cam_to_cam(path_cam2cam)
print(calib['S_Rect'])
# pc = read_data(file_path)
#
# extrinsic_params = read_extrinsic(param_path)
# intrinsic_params = read_intrinsic(param_path)
# # print(mat_intrinsic.shape, extrinsic_params.shape)
#
# pt0 = pc[750]
# X, Y, Z = pt0[0], pt0[1], pt0[2]
# print("Titik Point Cloud 3D",pt0)
# coor_3d = np.array([X,Y,Z,1])
#
# pt_2d = np.matmul(np.matmul(intrinsic_params, extrinsic_params), coor_3d)
#
# print("Titik 2D piksel",pt_2d)

# print(np.matmul(intrinsic_params, extrinsic_params))