import open3d as o3d
import numpy as np
import glob
import struct
import os
from pathlib import Path


dataset_path = "/media/virgantara/DATA/Penelitian/Datasets/KITTI/Object/"
bin_path = dataset_path + "data_object_velodyne/training/velodyne/"
bin_path = dataset_path + "data_object_velodyne/training/pcd/"
# label_path = dataset_path + "LCAS_20160523_1200_1218_labels/"
# bin_file_path = bin_path + "000000.bin"
bin_file_path = bin_path + "000000.pcd"
# label_file_path =  label_path + "1464001237.670017000.txt"



list_pcd = []
pcd = o3d.io.read_point_cloud(bin_file_path)#bin_pcd.reshape((-1, 4))[:, 0:3]

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.points))
viewer = o3d.visualization.Visualizer()
viewer.create_window()
viewer.add_geometry(pcd)

opt = viewer.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0.5, 0.5, 0.5])
#
viewer.run()
viewer.destroy_window()
