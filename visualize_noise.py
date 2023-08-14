import os.path

import open3d as o3d
import numpy as np
import re
from noise_generator import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from helper import *
# L-CAS
# Column 1: 	  	category (pedestrian or group)
# Column 2-4: 	  	centroid (x-y-z)
# Column 5-7: 	  	minimum bounds (x-y-z)
# Column 8-10: 	  	maximum bounds (x-y-z)
# Column 11: 	  	visibility (0 = visible, 1 = partially visible)

dataset_path = "dataset/45Deg_merged/standing/18.pcd"
# dataset_path = ""
# pcd_file_path = os.path.join(dataset_path,"tangan_atas","cropped_obj_000010.pcd")

pcd = o3d.io.read_point_cloud(dataset_path)
viewer = o3d.visualization.Visualizer()
viewer.create_window()

scaler = MinMaxScaler()
pcd_copy = scaler.fit_transform(pcd.points)
xyz_pts = np.asarray(pcd.points)

pcd.points = o3d.utility.Vector3dVector(pcd_copy)
pcd.paint_uniform_color(np.array([0, 0, 1]))
viewer.add_geometry(pcd)
# o3d.visualization.draw_geometries([pcd])
# print("Before",np.asarray(pcd.points).shape)
pcd_noisy = o3d.geometry.PointCloud()
pcd_noisy.points = o3d.utility.Vector3dVector(pcd_copy)
pcd_noised = add_gaussian_noise(pcd_noisy, mean=0.01, std_dev=0.025)

pcd_noisy.points = o3d.utility.Vector3dVector(pcd_noised)
noisy_color = np.array([1, 0, 0])
pcd_noisy.paint_uniform_color(noisy_color)
viewer.add_geometry(pcd_noisy)
# # print("After",np.asarray(pcd.points).shape)

# o3d.visualization.draw_geometries([pcd])
# viewer.add_geometry(pcd)
# print(xyz_pts.shape)
# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.02)
# viewer.add_geometry(voxel_grid)
#
# bbox_min = np.array([2,-0.7,-1])
# bbox_max = np.array([4,1,1])
#
# bbox_min = np.array([2,-1.6,-1])
# bbox_max = np.array([4,1.8,1])

# bbox = o3d.geometry.AxisAlignedBoundingBox()
# bbox.color = np.array([1,0,0])
# bbox.min_bound = bbox_min
# bbox.max_bound = bbox_max
# viewer.add_geometry(bbox)
#
opt = viewer.get_render_option()
# opt.show_coordinate_frame = True
opt.line_width = 0.1
opt.background_color = np.asarray([1, 1, 1])
viewer.run()
viewer.destroy_window()

