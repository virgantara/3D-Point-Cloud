import os.path

import open3d as o3d
import numpy as np
import re

from helper import *
# L-CAS
# Column 1: 	  	category (pedestrian or group)
# Column 2-4: 	  	centroid (x-y-z)
# Column 5-7: 	  	minimum bounds (x-y-z)
# Column 8-10: 	  	maximum bounds (x-y-z)
# Column 11: 	  	visibility (0 = visible, 1 = partially visible)

dataset_path = "dataset/ReducedNoise/tangan_atas/tangan_atas_cropped_obj_000018.pcd"
# dataset_path = ""
# pcd_file_path = os.path.join(dataset_path,"tangan_atas","cropped_obj_000010.pcd")

pcd = o3d.io.read_point_cloud(dataset_path)
viewer = o3d.visualization.Visualizer()
viewer.create_window()
viewer.add_geometry(pcd)
xyz_pts = np.asarray(pcd.points)
print(xyz_pts.shape)
# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.02)
# viewer.add_geometry(voxel_grid)
#
# bbox_min = np.array([2,-0.7,-1])
# bbox_max = np.array([4,1,1])

bbox_min = np.array([2,-1.6,-1])
bbox_max = np.array([4,1.8,1])

# bbox = o3d.geometry.AxisAlignedBoundingBox()
# bbox.color = np.array([1,0,0])
# bbox.min_bound = bbox_min
# bbox.max_bound = bbox_max
# viewer.add_geometry(bbox)

opt = viewer.get_render_option()
# opt.show_coordinate_frame = True
opt.line_width = 0.1
opt.background_color = np.asarray([1, 1, 1])
viewer.run()
viewer.destroy_window()

