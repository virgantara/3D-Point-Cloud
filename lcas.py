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

dataset_path = "/media/virgantara/DATA/Penelitian/Datasets/PointClouds/L-CAS 3D Point Cloud People Dataset/"
pcd_path = dataset_path + "LCAS_20160523_1200_1218_pcd/"
label_path = dataset_path + "LCAS_20160523_1200_1218_labels/"
pcd_file_path = pcd_path + "1464001237.670017000.pcd"
label_file_path =  label_path + "1464001237.670017000.txt"




pcd = o3d.io.read_point_cloud(pcd_file_path)
viewer = o3d.visualization.Visualizer()
viewer.create_window()
viewer.add_geometry(pcd)

list_param = []
f = open(label_file_path,'r')
mylines = f.readlines()
for line in mylines:
    lbl = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
    lbl = np.array(lbl).astype(np.float32)

    list_param.append(lbl)

list_param = np.array(list_param)

for lbl in list_param:
    # print(lbl)
    bbox_min = lbl[3:6]
    bbox_max = lbl[6:9]

    bbox = draw_bounding_box(bbox_min, bbox_max)

    viewer.add_geometry(bbox)

#
opt = viewer.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0.5, 0.5, 0.5])
viewer.run()
viewer.destroy_window()

f.close()