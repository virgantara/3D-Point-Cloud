import open3d as o3d
import numpy as np
import glob
import struct
import os
from pathlib import Path
from helper import *
import re

#Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
HRES = 0.35         # horizontal resolution (assuming 20Hz setting)
VRES = 0.4          # vertical res
VFOV = (-24.9, 2.0) # Field of view (-ve, +ve) along vertical axis
Y_FUDGE = 5         # y fudge factor for velodyne HDL 64E


dataset_path = "/media/virgantara/DATA/Penelitian/Datasets/KITTI/Object/"
# type_path = dataset_path + "data_object_velodyne/training/velodyne/"
type_path = dataset_path + "data_object_velodyne/training/pcd/"
label_path = dataset_path + "data_object_label_2/training/label_2/"
file_path = type_path + "000000.pcd"
label_file_path =  label_path + "000000.txt"

IMG_DIR = dataset_path+'data_object_image_2/training/image_2'
LABEL_DIR = dataset_path+'data_object_label_2/training/label_2'
POINT_CLOUD_DIR = dataset_path+'data_object_velodyne/training/velodyne'
CALIB_DIR = dataset_path+'data_object_calib/training/calib'
def get_bin_kitti(file_id):

    label_filename = os.path.join(LABEL_DIR, '{0:06d}.txt'.format(file_id))
    pc_filename = os.path.join(POINT_CLOUD_DIR, '{0:06d}.bin'.format(file_id))
    calib_filename = os.path.join(CALIB_DIR, '{0:06d}.txt'.format(file_id))

    obj = load_pc_from_bin(pc_filename)
    return obj


list_pcd = []
pcd = o3d.io.read_point_cloud(file_path)
pts = np.array(pcd.points)
# print(pts)

list_param = []
f = open(label_file_path,'r')
mylines = f.readlines()
for line in mylines:
    lbl = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
    lbl = np.array(lbl).astype(np.float32)

    list_param.append(lbl)

list_param = np.array(list_param)
print(list_param)
# print(list_param.shape)
for lbl in list_param:
    center_pt = lbl[10:13]
    print("Obj Loc: ",center_pt)
    # bbox_min = lbl[3:6]
    # bbox_max = lbl[6:9]
    # print("BboxMin:", bbox_min)
    # print("BboxMax:", bbox_max)
# viewer = o3d.visualization.Visualizer()
# viewer.create_window()
# viewer.add_geometry(pcd)
# opt = viewer.get_render_option()
# opt.show_coordinate_frame = True
# opt.background_color = np.asarray([0.5, 0.5, 0.5])
# viewer.run()
# viewer.destroy_window()
# pts = pcd.points
points = get_bin_kitti(file_id=0)
# print(np.array(points).shape)
im = point_cloud_to_panorama_reflectance(points,
                             v_res=VRES,
                             h_res=HRES,
                             v_fov=VFOV,
                             y_fudge=Y_FUDGE,
                             r_range=(0,1))
# im = point_cloud_to_panorama(pts,
#                              v_res=VRES,
#                              h_res=HRES,
#                              v_fov=VFOV,
#                              y_fudge=Y_FUDGE,
#                              d_range=(0,100))
plt.imshow(im,cmap='Spectral')
# plt.savefig("pic/spec.png")
plt.show()
# print(im.shape)