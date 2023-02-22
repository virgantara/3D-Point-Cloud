import open3d as o3d
import numpy as np
import glob
import struct
import os
from pathlib import Path
def bin2pcd(f_path,f_out_path):
    size_float = 4
    list_pcd = []
    with open(f_path, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd

# dataset_path = "/media/virgantara/DATA/Penelitian/Datasets/KITTI/Object/"
# bin_path = dataset_path + "data_object_velodyne/testing/velodyne/"
#
# list_pcd = []
# for f in glob.glob(bin_path+"/*"):
#     out_path = dataset_path + "data_object_velodyne/testing/pcd"
#
#     pcd = bin2pcd(f, out_path)
#     f_name = Path(f).stem
#
#     out_path = out_path + "/" + f_name+".pcd"
#     print("Writing : ", out_path)
#     o3d.io.write_point_cloud(out_path, pcd)
