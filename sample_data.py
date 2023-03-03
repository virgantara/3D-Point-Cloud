import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
import open3d as o3d
import trimesh
from pathlib import Path
import numpy as np

dirname = "modelnet10_pcd"
DATA_DIR = "/media/virgantara/DATA1/Penelitian/Datasets/ModelNet10"
SAVE_PATH = "/home/virgantara/PythonProjects/PointCloud/dataset/"+dirname
print(DATA_DIR)

folders = sorted(glob.glob(os.path.join(DATA_DIR, "*")))
print (folders)
train_points = []
train_labels = []
class_map={}

print(np.array(xyz).shape)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)