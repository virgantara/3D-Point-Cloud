import numpy as np

import open3d as o3d
from pygroundsegmentation import GroundPlaneFitting
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os
# from estimate_point_density import cluster_point_cloud, estimate_point_cloud_density,normalize_pc,normalize_point_cloud
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset_path = "/home/virgantara/PythonProjects/LiDAROuster/20230301/PCD_Cropped/45Deg/berbaring_depan/cropped_obj_000060.pcd"


pcd = o3d.io.read_point_cloud(dataset_path)
# viewer = o3d.visualization.Visualizer()
# viewer.create_window()
# viewer.add_geometry(pcd)
# xyz_pointcloud = np.asarray(normalize_point_cloud(np.asarray(pcd.points)))
xyz_pointcloud = np.asarray((pcd.points))
print(xyz_pointcloud.shape)

ground_estimator = GroundPlaneFitting(th_dist=0.03) #Instantiate one of the Estimators

# xyz_pointcloud = np.random.rand(1000,3) #Example Pointcloud
ground_idxs = ground_estimator.estimate_ground(xyz_pointcloud)
# print(ground_idxs)
ground_pcl = xyz_pointcloud[ground_idxs]
print(ground_pcl.shape)

human_pts = [not elem for elem in ground_idxs]
human_pts = xyz_pointcloud[human_pts]
print(human_pts.shape)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(human_pts)
# visualize_pcd(pcd)
# Number of clusters (K)
n_clusters = 3
scaled_points = StandardScaler().fit_transform(human_pts)
# Clustering:
model = KMeans(n_clusters=n_clusters, n_init=2)
model.fit(scaled_points)
# Get labels:
labels = model.labels_
# Get the number of colors:
n_clusters = len(set(labels))
print('clusters num:',n_clusters)
# Mapping the labels classes to a color map:
colors = plt.get_cmap("tab20")(labels / (n_clusters if n_clusters > 0 else 1))
# Attribute to noise the black color:
colors[labels < 0] = 0
# Update points colors:
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(human_pts)
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Display:
o3d.visualization.draw_geometries([pcd])