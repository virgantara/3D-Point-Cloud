import numpy as np

import open3d as o3d
from pygroundsegmentation import GroundPlaneFitting
from sklearn.neighbors import NearestNeighbors
import os
# from estimate_point_density import cluster_point_cloud, estimate_point_cloud_density,normalize_pc,normalize_point_cloud
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
def visualize_pcd(pcd):
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd)

    opt = viewer.get_render_option()
    # opt.show_coordinate_frame = True
    opt.line_width = 0.1
    opt.background_color = np.asarray([1, 1, 1])
    viewer.run()
    viewer.destroy_window()

dataset_path = "/home/virgantara/PythonProjects/LiDAROuster/20230301/PCD_Cropped/45Deg/duduk_depan/cropped_obj_000010.pcd"


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
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(human_pts)
# visualize_pcd(pcd)

eps = 0.5
min_samples = 10


pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
pcd.paint_uniform_color([0.6, 0.6, 0.6])
# o3d.visualization.draw_geometries([pcd]) #Works only outside Jupyter/Colab

labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=20))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])


human_only = [True if item == 1 else False for item in labels]
# print(np.asarray(pcd.points).shape)

human_only_pts = human_pts[human_only]
# print(human_only_pts.shape)
# human_only = [True if num == max_label else False for num in human_pts]
# human_only = human_pts[human_only]
# print(human_only)
pcd_human_only = o3d.geometry.PointCloud()
pcd_human_only.points = o3d.utility.Vector3dVector(human_only_pts)
#
o3d.visualization.draw_geometries([pcd_human_only])
# Cluster the point cloud.
# cluster_labels = cluster_point_cloud(human_pts, eps, min_samples)
# print("Cluster Labels:", cluster_labels, cluster_labels.shape)

# print(np.max(xyz_pointcloud))
# data = human_pts
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(data[:,0], data[:,1], data[:,2], s=300)
# ax.view_init(azim=200)
# plt.show()
#
# model = DBSCAN(eps=2.5, min_samples=500)
# model.fit_predict(data)
# pred = model.fit_predict(data)
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(data[:,0], data[:,1], data[:,2], c=model.labels_, s=300)
# ax.view_init(azim=200)
# plt.show()
#
# print("number of cluster found: {}".format(len(set(model.labels_))))
# print('cluster for each point: ', model.labels_)