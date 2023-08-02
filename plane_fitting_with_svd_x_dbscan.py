import numpy as np
from pathlib import Path

import glob
import os, sys
import open3d as o3d
from pygroundsegmentation import GroundPlaneFitting
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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

def process_fitting(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)

    xyz_pointcloud = np.asarray((pcd.points))

    ground_estimator = GroundPlaneFitting(th_dist=0.03) #Instantiate one of the Estimators

    ground_idxs = ground_estimator.estimate_ground(xyz_pointcloud)

    # ground_pcl = xyz_pointcloud[ground_idxs]

    human_pts = [not elem for elem in ground_idxs]
    human_pts = xyz_pointcloud[human_pts]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(human_pts)

    eps = 0.5

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=20))

    if labels.shape[0] == 0:
        return np.array([])

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    unique_labels, counts = np.unique(labels, return_counts=True)

    max_density_label = unique_labels[np.argmax(counts)]
    # print(max_density_label)
    human_only = [True if item == max_density_label else False for item in labels]
    human_only_pts = human_pts[human_only]

    return human_only_pts

def extract_ground_points(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)

    xyz_pointcloud = np.asarray((pcd.points))

    ground_estimator = GroundPlaneFitting(th_dist=0.03) #Instantiate one of the Estimators

    ground_idxs = ground_estimator.estimate_ground(xyz_pointcloud)

    ground_pcl = xyz_pointcloud[ground_idxs]

    return ground_pcl

def extract_human_only_points(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)

    xyz_pointcloud = np.asarray((pcd.points))

    ground_estimator = GroundPlaneFitting(th_dist=0.03) #Instantiate one of the Estimators

    ground_idxs = ground_estimator.estimate_ground(xyz_pointcloud)

    human_pts = [not elem for elem in ground_idxs]
    human_pts = xyz_pointcloud[human_pts]
    return human_pts

base_path = "/home/virgantara/PythonProjects/LiDAROuster/20230301/PCD_Cropped/45Deg/"
folders = sorted(glob.glob(os.path.join(base_path, "*")))
for i, folder in enumerate(folders):

    print("processing class: {}".format(os.path.basename(folder)))

    folder_name = Path(folder).stem
    save_path = "dataset/HumanOnly/"+folder_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    the_files = glob.glob(os.path.join(base_path, folder+"/*"))
    for fdata in sorted(the_files):

        fname = Path(fdata).stem
        print("Processing file ", fname)
        pcd_out = extract_human_only_points(fdata)
        if pcd_out.shape[0] > 0:

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_out)
            o3d.io.write_point_cloud(save_path+"/"+folder_name+"_"+fname+".pcd", pcd)
# pcd = o3d.io.read_point_cloud(dataset_path)
#
# xyz_pointcloud = np.asarray((pcd.points))
# print(xyz_pointcloud.shape)
#
# ground_estimator = GroundPlaneFitting(th_dist=0.03) #Instantiate one of the Estimators
#
# ground_idxs = ground_estimator.estimate_ground(xyz_pointcloud)
#
# ground_pcl = xyz_pointcloud[ground_idxs]
# print(ground_pcl.shape)
#
# human_pts = [not elem for elem in ground_idxs]
# human_pts = xyz_pointcloud[human_pts]
# print(human_pts.shape)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(human_pts)
#
# eps = 0.5
# min_samples = 10
#
#
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
# pcd.paint_uniform_color([0.6, 0.6, 0.6])
#
# labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=20))
# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")
#
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
#
# unique_labels, counts = np.unique(labels, return_counts=True)
#
# max_density_label = unique_labels[np.argmax(counts)]
# # print(max_density_label)
# human_only = [True if item == max_density_label else False for item in labels]
#
#
# human_only_pts = human_pts[human_only]
#
# pcd_human_only = o3d.geometry.PointCloud()
# pcd_human_only.points = o3d.utility.Vector3dVector(human_only_pts)
#
# o3d.visualization.draw_geometries([pcd_human_only])


# -------------------------------------------------------#
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