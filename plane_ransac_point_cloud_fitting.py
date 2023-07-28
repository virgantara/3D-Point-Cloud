import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import open3d as o3d
from sklearn.neighbors import KDTree
dataset_path = "/home/virgantara/PythonProjects/LiDAROuster/20230301/PCD_Cropped/45Deg/duduk_depan/cropped_obj_000010.pcd"


pcd = o3d.io.read_point_cloud(dataset_path)
# viewer = o3d.visualization.Visualizer()
# viewer.create_window()
# viewer.add_geometry(pcd)
xyz = np.asarray(pcd.points)
# xyz_pointcloud = np.asarray((pcd.points))
# print(xyz_pointcloud.shape)

tree = KDTree(np.array(xyz), leaf_size=2)
nearest_dist, nearest_ind = tree.query(xyz, k=8)
mean_distance = np.mean(nearest_dist[:,1:])
inliers=[]
idx_samples = random.sample(range(len(xyz)), 3)
pts = xyz[idx_samples]

vecA = pts[1] - pts[0]
vecB = pts[2] - pts[0]
normal = np.cross(vecA, vecB)
a,b,c = normal / np.linalg.norm(normal)
d=-np.sum(normal*pts[1])

idx_inliers = []  # list of inliers ids
distance = (a * xyz[:,0] + b * xyz[:,1] + c * xyz[:,2] + d
            ) / np.sqrt(a ** 2 + b ** 2 + c ** 2)

threshold=0.05
idx_candidates = np.where(np.abs(distance) <= threshold)[0]
if len(idx_candidates) > len(inliers):
    equation = [a,b,c,d]
    inliers = idx_candidates
    # print(inliers)

xyz_in=xyz[inliers]

mask = np.ones(len(xyz), dtype=bool)
mask[inliers] = False

xyz_out=xyz[mask]

# ax = plt.axes(projection='3d')
# ax.scatter(xyz_in[:,0], xyz_in[:,1], xyz_in[:,2], c = 'cornflowerblue', s=0.02)
# ax.scatter(xyz_out[:,0], xyz_out[:,1], xyz_out[:,2], c = 'salmon', s=0.02)
# plt.show()

def ransac_plane(xyz, threshold=0.05, iterations=1000):
    inliers = []
    n_points = len(xyz)
    i = 1

    while i < iterations:
        idx_samples = random.sample(range(n_points), 3)
        pts = xyz[idx_samples]

        vecA = pts[1] - pts[0]
        vecB = pts[2] - pts[0]
        normal = np.cross(vecA, vecB)
        a, b, c = normal / np.linalg.norm(normal)
        d = -np.sum(normal * pts[1])

        distance = (a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d
                    ) / np.sqrt(a ** 2 + b ** 2 + c ** 2)

        idx_candidates = np.where(np.abs(distance) <= threshold)[0]

        if len(idx_candidates) > len(inliers):
            equation = [a, b, c, d]
            inliers = idx_candidates

        i += 1
    return equation, inliers

eq,idx_inliers=ransac_plane(xyz,0.01)
inliers=xyz[idx_inliers]

mask = np.ones(len(xyz), dtype=bool)
mask[idx_inliers] = False

outliers=xyz[mask]

print(inliers.shape)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(inliers)
# viewer = o3d.visualization.Visualizer()
# viewer.create_window()
# viewer.add_geometry(pcd)
#
# opt = viewer.get_render_option()
# # opt.show_coordinate_frame = True
# opt.line_width = 0.1
# opt.background_color = np.asarray([1, 1, 1])
# viewer.run()
# viewer.destroy_window()

# ax = plt.axes(projection='3d')
# ax.scatter(inliers[:,0], inliers[:,1], inliers[:,2], c = 'cornflowerblue', s=0.02)
# ax.scatter(outliers[:,0], outliers[:,1], outliers[:,2], c = 'salmon', s=0.02)
# plt.show()