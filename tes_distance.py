import numpy as np
from scipy.spatial.distance import cdist
from noise_removal_evaluator import hausdorff_distance, point_to_point_distance, rmse
import open3d as o3d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#
# pcd = o3d.io.read_point_cloud("sample/input.xyz")
# pcd_denoised = o3d.io.read_point_cloud("sample/output.xyz")
pcd = o3d.io.read_point_cloud("/home/virgantara/PythonProjects/PointCleanNet/data/pointCleanNetDataset/galera100k_noise_white_1.00e-02.xyz")
pcd_denoised = o3d.io.read_point_cloud("/home/virgantara/PythonProjects/PointCleanNet/noise_removal/results/galera100k_noise_white_1.00e-02_0.xyz")
o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([pcd_denoised])
# # Example ground truth (clean) point cloud
# ground_truth_point_cloud = np.array([[1.0, 2.0, 3.0],
#                                     [4.0, 5.0, 6.0],
#                                     [7.0, 8.0, 9.0]])
#
# # Example denoised (or noisy) point cloud
# denoised_point_cloud = np.array([[2.0, 1.5, 3.2],
#                                  [4.1, 5.2, 6.5],
#                                  [7.5, 8.1, 8.8]])
#

ground_truth_point_cloud = np.asarray(pcd.points)
denoised_point_cloud = np.asarray(pcd_denoised.points)
distance = rmse(ground_truth_point_cloud, denoised_point_cloud)
print("RMSE:", distance)

distance = hausdorff_distance(ground_truth_point_cloud, denoised_point_cloud)
print("Hausdorff Distance:", distance)

distance = point_to_point_distance(ground_truth_point_cloud, denoised_point_cloud)
print("P2P Distance:", np.mean(distance))


