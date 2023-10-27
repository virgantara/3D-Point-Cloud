from noise_removal_evaluator import *
import open3d as o3d
from bilateral_filter import *
# from moving_least_square
import csv
import os

folder_names = ['berbaring','berdiri','bungkuk','duduk','jongkok','tangan_atas']
# folder_names = ['berbaring']
# list_rmse = []
# list_hd = []
# list_p2p = []
# list_mae = []
# list_cd = []
# for folder_name in folder_names:
#     with open('mapping_bilateral_'+folder_name+".csv") as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=",")
#         for row in csv_reader:
#             print(row)
#             dir_path = "sample"
#             pcd = o3d.io.read_point_cloud(os.path.join(row[2]))
#             xyz_denoised = run_bilateral_denoising(
#                 pcd_path_in=os.path.join(row[2])
#             )
#
#             print(np.asarray(pcd.points).shape)
#             pcd_denoised = o3d.geometry.PointCloud()
#             pcd_denoised.points = o3d.utility.Vector3dVector(xyz_denoised)
#             # pcd_denoised = o3d.io.read_point_cloud(os.path.join(dir_path,row[2]))
# # pcd = o3d.io.read_point_cloud("/home/virgantara/PythonProjects/PointCleanNet/data/pointCleanNetDataset/galera100k_noise_white_1.00e-02.xyz")
# # pcd_denoised = o3d.io.read_point_cloud("/home/virgantara/PythonProjects/PointCleanNet/noise_removal/results/galera100k_noise_white_1.00e-02_0.xyz")
# #             o3d.visualization.draw_geometries([pcd])
# #             o3d.visualization.draw_geometries([pcd_denoised])
#
#             ground_truth_point_cloud = np.asarray(pcd.points)
#             denoised_point_cloud = np.asarray(pcd_denoised.points)
