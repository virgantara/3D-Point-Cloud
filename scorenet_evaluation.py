from noise_removal_evaluator import *
import open3d as o3d

# from moving_least_square
import csv
import os
from pathlib import Path
from guided_filter import *

folder_names = ['berbaring','berdiri','bungkuk','duduk','jongkok','tangan_atas']
# folder_names = ['berbaring']
list_rmse = []
list_hd = []
list_p2p = []; list_mae = []
list_cd = []
list_dc = []
dir_path = "sample/clean"
idx = 6
denoised_dir_path = "sample/denoised_scorenet_laplace_0."+str(idx)

list_f1 = []
list_f2 = []
for f in sorted(os.listdir(dir_path)):
    list_f1.append(f)

for f in sorted(os.listdir(denoised_dir_path)):
    list_f2.append(f)
for f_gt, f_denoised in zip(list_f1, list_f2):
    print(f_gt, f_denoised)
    tmp1 = Path(f_gt[2]).stem
    tmp2 = Path(f_denoised[2]).stem

    # print(tmp)

    pcd = o3d.io.read_point_cloud(os.path.join(dir_path,f_gt))
    pcd_denoised = o3d.io.read_point_cloud(os.path.join(denoised_dir_path,f_denoised))
    ground_truth_point_cloud = np.asarray(pcd.points)
#
#     # pcd_denoised = guided_filter(pcd_denoised, 0.05, 0.1)
#     # pcd_tmp = o3d.geometry.PointCloud()
#     # pcd_tmp.points = o3d.utility.Vector3dVector(pcd_denoised)
#     # pcd_denoised = guided_filter(pcd_tmp, 0.05, 0.1)
#     #
#     # pcd_tmp = o3d.geometry.PointCloud()
#     # pcd_tmp.points = o3d.utility.Vector3dVector(pcd_denoised)
#     # pcd_denoised = guided_filter(pcd_tmp, 0.05, 0.1)
#
    denoised_point_cloud = np.asarray(pcd_denoised.points)
    distance = rmse(ground_truth_point_cloud, denoised_point_cloud)

    list_rmse.append(distance)
# #
    distance = hausdorff_distance(ground_truth_point_cloud, denoised_point_cloud)

    list_hd.append(distance)
    #
    distance = point_to_point_distance(ground_truth_point_cloud, denoised_point_cloud)

    list_p2p.append(np.mean(distance))

    distance = chamfer_distance(ground_truth_point_cloud, denoised_point_cloud)

    list_cd.append(distance)

    # distance = point_cloud_mae(ground_truth_point_cloud, denoised_point_cloud)
    #
    # list_mae.append(distance)

    distance = dice_coefficient(ground_truth_point_cloud, denoised_point_cloud)

    list_dc.append(distance)
        #

print("AVG RMSE",np.mean(list_rmse))
# print("AVG MAE",np.mean(list_mae))
print("AVG P2P",np.mean(list_p2p))
print("AVG HD",np.mean(list_hd))
print("AVG CD",np.mean(list_cd))
print("AVG DC",np.mean(list_cd))
# print("")
# print("STD RMSE",np.std(list_rmse))
# print("STD MAE",np.std(list_mae))
# print("STD P2P",np.std(list_p2p))
# print("STD HD",np.std(list_hd))
# print("STD CD",np.std(list_cd))