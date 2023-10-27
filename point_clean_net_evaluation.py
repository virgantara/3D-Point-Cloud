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
with open("mapping_point_clean_net.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        tmp = Path(row[2]).stem
        # print(tmp)
        dir_path = "../PointCleanNet/noise_removal/results"
        for f in os.listdir(dir_path):
            if f.startswith(tmp+"_1"):
                # print(f)
        # f =
                pcd = o3d.io.read_point_cloud(os.path.join(row[2]))
                pcd_denoised = o3d.io.read_point_cloud(os.path.join(dir_path,f))
                ground_truth_point_cloud = np.asarray(pcd.points)

                # pcd_denoised = guided_filter(pcd_denoised, 0.05, 0.1)
                # pcd_tmp = o3d.geometry.PointCloud()
                # pcd_tmp.points = o3d.utility.Vector3dVector(pcd_denoised)
                # pcd_denoised = guided_filter(pcd_tmp, 0.05, 0.1)
                #
                # pcd_tmp = o3d.geometry.PointCloud()
                # pcd_tmp.points = o3d.utility.Vector3dVector(pcd_denoised)
                # pcd_denoised = guided_filter(pcd_tmp, 0.05, 0.1)

                denoised_point_cloud = np.asarray(pcd_denoised.points)
                distance = rmse(ground_truth_point_cloud, denoised_point_cloud)
                # print("RMSE:", distance)
                list_rmse.append(distance)
# #
                distance = hausdorff_distance(ground_truth_point_cloud, denoised_point_cloud)
                list_hd.append(distance)
                #
                distance = point_to_point_distance(ground_truth_point_cloud, denoised_point_cloud)
                list_p2p.append(np.mean(distance))

                distance = chamfer_distance(ground_truth_point_cloud, denoised_point_cloud)
                list_cd.append(distance)

                distance = point_cloud_mae(ground_truth_point_cloud, denoised_point_cloud)
                list_mae.append(distance)

                distance = dice_coefficient(ground_truth_point_cloud, denoised_point_cloud)
                list_dc.append(distance)
        #

print("AVG RMSE",np.mean(list_rmse))
print("AVG MAE",np.mean(list_mae))
print("AVG P2P",np.mean(list_p2p))
print("AVG HD",np.mean(list_hd))
print("AVG CD",np.mean(list_cd))
print("AVG DC",np.mean(list_dc))
# print("")
# print("STD RMSE",np.std(list_rmse))
# print("STD MAE",np.std(list_mae))
# print("STD P2P",np.std(list_p2p))
# print("STD HD",np.std(list_hd))
# print("STD CD",np.std(list_cd))