from noise_removal_evaluator import *
import open3d as o3d

from pathlib import Path
# from moving_least_square
import csv
import os

list_rmse = []
list_hd = []
list_p2p = []; list_mae = []
list_cd = []
dir_path = "../PointCloudDenoisingCNN/inference"
dir_clean_path = "../PointCloudDenoisingCNN/clean"
for data_in in os.listdir(dir_path):
    f_name = Path(data_in).stem
    f_ground_truth = os.path.join(dir_path, f_name+".xyz" )
    f_clean = os.path.join(dir_clean_path,f_name+".xyz")
    print(f_name)
#         dir_path = "sample"
    pcd = o3d.io.read_point_cloud(f_ground_truth)

    pcd_denoised = o3d.io.read_point_cloud(f_clean)
#
    ground_truth_point_cloud = np.asarray(pcd.points)
    denoised_point_cloud = np.asarray(pcd_denoised.points)
    distance = rmse(ground_truth_point_cloud, denoised_point_cloud)
    # print("RMSE:", distance)
    list_rmse.append(distance)
#
    distance = hausdorff_distance(ground_truth_point_cloud, denoised_point_cloud)
    # print("Hausdorff Distance:", distance)
    list_hd.append(distance)
    #
    distance = point_to_point_distance(ground_truth_point_cloud, denoised_point_cloud)
    # print("P2P Distance:", np.mean(distance))
    list_p2p.append(np.mean(distance))

    distance = chamfer_distance(ground_truth_point_cloud, denoised_point_cloud)
    # print("Chamfer Distance:", np.mean(distance))
    list_cd.append(distance)

    # distance = point_cloud_mae(ground_truth_point_cloud, denoised_point_cloud)
    # print("MAE:", distance)
    # list_mae.append(distance)
#

print("AVG RMSE",np.mean(list_rmse))
print("AVG MAE",np.mean(list_mae))
print("AVG P2P",np.mean(list_p2p))
print("AVG HD",np.mean(list_hd))
print("AVG CD",np.mean(list_cd))
print("")
print("STD RMSE",np.std(list_rmse))
print("STD MAE",np.std(list_mae))
print("STD P2P",np.std(list_p2p))
print("STD HD",np.std(list_hd))
print("STD CD",np.std(list_cd))