from noise_removal_evaluator import *
import open3d as o3d
from bilateral_filter import *
# from moving_least_square
import csv
import os
import pcl

folder_names = ['berbaring','berdiri','bungkuk','duduk','jongkok','tangan_atas']
# folder_names = ['berbaring']
list_rmse = []
list_hd = []
list_p2p = []
list_mae = []
list_cd = []
for folder_name in folder_names:
    with open('mapping_bilateral_'+folder_name+".csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            print(row)
            dir_path = "sample"
            pcd = o3d.io.read_point_cloud(os.path.join(row[2]))
            xyz_denoised = run_bilateral_denoising(
                pcd_path_in=os.path.join(row[2])
            )

            print(np.asarray(pcd.points).shape)
            pcd_denoised = o3d.geometry.PointCloud()
            pcd_denoised.points = o3d.utility.Vector3dVector(xyz_denoised)
            # pcd_denoised = o3d.io.read_point_cloud(os.path.join(dir_path,row[2]))
# pcd = o3d.io.read_point_cloud("/home/virgantara/PythonProjects/PointCleanNet/data/pointCleanNetDataset/galera100k_noise_white_1.00e-02.xyz")
# pcd_denoised = o3d.io.read_point_cloud("/home/virgantara/PythonProjects/PointCleanNet/noise_removal/results/galera100k_noise_white_1.00e-02_0.xyz")
#             o3d.visualization.draw_geometries([pcd])
#             o3d.visualization.draw_geometries([pcd_denoised])

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

            distance = point_cloud_mae(ground_truth_point_cloud, denoised_point_cloud)
            # print("MAE:", distance)
            list_mae.append(distance)
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