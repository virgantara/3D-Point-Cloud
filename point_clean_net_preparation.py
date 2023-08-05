from noise_removal_evaluator import *
import open3d as o3d
import numpy as np
import os
import random
from noise_generator import add_gaussian_noise, add_noise, add_salt_and_pepper_noise
import csv

from sklearn.preprocessing import StandardScaler, MinMaxScaler

dir_path = "dataset/ReducedNoise/"
num_files = 6

file = open('mapping_point_clean_net.csv', 'w', newline='')
csv_writer = csv.writer(file)
indeks = 0
for dir in sorted(os.listdir(dir_path)):
    folder_name = dir
    directory_path = os.path.join(dir_path, dir)
    all_files = np.asarray(sorted(os.listdir(directory_path)))

    # random.shuffle(all_files)
    indices = np.array([10,20,30,40,50,60])
    random_files = all_files[indices]
    random_files_paths = [os.path.join(directory_path, file) for file in random_files]


    # print(np.asarray(random_files_paths))
    # for f in sorted(os.listdir(os.path.join(dir_path, dir))):
    #     print(f)
    for f in random_files_paths:
        print(f)
        output_path = "sample/input_point_clean_net_" + str(indeks) + ".xyz"
        csv_writer.writerow([str(indeks), f, output_path])

        pcd = o3d.io.read_point_cloud(f)
        # o3d.visualization.draw_geometries([pcd])
        # add_noise(pcd, 0.01)
        pcd_copy = np.array(pcd.points)
        scaler = MinMaxScaler()
        pcd_copy = scaler.fit_transform(pcd_copy)
        pcd.points = o3d.utility.Vector3dVector(pcd_copy)
        pcd_noised = add_salt_and_pepper_noise(pcd, probability=0.05)
        pcd.points = o3d.utility.Vector3dVector(pcd_noised)
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
        indeks += 1
file.close()


# pcd = o3d.io.read_point_cloud("../PointCleanNet/noise_removal/results/human_0.xyz")
#
# pcd_denoised = o3d.io.read_point_cloud("../PointCleanNet/noise_removal/results/human_1.xyz")
#
# o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([pcd_denoised])
#
# list_rmse = []
# list_hd = []
# list_p2p = []; list_mae = []
# list_cd = []
# ground_truth_point_cloud = np.asarray(pcd.points)
# denoised_point_cloud = np.asarray(pcd_denoised.points)
# distance = rmse(ground_truth_point_cloud, denoised_point_cloud)
# # print("RMSE:", distance)
# list_rmse.append(distance)
# #
# distance = hausdorff_distance(ground_truth_point_cloud, denoised_point_cloud)
# # print("Hausdorff Distance:", distance)
# list_hd.append(distance)
# #
# distance = point_to_point_distance(ground_truth_point_cloud, denoised_point_cloud)
# # print("P2P Distance:", np.mean(distance))
# list_p2p.append(np.mean(distance))
#
# distance = chamfer_distance(ground_truth_point_cloud, denoised_point_cloud)
# # print("Chamfer Distance:", np.mean(distance))
# list_cd.append(distance)
#
# distance = point_cloud_mae(ground_truth_point_cloud, denoised_point_cloud)
# # print("MAE:", distance)
# list_mae.append(distance)
#
#
# print("AVG RMSE",np.mean(list_rmse))
# print("AVG MAE",np.mean(list_mae))
# print("AVG P2P",np.mean(list_p2p))
# print("AVG HD",np.mean(list_hd))
# print("AVG CD",np.mean(list_cd))
# print("")
# print("STD RMSE",np.std(list_rmse))
# print("STD MAE",np.std(list_mae))
# print("STD P2P",np.std(list_p2p))
# print("STD HD",np.std(list_hd))
# print("STD CD",np.std(list_cd))