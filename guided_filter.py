import numpy as np
import open3d as o3d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from noise_generator import add_gaussian_noise, add_noise, add_salt_and_pepper_noise
from noise_removal_evaluator import *
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
from pathlib import Path
def plot_pc(point_cloud):
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates from the point cloud
    x_coords = point_cloud[:, 0]
    y_coords = point_cloud[:, 1]
    z_coords = point_cloud[:, 2]

    # Plot the points in the 3D space
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the title for the plot
    plt.title('3D Point Cloud')

    # Show the plot
    plt.show()
def guided_filter(pcd, radius, epsilon):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    points_copy = np.array(pcd.points)
    points = np.asarray(pcd.points)
    num_points = len(pcd.points)

    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        if k < 3:
            continue

        neighbors = points[idx, :]
        mean = np.mean(neighbors, 0)
        cov = np.cov(neighbors.T)
        e = np.linalg.inv(cov + epsilon * np.eye(3))

        A = cov @ e
        b = mean - A @ mean

        points_copy[i] = A @ points[i] + b


    return points_copy

def main():
    dir_path = "sample/noisy/gaussian_0.02"
    denoised_path = "sample/denoised_guided_filter"
    for f in sorted(os.listdir(dir_path)):
        pcd = o3d.io.read_point_cloud(os.path.join(dir_path, f))
        xyz_denoised = guided_filter(pcd, radius=0.25, epsilon=0.1)

        f_name = Path(f).stem
        output_path = os.path.join(denoised_path, f_name + ".xyz")
        pcd_denoised = o3d.geometry.PointCloud()
        pcd_denoised.points = o3d.utility.Vector3dVector(xyz_denoised)
        o3d.io.write_point_cloud(output_path, pcd_denoised, write_ascii=True)
    # ground_truth_point_cloud = np.asarray(pcd.points)
    # denoised_point_cloud = np.asarray(pcd_denoised)
    # distance = rmse(ground_truth_point_cloud, denoised_point_cloud)
    # print("RMSE:", distance)
    #
    # distance = hausdorff_distance(ground_truth_point_cloud, denoised_point_cloud)
    # print("Hausdorff Distance:", distance)
    #
    # distance = point_to_point_distance(ground_truth_point_cloud, denoised_point_cloud)
    # print("P2P Distance:", np.mean(distance))
    #
    # distance = chamfer_distance(ground_truth_point_cloud, denoised_point_cloud)
    # print("Chamfer Distance:", np.mean(distance))
    #
    # distance = point_cloud_mae(ground_truth_point_cloud, denoised_point_cloud)
    # print("MAE:", distance)

    # plot_pc(np.asarray(pcd_denoised))
    # distance = rmse(pcd_copy, pcd_denoised)
    # print("RMSE After: ", distance)
    #
    # distance = hausdorff_distance(pcd_copy, pcd_denoised)
    # print("Hausdorff: ", distance)
    #
    # distance = point_to_point_distance(pcd_copy, pcd_denoised)
    # print("P2P: ", np.mean(distance))

    # pcd.points = o3d.utility.Vector3dVector(pcd_denoised)
    # o3d.io.write_point_cloud("sample/output_gf.xyz", pcd)
    # o3d.visualization.draw_geometries([pcd])





if __name__ == '__main__':
    main()