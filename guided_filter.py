import numpy as np
import open3d as o3d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from noise_generator import add_gaussian_noise, add_noise, add_salt_and_pepper_noise
from noise_removal_evaluator import *
import matplotlib.pyplot as plt
import csv
import pandas as pd

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

def main():
    folder_name = "tangan_atas"
    list_file_name = []
    file = open('mapping_score_denoise_'+folder_name+'.csv', 'w', newline='')
    csv_writer = csv.writer(file)
    for indeks_number in range(1,7):
        # pcd = o3d.io.read_point_cloud("dataset/HumanOnly/tangan_atas/tangan_atas_cropped_obj_000018.pcd")
        input_path = "dataset/ReducedNoise/"+folder_name+"/"+folder_name+"_cropped_obj_0000"+str(indeks_number)+"0.pcd"
        pcd = o3d.io.read_point_cloud(input_path)
        # o3d.visualization.draw_geometries([pcd])
        # add_noise(pcd, 0.01)
        pcd_copy = np.array(pcd.points)
        scaler = MinMaxScaler()
        pcd_copy = scaler.fit_transform(pcd_copy)
        pcd.points = o3d.utility.Vector3dVector(pcd_copy)
        #
        # # o3d.visualization.draw_geometries([pcd])
        #
        pcd_noised = add_salt_and_pepper_noise(pcd, probability=0.05)
        #
        pcd.points = o3d.utility.Vector3dVector(pcd_noised)
        # # o3d.visualization.draw_geometries([pcd])
        # # filtering multiple times will reduce the noise significantly
        # # but may cause the points distribute unevenly on the surface.
        #
        # pcd_denoised = guided_filter(pcd, 0.25, 0.1)
        # pcd.points = o3d.utility.Vector3dVector(pcd_denoised)

        output_path = "sample/input_score_denoise_"+folder_name+"_"+str(indeks_number)+".xyz"
        csv_writer.writerow([str(indeks_number), input_path,output_path])
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
        # o3d.io.write_point_cloud(output_path, pcd)
    file.close()
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


if __name__ == '__main__':
    main()