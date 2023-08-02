import numpy as np
import open3d as o3d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from noise_generator import add_gaussian_noise, add_noise, add_salt_and_pepper_noise
from noise_removal_evaluator import hausdorff_distance, point_to_point_distance, rmse
import matplotlib.pyplot as plt

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
    pcd = o3d.io.read_point_cloud("dataset/45Deg/tangan_atas/cropped_obj_000018.pcd")
    # pcd = o3d.io.read_point_cloud("dataset/HumanOnly/tangan_atas/tangan_atas_cropped_obj_000018.pcd")
    pcd = o3d.io.read_point_cloud("dataset/ReducedNoise/tangan_atas/tangan_atas_cropped_obj_000018.pcd")
    # o3d.visualization.draw_geometries([pcd])
    # add_noise(pcd, 0.01)
    pcd_copy = np.array(pcd.points)
    scaler = MinMaxScaler()
    pcd_copy = scaler.fit_transform(pcd_copy)
    pcd.points = o3d.utility.Vector3dVector(pcd_copy)

    # o3d.visualization.draw_geometries([pcd])

    pcd_noised = add_salt_and_pepper_noise(pcd, probability=0.05)
    plot_pc(np.asarray(pcd.points))
    distance = rmse(pcd_copy, pcd_noised)
    print("RMSE Before: ",distance)
    pcd.points = o3d.utility.Vector3dVector(pcd_noised)
    # o3d.visualization.draw_geometries([pcd])
    # filtering multiple times will reduce the noise significantly
    # but may cause the points distribute unevenly on the surface.

    pcd_denoised = guided_filter(pcd, 0.25, 0.1)
    plot_pc(np.asarray(pcd_denoised))
    distance = rmse(pcd_copy, pcd_denoised)
    print("RMSE After: ", distance)

    distance = hausdorff_distance(pcd_copy, pcd_denoised)
    print("Hausdorff: ", distance)

    distance = point_to_point_distance(pcd_copy, pcd_denoised)
    print("P2P: ", np.mean(distance))

    pcd.points = o3d.utility.Vector3dVector(pcd_denoised)
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