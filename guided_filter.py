import numpy as np
import open3d as o3d
from sklearn.preprocessing import StandardScaler
from noise_generator import add_gaussian_noise, add_noise, add_salt_and_pepper_noise
def main():
    pcd = o3d.io.read_point_cloud("dataset/45Deg/tangan_atas/cropped_obj_000018.pcd")
    # o3d.visualization.draw_geometries([pcd])
    # add_noise(pcd, 0.01)
    pcd_copy = np.array(pcd.points)
    scaler = StandardScaler()
    pcd_copy = scaler.fit_transform(pcd_copy)
    pcd.points = o3d.utility.Vector3dVector(pcd_copy)

    o3d.visualization.draw_geometries([pcd])

    pcd_noised = add_salt_and_pepper_noise(pcd, probability=0.05)
    # pcd_noised = add_gaussian_noise(pcd, mean=0, std_dev=0.01)
    pcd.points = o3d.utility.Vector3dVector(pcd_noised)


    o3d.visualization.draw_geometries([pcd])
    # filtering multiple times will reduce the noise significantly
    # but may cause the points distribute unevenly on the surface.
    guided_filter(pcd, 0.04, 0.1)

    o3d.visualization.draw_geometries([pcd])


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

    pcd.points = o3d.utility.Vector3dVector(points_copy)



if __name__ == '__main__':
    main()