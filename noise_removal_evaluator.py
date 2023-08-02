import numpy as np
from scipy.spatial.distance import cdist

def hausdorff_distance(point_cloud1, point_cloud2):
    """
    Calculate the Hausdorff Distance between two 3D point clouds.

    Parameters:
        point_cloud1 (numpy array): The first 3D point cloud represented as a numpy array where each row is a point with (x, y, z) coordinates.
        point_cloud2 (numpy array): The second 3D point cloud represented as a numpy array where each row is a point with (x, y, z) coordinates.

    Returns:
        float: The Hausdorff Distance between the two point clouds.
    """
    # Calculate pairwise distances between points in both point clouds
    distances_1_to_2 = cdist(point_cloud1, point_cloud2)
    distances_2_to_1 = cdist(point_cloud2, point_cloud1)

    # Find the maximum distance from point_cloud1 to point_cloud2
    max_distance_1_to_2 = np.max(np.min(distances_1_to_2, axis=1))

    # Find the maximum distance from point_cloud2 to point_cloud1
    max_distance_2_to_1 = np.max(np.min(distances_2_to_1, axis=1))

    # The Hausdorff Distance is the maximum of these two distances
    hausdorff_distance = max(max_distance_1_to_2, max_distance_2_to_1)

    return hausdorff_distance

def point_to_point_distance(point_cloud1, point_cloud2):
    """
    Calculate the Point-to-Point Distance between two 3D point clouds.

    Parameters:
        point_cloud1 (numpy array): The first 3D point cloud represented as a numpy array where each row is a point with (x, y, z) coordinates.
        point_cloud2 (numpy array): The second 3D point cloud represented as a numpy array where each row is a point with (x, y, z) coordinates.

    Returns:
        numpy array: A 1D array containing the point-to-point distances between corresponding points in the two point clouds.
    """
    # Calculate pairwise distances between points in both point clouds
    distances = cdist(point_cloud1, point_cloud2)

    # Get the minimum distance for each point in point_cloud1 to its corresponding point in point_cloud2
    point_to_point_distances = np.min(distances, axis=1)

    return point_to_point_distances

def rmse(point_cloud1, point_cloud2):
    """
    Calculate the Root Mean Squared Error (RMSE) between two 3D point clouds.

    Parameters:
        point_cloud1 (numpy array): The first 3D point cloud represented as a numpy array where each row is a point with (x, y, z) coordinates.
        point_cloud2 (numpy array): The second 3D point cloud represented as a numpy array where each row is a point with (x, y, z) coordinates.

    Returns:
        float: The Root Mean Squared Error (RMSE) between the two point clouds.
    """
    # Calculate pairwise distances between points in both point clouds
    distances = cdist(point_cloud1, point_cloud2)

    # Compute the RMSE
    squared_distances = np.square(distances)
    mean_squared_distance = np.mean(squared_distances)

    rmse = np.sqrt(mean_squared_distance)

    return rmse