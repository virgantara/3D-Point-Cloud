import numpy as np
from scipy.spatial.distance import cdist

def chamfer_distance(point_set1, point_set2):
    """
    Calculate the Chamfer Distance between two point sets.

    Parameters:
        point_set1 (numpy array): The first set of points represented as a numpy array where each row is a point with (x, y, z) coordinates.
        point_set2 (numpy array): The second set of points represented as a numpy array where each row is a point with (x, y, z) coordinates.

    Returns:
        float: The Chamfer Distance between the two point sets.
    """
    # Calculate pairwise distances between points in both sets
    distances_set1_to_set2 = cdist(point_set1, point_set2)
    distances_set2_to_set1 = cdist(point_set2, point_set1)

    # Find the minimum distance for each point in each set
    min_distances_set1 = np.min(distances_set1_to_set2, axis=1)
    min_distances_set2 = np.min(distances_set2_to_set1, axis=1)

    # Sum up the distances from each point in each set to its nearest neighbor in the other set
    chamfer_distance = np.sum(min_distances_set1) + np.sum(min_distances_set2)

    return chamfer_distance

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

def point_cloud_mae(point_cloud1, point_cloud2):
    """
    Calculate the Mean Absolute Error (MAE) between two 3D point clouds.

    Parameters:
        point_cloud1 (numpy array): The first 3D point cloud represented as a numpy array where each row is a point with (x, y, z) coordinates.
        point_cloud2 (numpy array): The second 3D point cloud represented as a numpy array where each row is a point with (x, y, z) coordinates.

    Returns:
        float: The Mean Absolute Error (MAE) between the two point clouds.
    """
    # Ensure that both point clouds have the same number of points
    assert point_cloud1.shape == point_cloud2.shape, "Both point clouds must have the same number of points."

    # Calculate the absolute difference between corresponding points in the two point clouds
    absolute_errors = np.abs(point_cloud1 - point_cloud2)

    # Calculate the mean absolute error
    mae = np.mean(absolute_errors)

    return mae
