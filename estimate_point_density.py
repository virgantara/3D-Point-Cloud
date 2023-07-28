import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
def normalize_point_cloud(point_cloud):
    # Find the minimum and maximum values for X, Y, and Z coordinates
    min_x = min(point[0] for point in point_cloud)
    max_x = max(point[0] for point in point_cloud)
    min_y = min(point[1] for point in point_cloud)
    max_y = max(point[1] for point in point_cloud)
    min_z = min(point[2] for point in point_cloud)
    max_z = max(point[2] for point in point_cloud)

    # Calculate the ranges for X, Y, and Z coordinates
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z

    normalized_point_cloud = []

    for point in point_cloud:
        x_normalized = (point[0] - min_x) / range_x
        y_normalized = (point[1] - min_y) / range_y
        z_normalized = (point[2] - min_z) / range_z
        normalized_point_cloud.append((x_normalized, y_normalized, z_normalized))

    return normalized_point_cloud
def normalize_pc(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance

    return points
#RANSAC PLANE IN POINTS FITTING
def fit_plane(points, threshold_distance=0.01, max_iterations=100):
    best_plane = None
    best_num_inliers = 0

    for _ in range(max_iterations):
        # Randomly choose three points to form a plane
        random_indices = np.random.choice(len(points), 3, replace=False)
        plane_points = points[random_indices]

        # Fit a plane to the selected points using least squares
        A = np.column_stack((plane_points[:, 0], plane_points[:, 1], np.ones(3)))
        plane_params = np.linalg.lstsq(A, plane_points[:, 2], rcond=None)[0]
        print(plane_params[:3], plane_params[2])

        # Calculate the distances from all points to the plane
        distances = np.abs(points.dot(plane_params[:3]) - plane_params[3])

        # Count inliers (points that lie within the threshold distance from the plane)
        num_inliers = np.sum(distances < threshold_distance)

        # Update best plane if the current model has more inliers
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_plane = plane_params

    # Extract the coefficients of the best plane (Ax + By + Cz + D = 0)
    A, B, C, D = best_plane
    return A, B, C, D

def estimate_point_cloud_density(points, roi_size):
    """
    Estimate point cloud density using Euclidean distance.

    Parameters:
        points (numpy array): 3D array representing the point cloud where each row is a point [x, y, z].
        roi_size (float): The size of the region of interest (ROI) to analyze.

    Returns:
        float: Estimated point cloud density.
    """
    # Create a Nearest Neighbors object to find the k-nearest neighbors for each point.
    k = 5  # You can adjust this value to control the number of nearest neighbors to consider.
    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)

    # Estimate density by calculating the average Euclidean distance to k-nearest neighbors for each point.
    avg_distances = []
    for point in points:
        distances, _ = nn.kneighbors([point])
        avg_distances.append(np.mean(distances))

    # Estimate density by taking the reciprocal of the average distance.
    estimated_density = 1.0 / np.mean(avg_distances)

    return estimated_density

def cluster_point_cloud(point_cloud, eps, min_samples):
    """
    Cluster point cloud using DBSCAN.

    Parameters:
        point_cloud (numpy array): 3D array representing the point cloud where each row is a point [x, y, z].
        eps (float): The maximum distance between two points for them to be considered as part of the same cluster.
        min_samples (int): The minimum number of points required to form a dense region (core point).

    Returns:
        numpy array: An array with cluster labels (-1 represents outliers).
    """
    # Standardize the point cloud data (optional but can improve clustering performance).
    scaler = StandardScaler()
    point_cloud = scaler.fit_transform(point_cloud)

    # Apply DBSCAN clustering.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(point_cloud)

    return cluster_labels

if __name__ == "__main__":
    # Generate sample point cloud data (replace this with your actual point cloud data)
    np.random.seed(42)
    num_points = 1000
    points = np.random.rand(num_points, 3)

    # Introduce a plane in the point cloud (plane equation: 2x - 3y + 4z - 5 = 0)
    points[:, 2] = (5 - 2 * points[:, 0] + 3 * points[:, 1]) / 4

    # Detect the plane in the point cloud using RANSAC
    A, B, C, D = fit_plane(points)

    print(f"Detected Plane Equation: {A:.2f}x + {B:.2f}y + {C:.2f}z + {D:.2f} = 0")