import numpy as np
def add_noise(pcd, sigma):
    points = np.asarray(pcd.points)
    noise = sigma * np.random.randn(points.shape[0], points.shape[1])
    points += noise
    return points
def add_gaussian_noise(pcd, mean=0, std_dev=0.01):
    points = np.asarray(pcd.points)
    noise = np.random.normal(loc=mean, scale=std_dev, size=points.shape)

    points += noise
    return points
def add_salt_and_pepper_noise(pcd, probability=0.05):
    point_cloud = np.asarray(pcd.points)
    noisy_point_cloud = np.copy(point_cloud)

    # Generate random indices for salt noise (maximum value)
    num_salt = np.ceil(probability * point_cloud.shape[0])
    salt_indices = np.random.randint(0, point_cloud.shape[0], size=int(num_salt))

    # Set points to maximum value at salt indices
    noisy_point_cloud[salt_indices] = np.max(point_cloud)

    # Generate random indices for pepper noise (minimum value)
    num_pepper = np.ceil(probability * point_cloud.shape[0])
    pepper_indices = np.random.randint(0, point_cloud.shape[0], size=int(num_pepper))

    # Set points to minimum value at pepper indices
    noisy_point_cloud[pepper_indices] = np.min(point_cloud)

    return noisy_point_cloud