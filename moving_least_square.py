import numpy as np
import matplotlib.pyplot as plt

# Create a sample 3D point cloud (replace this with your own point cloud data)
num_points = 1000
x = np.random.rand(num_points)
y = np.random.rand(num_points)
z = 5 * np.sin(10 * x) + 5 * np.cos(10 * y) + np.random.normal(0, 0.1, num_points)


# Convert the data to homogeneous coordinates (x, y, z, 1)
homogeneous_coords = np.column_stack((x, y, z, np.ones_like(x)))
pcd = np.array([x, y, z])
pcd = pcd.T
print(pcd.shape)
# Parameters for Moving Least Squares
radius = 0.1  # Radius for local surface fitting

# Perform Moving Least Squares denoising
denoised_points = []
for i in range(num_points):
    # Find neighbors within the radius
    dist = np.linalg.norm(homogeneous_coords[:, :3] - homogeneous_coords[i, :3], axis=1)
    neighbors = homogeneous_coords[dist < radius]

    if len(neighbors) > 1:
        # Fit a plane to the neighbors using least squares
        A = neighbors[:, :3]
        b = neighbors[:, 3]
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]

        # Evaluate the plane at the current point
        denoised_z = np.dot(coeffs, homogeneous_coords[i, :3])
        denoised_points.append([x[i], y[i], denoised_z])

denoised_points = np.array(denoised_points)

# Access the denoised (x, y, z) coordinates
denoised_x = denoised_points[:, 0]
denoised_y = denoised_points[:, 1]
denoised_z = denoised_points[:, 2]

pcd_denoised = np.array([denoised_x, denoised_y, denoised_z])
pcd_denoised = pcd_denoised.T
print(pcd_denoised.shape)
# Now you can use denoised_x, denoised_y, and denoised_z as your denoised point cloud data

# Plot the original and denoised point clouds
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', marker='o', label='Original Point Cloud')
ax.scatter(denoised_x, denoised_y, denoised_z, c='r', marker='x', label='Denoised Point Cloud')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Moving Least Squares Denoising')
ax.legend()
plt.show()