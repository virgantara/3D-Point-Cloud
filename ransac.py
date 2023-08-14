import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data with outliers
np.random.seed(0)
num_points = 100
inlier_ratio = 0.8
noise_std = 0.5

true_slope = 2.0
true_intercept = 1.0

x = np.linspace(0, 10, num_points)
y = true_slope * x + true_intercept + np.random.normal(0, noise_std, num_points)
outliers = np.random.choice(num_points, size=int(num_points * (1 - inlier_ratio)), replace=False)
y[outliers] += 10 * noise_std  # Adding outliers

# RANSAC parameters
num_iterations = 100
sample_size = 2
threshold = 1.0

best_model = None
best_inliers = []

for _ in range(num_iterations):
    # Randomly select a subset of points (sample)
    sample_indices = np.random.choice(num_points, size=sample_size, replace=False)

    sample_x = x[sample_indices]
    sample_y = y[sample_indices]

    # Fit a model to the sample (in this case, a linear model)
    model = np.polyfit(sample_x, sample_y, 1)
    model_line = np.poly1d(model)

    # Calculate distances from all points to the model line
    distances = np.abs(y - model_line(x))

    # Identify inliers based on the threshold
    inliers = np.where(distances < threshold)[0]
    # print("Dist",distances)

    # Check if the current model is better than the previous best model
    if len(inliers) > len(best_inliers):
        best_model = model
        best_inliers = inliers

# Fit the final model using all the inliers
final_model = np.polyfit(x[best_inliers], y[best_inliers], 1)
final_model_line = np.poly1d(final_model)

# Plot the data and the fitted lines
plt.scatter(x, y, label='Data', color='blue')
plt.plot(x, true_slope * x + true_intercept, label='True Line', color='green')
plt.plot(x, final_model_line(x), label='RANSAC Line', color='red')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('RANSAC Linear Regression')
plt.show()