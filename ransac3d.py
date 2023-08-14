import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(-10, 10, size=(10,))
y = np.random.uniform(-10, 10, size=(10,))
z = np.random.uniform(-10, 10, size=(10,))

# x, y = np.meshgrid(x, y)
z = x + 0.3*y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z,s=100, color='red')
plt.show()
