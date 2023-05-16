import numpy as np
import open3d as o3d

class TimeSeriesVoxelGrid:
    def __init__(self, width, height, depth, num_steps, voxel_size=1.0):
        self.width = width
        self.height = height
        self.depth = depth
        self.num_steps = num_steps
        self.voxel_size = voxel_size
        self.grid = np.zeros((width, height, depth, num_steps), dtype=bool)

    def set_voxel(self, x, y, z, step):
        self.grid[x, y, z, step] = True

    def clear_voxel(self, x, y, z, step):
        self.grid[x, y, z, step] = False

    def is_voxel_set(self, x, y, z, step):
        return self.grid[x, y, z, step]

    def visualize(self, step):
        points = []
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    if self.is_voxel_set(x, y, z, step):
                        voxel_center = np.array([x, y, z]) * self.voxel_size
                        points.append(voxel_center)
        if len(points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            o3d.visualization.draw_geometries([pcd])

# Example usage
grid = TimeSeriesVoxelGrid(10, 10, 10, 5)
grid.set_voxel(5, 5, 5, 2)
grid.visualize(2)