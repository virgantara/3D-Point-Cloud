import numpy as np
import open3d as o3d

def read_pcd_file(file_path):
    """
    Read a PCD file and convert it to an Open3D point cloud.

    Parameters:
        file_path (str): Path to the PCD file.

    Returns:
        open3d.geometry.PointCloud: An Open3D point cloud object.
    """
    point_cloud = o3d.io.read_point_cloud(file_path)
    return point_cloud
def save_xyz_file(file_path, point_cloud):
    """
    Save an Open3D point cloud to an .xyz file.

    Parameters:
        file_path (str): Path to the .xyz file.
        point_cloud (open3d.geometry.PointCloud): An Open3D point cloud object.
    """
    o3d.io.write_point_cloud(file_path, point_cloud, write_ascii=True)

pcd_file_path = 'dataset/HumanOnly/tangan_atas/tangan_atas_cropped_obj_000018.pcd'
xyz_file_path = 'sample/input.xyz'

# Read the PCD file and extract (x, y, z) coordinates
point_cloud = read_pcd_file(pcd_file_path)

# Save the (x, y, z) coordinates to the .xyz file
save_xyz_file(xyz_file_path, point_cloud)