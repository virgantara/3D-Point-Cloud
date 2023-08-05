import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import open3d as o3d
import time

def cartesian_to_spherical(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (x,y,z) to (phi, theta, r) coordinates"""
    list_r = np.linalg.norm(xyz, axis=1)
    list_phi = np.arcsin(xyz[:, 1]/list_r)
    list_theta = np.arctan2(xyz[:, 0], xyz[:, 2])
    return list_phi, list_theta, list_r


def spherical_angles_to_pixels(list_phi: np.ndarray, list_theta: np.ndarray,
                               height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (phi, theta) to (u, v) coordinates in the spherical image (in pixels)"""
    list_u = height * (0.5 + list_phi / np.pi)
    list_v = width * (0.5 + list_theta / (2*np.pi))

    list_u = np.floor(list_u).astype(int)
    list_v = np.floor(list_v).astype(int)

    return list_u, list_v

def render_pcd(xyz: np.ndarray, rgb: np.ndarray,
               img_360_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Render a spherical image from a colored pointcloud"""
    img_360_w = 2*img_360_h

    list_phi, list_theta, list_r = cartesian_to_spherical(xyz)
    list_u, list_v = spherical_angles_to_pixels(list_phi, list_theta,
                                                img_360_h, img_360_w)

    # Sort depths in descending order
    sorted_indices = np.argsort(-list_r, axis=None)
    list_u = list_u[sorted_indices]
    list_v = list_v[sorted_indices]
    list_r = list_r[sorted_indices]
    rgb = rgb[sorted_indices]

    # Render depthmap
    depthmap_360 = np.full((img_360_h, img_360_w), dtype=np.float32,
                           fill_value=-1)
    depthmap_360[list_u, list_v] = list_r

    # Render colored image
    img_360 = np.zeros((img_360_h, img_360_w, 3), dtype=np.uint8)
    img_360[list_u, list_v] = rgb

    return img_360, depthmap_360

my_pcd_path = "dataset/45Deg/tangan_atas/cropped_obj_000018.pcd"
o3d_pcd = o3d.io.read_point_cloud(my_pcd_path)
xyz = np.asarray(o3d_pcd.points)
rgb = (np.asarray(o3d_pcd.colors) * 255.0).astype(np.uint8)

# Render Pointcloud
start_time = time.perf_counter()
output_height = 200
img_360, depthmap_360 = render_pcd(xyz, rgb, output_height)
end_time = time.perf_counter()
print(f"--- Render Pcd: {(end_time - start_time):.3f} s ---")

plt.figure()
plt.matshow(img_360)
plt.matshow(depthmap_360)
plt.show()