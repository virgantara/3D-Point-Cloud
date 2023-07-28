#libraries used
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

dataset_path = "/home/virgantara/PythonProjects/LiDAROuster/20230301/PCD_Cropped/45Deg/duduk_depan/cropped_obj_000010.pcd"


pcd = o3d.io.read_point_cloud(dataset_path)

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
pcd.paint_uniform_color([0.6, 0.6, 0.6])
# o3d.visualization.draw_geometries([pcd]) #Works only outside Jupyter/Colab

labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([pcd])