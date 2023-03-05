import numpy as np
import copy
import open3d as o3d


if __name__ == "_main_":
   pcd = o3d.io.read_point_cloud("path_to_pcd.pcd")
   vis = o3d.visualization.VisualizerWithEditing(-1, False, "")
   vis.create_window()
   vis.add_geometry(pcd)
   vis.run()
   vis.destroy_window()
   cropped_geometry= vis.get_cropped_geometry()
   o3d.io.write_point_cloud("path_to_output_pcd.pcd", cropped_geometry)