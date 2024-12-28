import numpy as np
import copy
import open3d as o3d


# if __name__ == "_main_":
pcd = o3d.io.read_point_cloud("dataset/45Deg/tangan_atas/cropped_obj_000000.pcd")
with open("Output.txt", "w") as text_file:
   for pts in pcd.points:
      # text_file.write("{'x': %s".pts[0]."},")
      print("{'x':",pts[0],",'y':",pts[1],",'z':",pts[2],"},", file=text_file)
   # print(np.array(pts))