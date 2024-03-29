import os.path
from pathlib import Path
import open3d as o3d
import numpy as np
import re
from noise_generator import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from helper import *
# L-CAS
# Column 1: 	  	category (pedestrian or group)
# Column 2-4: 	  	centroid (x-y-z)
# Column 5-7: 	  	minimum bounds (x-y-z)
# Column 8-10: 	  	maximum bounds (x-y-z)
# Column 11: 	  	visibility (0 = visible, 1 = partially visible)

def generate_noisy_pcd():
    i = 0
    for f in os.listdir("sample/clean"):
        dataset_path = os.path.join("sample", "clean", f)  # sample/clean/input_point_clean_net_0.xyz"
        # dataset_path = "dataset/45Deg_merged/standing/18.pcd"
        # dataset_path = ""
        # pcd_file_path = os.path.join(dataset_path,"tangan_atas","cropped_obj_000010.pcd")

        pcd = o3d.io.read_point_cloud(dataset_path)

        scaler = MinMaxScaler()
        pcd_copy = scaler.fit_transform(pcd.points)
        xyz_pts = np.asarray(pcd.points)

        pcd.points = o3d.utility.Vector3dVector(pcd_copy)
        pcd.paint_uniform_color(np.array([0, 0, 1]))
        # viewer.add_geometry(pcd)
        # o3d.visualization.draw_geometries([pcd])
        # print("Before",np.asarray(pcd.points).shape)
        pcd_noisy = o3d.geometry.PointCloud()
        pcd_noisy.points = o3d.utility.Vector3dVector(pcd_copy)
        std_dev = 0.75
        scale_parameter = 0.9
        #### Gaussian Noise #####
        pcd_noised = add_gaussian_noise(pcd_noisy, mean=0.01, std_dev=std_dev)

        pcd_noisy.points = o3d.utility.Vector3dVector(pcd_noised)
        noisy_color = np.array([1, 0, 0])
        pcd_noisy.paint_uniform_color(noisy_color)
        # viewer.add_geometry(pcd_noisy)

        fname = Path(dataset_path).stem
        print("Writing Gaussian noise...")
        gauss_path = "sample/noisy/gaussian_"+str(std_dev)
        if not os.path.exists(gauss_path):
            os.makedirs(gauss_path)


        o3d.io.write_point_cloud(gauss_path+"/" + str(i) + "_gaussian_"+str(std_dev)+".xyz", pcd_noisy, write_ascii=True)

        #### Laplacian Noise ######
        # pcd_noisy = o3d.geometry.PointCloud()
        # pcd_noisy.points = o3d.utility.Vector3dVector(pcd_copy)
        #
        #
        # pcd_noised = add_laplacian_noise(pcd_noisy, scale_parameter=scale_parameter)
        #
        # pcd_noisy.points = o3d.utility.Vector3dVector(pcd_noised)
        # noisy_color = np.array([1, 0, 0])
        # pcd_noisy.paint_uniform_color(noisy_color)
        #
        # fname = Path(dataset_path).stem
        #
        # print("Writing Laplacian noise...")
        # laplace_path = "sample/noisy/laplace_" + str(scale_parameter)
        # if not os.path.exists(laplace_path):
        #     os.makedirs(laplace_path)
        # o3d.io.write_point_cloud(laplace_path+"/"+str(i)+"_laplacian_" + str(scale_parameter) + ".xyz",
        #                          pcd_noisy, write_ascii=True)

        i += 1

# o3d.visualization.draw_geometries([pcd])
# print("Before",np.asarray(pcd.points).shape)
# pcd_noisy = o3d.geometry.PointCloud()
# pcd_noisy.points = o3d.utility.Vector3dVector(pcd_copy)
# pcd_noised = add_gaussian_noise(pcd_noisy, mean=0.01, std_dev=0.015)
#
# pcd_noisy.points = o3d.utility.Vector3dVector(pcd_noised)
# noisy_color = np.array([1, 0, 0])
# pcd_noisy.paint_uniform_color(noisy_color)
# viewer.add_geometry(pcd_noisy)

def view_pcd(path):
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    # dataset_path = "sample/noisy/input_point_clean_net_10_gaussian.xyz"
    # dataset_path = "sample/clean/input_point_clean_net_10.xyz"
    # dataset_path = "sample/noisy/input_point_clean_net_10_gaussian.xyz"
    dataset_path = path
    # dataset_path = ""
    # pcd_file_path = os.path.join(dataset_path,"tangan_atas","cropped_obj_000010.pcd")

    pcd = o3d.io.read_point_cloud(dataset_path)


    scaler = MinMaxScaler()
    pcd_copy = scaler.fit_transform(pcd.points)

    pcd.points = o3d.utility.Vector3dVector(pcd_copy)
    pcd.paint_uniform_color(np.array([0, 0, 1]))
    viewer.add_geometry(pcd)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = False
    opt.line_width = 0.1
    opt.background_color = np.asarray([1, 1,1])
    viewer.run()
    viewer.destroy_window()

view_pcd(path="sample/denoised_scorenet/output_score_denoise35.xyz")
# generate_noisy_pcd()