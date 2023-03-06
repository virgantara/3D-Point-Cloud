import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
import open3d as o3d
import trimesh
from pathlib import Path
import numpy as np

dirname = "modelnet10_pcd"
DATA_DIR = "/media/virgantara/DATA1/Penelitian/Datasets/ModelNet10"
SAVE_PATH = "/home/virgantara/PythonProjects/PointCloud/dataset/"+dirname
print(DATA_DIR)

folders = sorted(glob.glob(os.path.join(DATA_DIR, "*")))
print (folders)
train_points = []
train_labels = []
class_map={}

#
# list_pcd = []
# for f in glob.glob(DATA_DIR+"/*"):
#     out_path = SAVE_PATH
#
#     pcd = bin2pcd(f, out_path)
#     f_name = Path(f).stem
#
#     out_path = out_path + "/" + f_name+".pcd"
#     print("Writing : ", out_path)
#     o3d.io.write_point_cloud(out_path, pcd)

for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        # class_map[i] = folder.split("\\")[-1]
        class_map = Path(folder).stem
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))
        #print(train_files)
        for f in train_files:
                #print(f)
                fname = Path(f).stem
                # nama_train = os.path.splitext(f.split("\\")[-1])[0]
                xyz=trimesh.load(f).sample(2048)
                print(np.array(xyz).shape)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                if not os.path.exists(SAVE_PATH):
                        os.mkdir(SAVE_PATH)

                if not os.path.exists(SAVE_PATH + "/"+class_map):
                        os.mkdir(SAVE_PATH + "/"+class_map)

                if not os.path.exists(SAVE_PATH + "/"+ class_map + "/train/"):
                        os.mkdir(SAVE_PATH + "/"+ class_map + "/train/")

                # save_path = SAVE_PATH + "/"+ class_map + "/train/"+ fname+".pcd"
                # print("Writing Train:", save_path)
                # o3d.io.write_point_cloud(save_path, pcd)



        for f in test_files:
                fname = Path(f).stem
                # nama_train = os.path.splitext(f.split("\\")[-1])[0]
                xyz = trimesh.load(f).sample(2048)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                if not os.path.exists(SAVE_PATH):
                        os.mkdir(SAVE_PATH)

                if not os.path.exists(SAVE_PATH + "/" + class_map):
                        os.mkdir(SAVE_PATH + "/" + class_map)

                if not os.path.exists(SAVE_PATH + "/" + class_map + "/test/"):
                        os.mkdir(SAVE_PATH + "/" + class_map + "/test/")

                # save_path =os.path.join(SAVE_PATH,class_map + "/test/"+ fname+".pcd")
                # print("Writing Test:", save_path)
                # o3d.io.write_point_cloud(save_path, pcd)
        

# print(class_map)
#print(train_files)


