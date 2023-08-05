from VoxelGrid import VoxelGrid
import glob
import h5py
import numpy as np
import tqdm



data_path = '../ModelNet40-C/data/modelnet40_ply_hdf5_2048'

def points3d_to_voxel(points, voxel_size=16):
    voxel_grid = VoxelGrid(points, x_y_z=[voxel_size, voxel_size, voxel_size])
    voxel_2d = np.array(voxel_grid.vector[:, :, :])
    # voxel_2d = voxel_2d.reshape(-1)
    voxel_final = voxel_2d.astype('float64')
    return voxel_final

def convert_modelnet40_to_voxel(partition):
    all_data = []
    all_label = []
    for h5_name in tqdm.tqdm(glob.glob(data_path + '/ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()

        for pts, lb in zip(data, label):
            vxl = points3d_to_voxel(pts)
            all_data.append(vxl)
            all_label.append(lb)


    with h5py.File('modelnet40_voxel_'+partition+'.h5', 'w') as f:
        f.create_dataset("data", data=np.asarray(all_data))
        f.create_dataset("label", data=np.asarray(all_label))

convert_modelnet40_to_voxel(partition='train')
convert_modelnet40_to_voxel(partition='test')