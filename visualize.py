#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from ouster import client
from ouster.sdk import viz
from ouster import pcap
from contextlib import closing
from more_itertools import time_limited
from datetime import datetime
import cv2
import numpy as np
from more_itertools import nth

import matplotlib.pyplot as plt
from helper import read_data
base_path = 'dataset/2011_09_26_drive_0005_extract/2011_09_26/2011_09_26_drive_0005_extract/velodyne_points/data/'
file_path = base_path+'0000000000.txt'

data_pc = read_data(file_path)

# Creating figure
# fig = plt.figure(figsize = (10, 7))
#ax = plt.axes(projection ="3d")
point_viz = viz.PointViz("Ouster Visualisasi")
viz.add_default_controls(point_viz)

x, y, z, intensitas = data_pc[:,0], data_pc[:,1], data_pc[:,2], data_pc[:,3]

ax = plt.axes(projection='3d')
r = 3
ax.set_xlim3d([-r, r])
ax.set_ylim3d([-r, r])
ax.set_zlim3d([-r, r])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(x, y, z, c=z / max(z), s=0.2)
plt.show()

# konvert data ke Open3D PointCloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(data_pc)
#
# o3d.visualization.draw_geometries([pcd])
