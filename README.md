# This project is about Point Cloud Data exploration from LiDAR only. #

### Label notes ###
```code

1. (column 0). There are 9 types of objects. The corresponding counts are tabulated below. See python/label_info.py
    Car 	DontCare 	Pedestrian 	Van 	Cyclist 	Truck 	Misc 	Tram 	Person_sitting
    28742 	11295 	4487 	2914 	1627 	1094 	973 	511 	222

2.  (column 1). Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries
3.  (column 2). Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
4.  (column 3). Observation angle of object, ranging [-pi..pi]
5.  (column 4-7). 2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates
6.  (column 8-10). 3D object dimensions: height, width, length (in meters)
7.  (column 11-13). The center location x, y, z of the 3D object in camera coordinates (in meters)
8.  (column 14). Rotation ry around Y-axis in camera coordinates [-pi..pi]

```