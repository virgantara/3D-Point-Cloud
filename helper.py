from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d


def draw_bounding_box(bbox_min, bbox_max):
    points = [
        bbox_min,
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
        bbox_max
    ]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    bbox = line_set
    return bbox

def load_pc_from_pcd(pcd_path):
    """Load PointCloud data from pcd file."""
    p = pcl.load(pcd_path)
    return np.array(list(p), dtype=np.float32)

def load_pc_from_bin(bin_path):
    """Load PointCloud data from pcd file."""
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj
# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


# ==============================================================================
#                                                          BIRDS_EYE_POINT_CLOUD
# ==============================================================================
def birds_eye_point_cloud(points,
                          side_range=(-10, 10),
                          fwd_range=(-10,10),
                          res=0.1,
                          min_height = -2.73,
                          max_height = 1.27,
                          saveto=None):
    """ Creates an 2D birds eye view representation of the point cloud data.
        You can optionally save the image to specified filename.
    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size.
        min_height:  (float)(default=-2.73)
                    Used to truncate height values to this minumum height
                    relative to the sensor (in metres).
                    The default is set to -2.73, which is 1 metre below a flat
                    road surface given the configuration in the kitti dataset.
        max_height: (float)(default=1.27)
                    Used to truncate height values to this maximum height
                    relative to the sensor (in metres).
                    The default is set to 1.27, which is 3m above a flat road
                    surface given the configuration in the kitti dataset.
        saveto:     (str or None)(default=None)
                    Filename to save the image as.
                    If None, then it just displays the image.
    """
    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    # r_lidar = points[:, 3]  # Reflectance

    # INDICES FILTER - of values within the desired rectangle
    # Note left side is positive y axis in LIDAR coordinates
    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff,ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices]/res).astype(np.int32) # x axis is -y in LIDAR
    y_img = (x_lidar[indices]/res).astype(np.int32)  # y axis is -x in LIDAR
                                                     # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0]/res))
    y_img -= int(np.floor(fwd_range[0]/res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a = z_lidar[indices],
                           a_min=min_height,
                           a_max=max_height)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values  = scale_to_255(pixel_values, min=min_height, max=max_height)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0])/res)
    y_max = int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[-y_img, x_img] = pixel_values # -y because images start from top left
    return im
    # Convert from numpy array to a PIL image
    im = Image.fromarray(im)

    # SAVE THE IMAGE
    if saveto is not None:
        im.save(saveto)
    else:
        im.show()

def point_cloud_to_panorama_reflectance(points,
                            v_res=0.42,
                            h_res = 0.35,
                            v_fov = (-24.9, 2.0),
                            r_range = (0,1),
                            y_fudge=3
                            ):
    """ Takes point cloud data as input and creates a 360 degree panoramic
        image, returned as a numpy array.
    Args:
        points: (np array)
            The numpy array containing the point cloud. .
            The shape should be at least Nx3 (allowing for more columns)
            - Where N is the number of points, and
            - each point is specified by at least 3 values (x, y, z)
        v_res: (float)
            vertical angular resolution in degrees. This will influence the
            height of the output image.
        h_res: (float)
            horizontal angular resolution in degrees. This will influence
            the width of the output image.
        v_fov: (tuple of two floats)
            Field of view in degrees (-min_negative_angle, max_positive_angle)
        d_range: (tuple of two floats) (default = (0,100))
            Used for clipping distance values to be within a min and max range.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical image height do not match the actual data.
    Returns:
        A numpy array representing a 360 degree panoramic image of the point
        cloud.
    """
    # Projecting to 2D
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    r_points = points[:, 3]

    # d_points = np.sqrt(x_points ** 2 + y_points ** 2)  # map distance relative to origin
    # print(np.min(r_points), np.max(r_points))
    # print(np.min(d_points), np.max(d_points))
    d_points = np.sqrt(x_points**2 + y_points**2 + z_points**2) # abs distance

    # We use map distance, because otherwise it would not project onto a cylinder,
    # instead, it would map onto a segment of slice of a sphere.

    # RESOLUTION AND FIELD OF VIEW SETTINGS
    v_fov_total = -v_fov[0] + v_fov[1]

    # CONVERT TO RADIANS
    v_res_rad = v_res * (np.pi / 180)
    h_res_rad = h_res * (np.pi / 180)

    # MAPPING TO CYLINDER
    x_img = np.arctan2(y_points, x_points) / h_res_rad
    y_img = -(np.arctan2(z_points, d_points) / v_res_rad)

    # THEORETICAL MAX HEIGHT FOR IMAGE
    d_plane = (v_fov_total/v_res) / (v_fov_total* (np.pi / 180))
    h_below = d_plane * np.tan(-v_fov[0]* (np.pi / 180))
    h_above = d_plane * np.tan(v_fov[1] * (np.pi / 180))
    y_max = int(np.ceil(h_below+h_above + y_fudge))

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2
    x_img = np.trunc(-x_img - x_min).astype(np.int32)
    x_max = int(np.ceil(360.0 / h_res))

    y_min = -((v_fov[1] / v_res) + y_fudge)
    y_img = np.trunc(y_img - y_min).astype(np.int32)

    # CLIP DISTANCES
    r_points = np.clip(r_points, a_min=r_range[0], a_max=r_range[1])

    # CONVERT TO IMAGE ARRAY
    img = np.zeros([y_max + 1, x_max + 1], dtype=np.uint8)
    img[y_img, x_img] = scale_to_255(r_points, min=r_range[0], max=r_range[1])

    return img

def point_cloud_to_panorama(points,
                            v_res=0.42,
                            h_res = 0.35,
                            v_fov = (-24.9, 2.0),
                            d_range = (0,100),
                            y_fudge=3
                            ):
    """ Takes point cloud data as input and creates a 360 degree panoramic
        image, returned as a numpy array.
    Args:
        points: (np array)
            The numpy array containing the point cloud. .
            The shape should be at least Nx3 (allowing for more columns)
            - Where N is the number of points, and
            - each point is specified by at least 3 values (x, y, z)
        v_res: (float)
            vertical angular resolution in degrees. This will influence the
            height of the output image.
        h_res: (float)
            horizontal angular resolution in degrees. This will influence
            the width of the output image.
        v_fov: (tuple of two floats)
            Field of view in degrees (-min_negative_angle, max_positive_angle)
        d_range: (tuple of two floats) (default = (0,100))
            Used for clipping distance values to be within a min and max range.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical image height do not match the actual data.
    Returns:
        A numpy array representing a 360 degree panoramic image of the point
        cloud.
    """
    # Projecting to 2D
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    r_points = points[:, 3]

    d_points = np.sqrt(x_points ** 2 + y_points ** 2)  # map distance relative to origin
    # print(np.min(r_points), np.max(r_points))
    # print(np.min(d_points), np.max(d_points))
    #d_points = np.sqrt(x_points**2 + y_points**2 + z_points**2) # abs distance

    # We use map distance, because otherwise it would not project onto a cylinder,
    # instead, it would map onto a segment of slice of a sphere.

    # RESOLUTION AND FIELD OF VIEW SETTINGS
    v_fov_total = -v_fov[0] + v_fov[1]

    # CONVERT TO RADIANS
    v_res_rad = v_res * (np.pi / 180)
    h_res_rad = h_res * (np.pi / 180)

    # MAPPING TO CYLINDER
    x_img = np.arctan2(y_points, x_points) / h_res_rad
    y_img = -(np.arctan2(z_points, d_points) / v_res_rad)

    # THEORETICAL MAX HEIGHT FOR IMAGE
    d_plane = (v_fov_total/v_res) / (v_fov_total* (np.pi / 180))
    h_below = d_plane * np.tan(-v_fov[0]* (np.pi / 180))
    h_above = d_plane * np.tan(v_fov[1] * (np.pi / 180))
    y_max = int(np.ceil(h_below+h_above + y_fudge))

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2
    x_img = np.trunc(-x_img - x_min).astype(np.int32)
    x_max = int(np.ceil(360.0 / h_res))

    y_min = -((v_fov[1] / v_res) + y_fudge)
    y_img = np.trunc(y_img - y_min).astype(np.int32)

    # CLIP DISTANCES
    d_points = np.clip(d_points, a_min=d_range[0], a_max=d_range[1])

    # CONVERT TO IMAGE ARRAY
    img = np.zeros([y_max + 1, x_max + 1], dtype=np.uint8)
    img[y_img, x_img] = scale_to_255(d_points, min=d_range[0], max=d_range[1])

    return img

def lidar_to_2d_front_view(points,
                           v_res,
                           h_res,
                           v_fov,
                           val="depth",
                           cmap="jet",
                           saveto=None,
                           y_fudge=0.0
                           ):
    """ Takes points in 3D space from LIDAR data and projects them to a 2D
        "front view" image, and saves that image.
    Args:
        points: (np array)
            The numpy array containing the lidar points.
            The shape should be Nx4
            - Where N is the number of points, and
            - each point is specified by 4 values (x, y, z, reflectance)
        v_res: (float)
            vertical resolution of the lidar sensor used.
        h_res: (float)
            horizontal resolution of the lidar sensor used.
        v_fov: (tuple of two floats)
            (minimum_negative_angle, max_positive_angle)
        val: (str)
            What value to use to encode the points that get plotted.
            One of {"depth", "height", "reflectance"}
        cmap: (str)
            Color map to use to color code the `val` values.
            NOTE: Must be a value accepted by matplotlib's scatter function
            Examples: "jet", "gray"
        saveto: (str or None)
            If a string is provided, it saves the image as this filename.
            If None, then it just shows the image.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical range do not match the actual data.
            For a Velodyne HDL 64E, set this value to 5.
    """

    # DUMMY PROOFING
    assert len(v_fov) ==2, "v_fov must be list/tuple of length 2"
    assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"
    assert val in {"depth", "height", "reflectance"}, \
        'val must be one of {"depth", "height", "reflectance"}'


    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    r_lidar = points[:, 3] # Reflectance
    # Distance relative to origin when looked from top
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]

    # Convert to Radians
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)

    # PROJECT INTO IMAGE COORDINATES
    x_img = np.arctan2(-y_lidar, x_lidar)/ h_res_rad
    y_img = np.arctan2(z_lidar, d_lidar)/ v_res_rad

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2  # Theoretical min x value based on sensor specs
    x_img -= x_min              # Shift
    x_max = 360.0 / h_res       # Theoretical max x value after shifting

    y_min = v_fov[0] / v_res    # theoretical min y value based on sensor specs
    y_img -= y_min              # Shift
    y_max = v_fov_total / v_res # Theoretical max x value after shifting

    y_max += y_fudge            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.

    # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
    if val == "reflectance":
        pixel_values = r_lidar
    elif val == "height":
        pixel_values = z_lidar
    else:
        pixel_values = -d_lidar

    # PLOT THE IMAGE
    cmap = "jet"            # Color map to use
    dpi = 100               # Image resolution
    fig, ax = plt.subplots(figsize=(x_max/dpi, y_max/dpi), dpi=dpi)
    ax.scatter(x_img,y_img, s=1, c=pixel_values, linewidths=0, alpha=1, cmap=cmap)
    ax.set_facecolor((0, 0, 0)) # Set regions with no points to black
    ax.axis('scaled')              # {equal, scaled}
    ax.xaxis.set_visible(False)    # Do not draw axis tick marks
    ax.yaxis.set_visible(False)    # Do not draw axis tick marks
    plt.xlim([0, x_max])   # prevent drawing empty space outside of horizontal FOV
    plt.ylim([0, y_max])   # prevent drawing empty space outside of vertical FOV

    if saveto is not None:
        fig.savefig(saveto, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    else:
        fig.show()


def read_variable(param_path, variabel, baris, kolom):
    fileku = open(param_path, 'r')
    content = fileku.readlines()
    df = []

    for line in content:

        if line.startswith(variabel+":"):
            ln = line.split(variabel+":")
            df = np.fromstring(ln[1], sep=" ")
            break

    if len(df) != 0:
        return df.reshape(baris,kolom)
    else:
        return df


def load_calibration_rigid(param_path):
    R = read_variable(param_path, 'R', 3, 3)
    T = read_variable(param_path, 'T', 3, 1)
    stack_R_T = np.hstack([R, T])
    stack_R_T = np.vstack([stack_R_T, np.array([0, 0, 0, 1])])
    return stack_R_T

def load_calibration_cam_to_cam(param_path):
    calib = {}

    calib['cornerdist'] = read_variable(param_path, 'corner_dist', 1, 1)

    list_s_rect = {}
    list_r_rect = {}
    list_p_rect = {}
    for cam in range(1,101):
        list_s_rect[cam - 1] = []
        list_r_rect[cam - 1] = []
        list_p_rect[cam - 1] = []
        S_ = read_variable(param_path, 'S_'+format(cam-1,'02'), 1, 2)
        K_ = read_variable(param_path, 'K_' + format(cam - 1, '02'), 3, 3)
        D_ = read_variable(param_path, 'D_' + format(cam - 1, '02'), 1, 5)
        R_ = read_variable(param_path, 'R_' + format(cam - 1, '02'), 3, 3)
        T_ = read_variable(param_path, 'T_' + format(cam - 1, '02'), 3, 1)
        S_rect_ = read_variable(param_path, 'S_rect_' + format(cam - 1, '02'), 1, 2)
        R_rect_ = read_variable(param_path, 'R_rect_' + format(cam - 1, '02'), 3, 3)
        P_rect_ = read_variable(param_path, 'P_rect_' + format(cam - 1, '02'), 3, 4)
        # print("S_",'S_'+format(cam-1,'02'))
        if len(S_) == 0 or len(K_) == 0 or len(D_) == 0 or len(R_) == 0 or len(T_) == 0:
            break

        if len(S_rect_) >= 0 and len(R_rect_) > 0 and len(P_rect_) > 0:
            list_s_rect[cam - 1] = S_rect_
            list_r_rect[cam - 1] = R_rect_
            list_p_rect[cam - 1] = P_rect_
            break

        calib['S_Rect'] = list_s_rect
        calib['R_Rect'] = list_r_rect
        calib['P_Rect'] = list_p_rect

    return calib
# def read_variabel(nama_var, file_path, baris, kolom):
#     fileku = open(file_path, 'r')
#     content = fileku.readlines()
#     ex_rotations = []
#     ex_translations = []
#     for line in content:
#         if line.startswith("R:"):
#             ln = line.split("R:")
#             par = np.fromstring(ln[1], sep=" ")
#             par = np.append(par, [0, 0, 0])
#             par = par.reshape(4, 3)
#             ex_rotations = par

def read_data(file_path):
    pc_list = []
    fileku = open(file_path, 'r')
    #    with open(file_path,'rb') as f:
    content = fileku.readlines()
    count = 0
    for line in content:
        count = + 1
        pt = np.fromstring(line, sep=" ")

        pc_list.append([pt[0], pt[1], pt[2], pt[3]])

    return np.asarray(pc_list, dtype=np.float32)

def read_extrinsic(param_path):
    fileku = open(param_path, 'r')
    content = fileku.readlines()
    ex_rotations = []
    ex_translations = []
    for line in content:
        if line.startswith("R:"):
            ln = line.split("R:")
            par = np.fromstring(ln[1], sep=" ")
            par = np.append(par, [0, 0, 0])
            par = par.reshape(4, 3)
            ex_rotations = par
        elif line.startswith("T:"):
            ln = line.split("T:")
            par = np.fromstring(ln[1], sep=" ")
            ex_translations = np.append(par, 1).reshape(-1, 1)

    return np.hstack((ex_rotations, ex_translations))

def read_intrinsic(param_path):
    fileku = open(param_path, 'r')
    content = fileku.readlines()
    df = []
    dc = []
    for line in content:
        if line.startswith("delta_f:"):
            ln = line.split("delta_f:")
            df = np.fromstring(ln[1], sep=" ")
        elif line.startswith("delta_c:"):
            ln = line.split("delta_c:")
            dc = np.fromstring(ln[1], sep=" ")

    mat_normal = np.hstack((np.identity(3), np.zeros(3).reshape(-1, 1)))
    fx, fy = df
    cx, cy = dc

    mat_f_c = [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ]

    return np.matmul(mat_f_c, mat_normal)