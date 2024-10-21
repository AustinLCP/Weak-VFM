import os

import numpy as np
import pickle
import cv2
from tqdm import tqdm

from dataloader import calib_parse


def project_3d_box_to_2d_box(bounding_box, cam_to_cam, velo_to_cam):

    P2 = None
    R = None
    T = None
    with open(cam_to_cam, 'r') as f1:
        lines = f1.readlines()
        line = lines[-9]
        P2 = line.strip().split(' ')[1:]
        P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
    intrinsic_matrix = P2[:3, :3]

    with open(velo_to_cam, 'r') as f2:
        lines = f2.readlines()
        line_R = lines[1]
        R = line_R.strip().split(' ')[1:]
        R = np.array(R, dtype=np.float32).reshape(3, 3)

        line_T = lines[2]
        T = line_T.strip().split(' ')[1:]
        T = np.array(T, dtype=np.float32).reshape(3, 1)
    extrinsic_matrix = np.hstack((R, T.reshape(-1, 1)))

    projection_matrix = intrinsic_matrix @ extrinsic_matrix


    # 3d bbox
    min_coords = bounding_box["min_coords"]
    max_coords = bounding_box["max_coords"]
    corners = np.array([[min_coords[0], min_coords[1], min_coords[2],1],
                        [max_coords[0], min_coords[1], min_coords[2],1],
                        [max_coords[0], max_coords[1], min_coords[2],1],
                        [min_coords[0], max_coords[1], min_coords[2],1],
                        [min_coords[0], min_coords[1], max_coords[2],1],
                        [max_coords[0], min_coords[1], max_coords[2],1],
                        [max_coords[0], max_coords[1], max_coords[2],1],
                        [min_coords[0], max_coords[1], max_coords[2],1]])

    corners_2d = []
    for corner in corners:
        corner_2d_homogeneous = projection_matrix @ corner.T
        corner_2d = corner_2d_homogeneous[:2] / corner_2d_homogeneous[2]  # 齐次坐标转换到二维坐标
        corners_2d.append(corner_2d)

    corners_2d = np.array(corners_2d) # 包含三维边界框8个角点投影到二维图像平面后得到的二维坐标的数组

    # 计算2D边界框的最小值和最大值
    min_2d = corners_2d.min(axis=0) # 所有投影点在X轴和Y轴上的最小值
    max_2d = corners_2d.max(axis=0)

    bbox_2d_coord = [min_2d[0], min_2d[1], max_2d[0], max_2d[1]]

    return bbox_2d_coord

# cam_to_cam = "data/kitti/raw_data/2011_09_26/calib_cam_to_cam.txt"
# velo_to_cam = "data/kitti/raw_data/2011_09_26/calib_velo_to_cam.txt"
# image_path = "data/kitti/raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png"
# bin_file_path = 'data/kitti/raw_data/2011_09_26/2011_09_26_drive_0001_sync/pointSAM_label/0000000000.bin'
# velo_points_path = "data/kitti/raw_data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin"

# cam_to_cam = "data/kitti/raw_data/9999_99_99/9999_99_99_drive_9999_sync/calib_cam_to_cam/000142.txt"
# velo_to_cam = "data/kitti/raw_data/9999_99_99/9999_99_99_drive_9999_sync/calib_velo_to_cam/000142.txt"
# image_path = "data/kitti/raw_data/9999_99_99/9999_99_99_drive_9999_sync/image_02/data/000142.png"
# bin_file_path = 'data/kitti/raw_data/9999_99_99/9999_99_99_drive_9999_sync/pointSAM_label/000142.bin'
# velo_points_path = "data/kitti/raw_data/9999_99_99/9999_99_99_drive_9999_sync/velodyne_points/data/000142.bin"

train_file = np.loadtxt("data/kitti/data_file/split/train_raw.txt", dtype=str)
img_list, velo_list, calib_cam_to_cam_list, calib_velo_to_cam_list = train_file[:, 0], train_file[:, 1], train_file[:, 2], train_file[:, 3]
bin_file_path_list_data = [i.replace('image_02', "pointSAM_label").replace('png', 'bin') for i in img_list]  # 多了一层data目录
bin_file_path_list = []


for p in bin_file_path_list_data:  # 消除多的data目录
    directory_path = os.path.dirname(os.path.dirname(p))
    file_name = os.path.basename(p)
    bin_file_path = os.path.join(directory_path, file_name)
    bin_file_path_list.append(bin_file_path)

for i in tqdm(range(len(img_list))):

    cam_to_cam = calib_cam_to_cam_list[i]
    velo_to_cam = calib_velo_to_cam_list[i]
    image_path = img_list[i]
    bin_file_path = bin_file_path_list[i]
    velo_points_path = velo_list[i]

    panoptic_label = np.fromfile(bin_file_path, dtype=np.uint16).reshape(-1, 2)
    instance_ids = panoptic_label[:, 0]
    semantic_ids = panoptic_label[:, 1]

    # 假设已经加载了原始点云的3D坐标
    # points_3d 是形状为 (N, 3) 的 numpy 数组，N 是点的数量
    points_3d = np.fromfile(velo_points_path, dtype=np.float32).reshape(-1, 4)
    points_3d = points_3d[:, :3]

    unique_instance_ids = np.unique(instance_ids)

    RoI_box_points = {'bbox2d':[], 'RoI_points':[]}
    for instance_id in unique_instance_ids:
        if instance_id == 65535:  # 忽略背景点或无效点
            continue
        # 找到当前实例的所有点
        instance_mask = (instance_ids == instance_id)
        points_instance = points_3d[instance_mask]  # 对应实例的所有点

        if points_instance.shape[0] < 10:
            continue

        # 计算边界框的最小值和最大值
        min_coords = points_instance.min(axis=0)
        max_coords = points_instance.max(axis=0)

        # 保存边界框
        bounding_box = {
            'instance_id': instance_id,
            'min_coords': min_coords,
            'max_coords': max_coords
        }

        bbox_2d = project_3d_box_to_2d_box(bounding_box, cam_to_cam, velo_to_cam)

        # 把点云从激光雷达坐标系投影到相机坐标系
        calib = calib_parse.parse_calib('raw', [cam_to_cam, velo_to_cam])
        points_instance = (calib['l2i'] @ np.concatenate([points_instance, np.ones_like(points_instance[:, 0:1])], axis=1).T).T
        points_instance = np.vstack(points_instance)

        RoI_box_points['bbox2d'].append(bbox_2d)
        RoI_box_points['RoI_points'].append(points_instance)


    # data/kitti/raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png
    output_path = image_path.replace('image_02', "pointSAM_pkl").replace('png', "pkl")
    output_directory = os.path.dirname(output_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if len(RoI_box_points['bbox2d']) < 1:
        with open(output_path, 'wb') as f:
            pickle.dump(RoI_box_points, f)
        continue


    RoI_box_points['bbox2d'] = np.array(RoI_box_points['bbox2d'])
    RoI_box_points['RoI_points'] = np.array(RoI_box_points['RoI_points'])
    with open(output_path, 'wb') as f:
        pickle.dump(RoI_box_points, f)













