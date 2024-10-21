import pickle
import matplotlib.pyplot as plt
import numpy as np
from dataloader import calib_parse
import cv2 as cv
import torch


def draw_points_on_2d_box(image_path, bbox_2d_list, l_3d_cam, calib_cam_to_cam,calib_velo_to_cam):
    l_3d_cam = np.vstack(l_3d_cam)

    # l2i: 激光雷达到相机的变换矩阵
    # P2: 相机投影矩阵，用于将3D点投影到2D图像平面
    calib = calib_parse.parse_calib('raw', [calib_cam_to_cam, calib_velo_to_cam])
    # 相机坐标系下的点 (N,(x, y, z，1))
    l_2d = (calib['P2'] @ np.concatenate([l_3d_cam, np.ones_like(l_3d_cam[:, 0:1])], axis=1).T).T
    # 图像坐标系下的点 (N,(x,y))
    l_2d = (l_2d[:, :2] / l_2d[:, 2:3]).astype(int)

    # remove points outside fov 过滤掉视场外的点
    rgb_img = cv.imread(image_path)
    h, w, _ = rgb_img.shape
    valid_ind = (l_2d[:, 0] > 0) & (l_2d[:, 0] < w) & (l_2d[:, 1] > 0) & (l_2d[:, 1] < h) & (l_3d_cam[:, 2] > 0)
    l_2d = l_2d[valid_ind]

    # 创建一个1行2列的子图布局，增加figsize和dpi
    fig, axs = plt.subplots(2, 1, figsize=(16, 8), dpi=150)

    # 左边的子图：仅显示2D bounding box
    for bbox in bbox_2d_list:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        axs[0].add_patch(plt.Rectangle(
            (bbox[0], bbox[1]), width, height,
            fill=False, edgecolor='red', linewidth=1))

    # 显示图像
    axs[0].imshow(rgb_img)
    axs[0].set_title('2d bbox projected from 3d instance segmentation')

    # 右边的子图：显示2D bounding box和投影后的3D点
    for bbox in bbox_2d_list:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        axs[1].add_patch(plt.Rectangle(
            (bbox[0], bbox[1]), width, height,
            fill=False, edgecolor='green', linewidth=1))

    # 投影后的3D点
    l_2d = l_2d.cpu().numpy() if isinstance(l_2d, torch.Tensor) else l_2d
    axs[1].scatter(l_2d[:, 0], l_2d[:, 1], c='r', s=2)

    # 显示图像
    axs[1].imshow(rgb_img)
    axs[1].set_title('2d bbox and 3d points projected from 3d instance segmentation')

    # 确保子图之间没有重叠，增加显示效果
    plt.tight_layout()

    # 显示结果
    plt.show()


# /lidar_RoI_points/
# /pointSAM_pkl/
pkl_path = "data/kitti/raw_data/2011_09_26/2011_09_26_drive_0005_sync/pointSAM_pkl/data/0000000109.pkl"
image_path = "data/kitti/raw_data/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/0000000109.png"
calib_cam_to_cam = "data/kitti/raw_data/2011_09_26/calib_cam_to_cam.txt"
calib_velo_to_cam = "data/kitti/raw_data/2011_09_26/calib_velo_to_cam.txt"

with open(pkl_path, 'rb') as f:
    RoI_box_points = pickle.load(f)

bbox2d = RoI_box_points['bbox2d']
l_3d = RoI_box_points['RoI_points']

draw_points_on_2d_box(image_path, bbox2d, l_3d, calib_cam_to_cam, calib_velo_to_cam)





