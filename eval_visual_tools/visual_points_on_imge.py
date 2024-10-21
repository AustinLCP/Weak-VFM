import os
import numpy as np
import cv2
import pickle
from dataloader import calib_parse
import sys
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# lidar_RoI_points
# pointSAM_pkl
# YOLO_RoI_points
# langSAM_RoI_points
lidar_path = 'data\\kitti\\raw_data\\2011_09_30\\2011_09_30_drive_0028_sync\\YOLO_RoI_points\\data\\0000001390.pkl'
image_path = 'data\\kitti\\raw_data\\2011_09_30\\2011_09_30_drive_0028_sync\\image_02\\data\\0000001390.png'
calib_cam_to_cam_list = 'data\\kitti\\raw_data\\2011_09_30\\calib_cam_to_cam.txt'
calib_velo_to_cam_list = 'data\\kitti\\raw_data\\2011_09_30\\calib_velo_to_cam.txt'


with open(lidar_path, 'rb') as f:
    bbox2d_point_list = pickle.load(f)

projected_points = []
for l_3d in bbox2d_point_list["RoI_points"]:

    # obtain 2d coordinates
    # l2i: 激光雷达到相机的变换矩阵
    # P2: 相机投影矩阵，用于将3D点投影到2D图像平面
    calib = calib_parse.parse_calib('raw', [calib_cam_to_cam_list, calib_velo_to_cam_list])
    # 相机坐标系下的点 (N,(x, y, z，1))
    # l_3d = (calib['l2i'] @ np.concatenate([l_3d, np.ones_like(l_3d[:, 0:1])], axis=1).T).T
    l_2d = (calib['P2'] @ np.concatenate([l_3d, np.ones_like(l_3d[:, 0:1])], axis=1).T).T
    # 图像坐标系下的点 (N,(x,y))
    l_2d = (l_2d[:, :2] / l_2d[:, 2:3]).astype(int)

    projected_points.append(l_2d)


image = cv2.imread(image_path)  # 加载图片，可以根据需要改成对应的图片路径
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB，便于matplotlib显示
for points_2d in projected_points:
    for point in points_2d:
        # 将点绘制在图片上，点的大小和颜色可以根据需要调整
        cv2.circle(image_rgb, (int(point[0]), int(point[1])), radius=3, color=(255, 0, 0), thickness=-1)



fig, ax = plt.subplots(1,dpi=200)
bboxes = bbox2d_point_list["bbox2d"]
# for bbox in bboxes:
#     xmin, ymin, xmax, ymax = bbox
#     # 使用 Rectangle 绘制边界框
#     rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)


# 显示结果
plt.imshow(image_rgb)
plt.axis('off')  # 隐藏坐标轴
plt.title("LangSAM's Label")
plt.show()

