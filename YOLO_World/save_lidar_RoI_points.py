from sklearn.cluster import DBSCAN
import os
from tqdm import tqdm
import numpy as np
import cv2 as cv
import pickle

import sys
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt


kitti_raw_file_name = 'H:\\Honours\\WeakM3D\\WeakM3D\\data\\kitti\\data_file\\split\\train_raw.txt'
full_path_prefix = "H:\\Honours\\WeakM3D\\WeakM3D\\"

def parse_calib(mode, calib_path=None):
    if mode == '3d':
        with open(calib_path, encoding='utf-8') as f:
            text = f.readlines()
            P0 = np.array(text[0].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            P1 = np.array(text[1].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            P2 = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            P3 = np.array(text[3].split(' ')[1:], dtype=np.float32).reshape(3, 4)

            Tr_velo_to_cam = np.array(text[5].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            # Tr_imu_to_velo = np.array(text[6].split(' ')[1:], dtype=np.float32).reshape(3, 4)

            R_rect = np.zeros((4, 4))
            R_rect_tmp = np.array(text[4].split(' ')[1:], dtype=np.float32).reshape(3, 3)
            R_rect[:3, :3] = R_rect_tmp
            R_rect[3, 3] = 1

            Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
            '''lidar to image pixel plane'''
            l2p = np.dot(np.dot(P2, R_rect), Tr_velo_to_cam)
            '''lidar to image plane'''
            l2i = np.dot(R_rect_tmp, Tr_velo_to_cam[:3])

    elif mode == 'raw':
        calib_cam2cam_path, velo2cam_calib_path = calib_path
        with open(velo2cam_calib_path, encoding='utf-8') as f:
            text = f.readlines()
            R = np.array(text[1].split(' ')[1:], dtype=np.float32).reshape(3, 3)
            T = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 1)

            trans = np.concatenate([R, T], axis=1)
            vel2cam = trans.copy()

            Tr_velo_to_cam = np.concatenate([trans, np.array([[0, 0, 0, 1]])], axis=0)

        with open(calib_cam2cam_path, encoding='utf-8') as f:
            text = f.readlines()
            P2 = np.array(text[-9].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            R_rect = np.zeros((4, 4))
            R_rect_tmp = np.array(text[8].split(' ')[1:], dtype=np.float32).reshape(3, 3)
            R_rect[:3, :3] = R_rect_tmp
            R_rect[3, 3] = 1

            '''lidar to image pixel plane'''
            l2p = np.dot(np.dot(P2, R_rect), Tr_velo_to_cam)
            '''lidar to image plane'''
            l2i = np.dot(R_rect_tmp, vel2cam)

    calib = {
        'P2': P2,
        'l2p': l2p,
        'l2i': l2i
    }
    return calib

def obtain_cluster_RoI_points_for_raw_kitti(l_3d, l_2d, seg_bbox_path, seg_mask_path, file_ind, P2):
    # (x1,y1,x2,y2,score)
    bbox2d = np.loadtxt(seg_bbox_path).reshape(-1, 5)
    bbox_mask = (np.load(seg_mask_path))['masks']

    all_box, all_RoI_points = [], []
    for index, b in enumerate(bbox2d):

        obj_mask = bbox_mask[index]
        ind = obj_mask[l_2d[:, 1], l_2d[:, 0]].astype(np.bool_)
        cam_points = l_3d[ind]
        if len(cam_points) < 10:
            continue

        cluster_index = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1).fit_predict(cam_points)

        cam_points = cam_points[cluster_index > -1]
        cluster_index = cluster_index[cluster_index > -1]

        if len(cam_points) < 10:
            continue

        cluster_set = set(cluster_index[cluster_index > -1])
        cluster_sum = np.array([len(cam_points[cluster_index == i]) for i in cluster_set])
        RoI_points = cam_points[cluster_index == np.argmax(cluster_sum)]

        all_box.append(b[:-1])
        all_RoI_points.append(RoI_points)  # 点数必须大于10

    return np.array(all_box), np.array(all_RoI_points, dtype=object)

def visual_points_on_img(img, proj_coords):
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.scatter(proj_coords[:, 0], proj_coords[:, 1], c='r', s=2)
    plt.show()

if __name__ == '__main__':
    train_file = np.loadtxt(kitti_raw_file_name, dtype=str)
    img_list, velo_list, calib_cam_to_cam_list, calib_velo_to_cam_list = \
        train_file[:, 0], train_file[:, 1], train_file[:, 2], train_file[:, 3]

    for i, _ in enumerate(tqdm(img_list, desc="Projecting lidar points")):
        raw_lidar_path = os.path.join(full_path_prefix, velo_list[i])
        img_path = os.path.join(full_path_prefix, img_list[i])
        calib_cam_to_cam_list[i] = os.path.join(full_path_prefix, calib_cam_to_cam_list[i])
        calib_velo_to_cam_list[i] = os.path.join(full_path_prefix, calib_velo_to_cam_list[i])

        seg_mask_path = img_path.replace('image_02', 'YOLO_seg_mask').replace('png', 'npz')
        seg_bbox_path = img_path.replace('image_02', 'YOLO_seg_bbox').replace('png', 'txt')
        lidar_save_RoI_points_path = img_path.replace('image_02', 'YOLO_RoI_points').replace('png', 'pkl')

        lidar_save_RoI_points_dir = os.path.dirname(lidar_save_RoI_points_path)
        if not os.path.exists(lidar_save_RoI_points_dir):
            os.makedirs(lidar_save_RoI_points_dir)

        # 激光雷达坐标系下的 (x, y, z)
        l_3d = np.fromfile(str(raw_lidar_path), dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]

        # obtain 2d coordinates
        # l2i: 激光雷达到相机的变换矩阵
        # P2: 相机投影矩阵，用于将3D点投影到2D图像平面
        calib = parse_calib('raw', [calib_cam_to_cam_list[i], calib_velo_to_cam_list[i]])
        # 相机坐标系下的点 (N,(x, y, z，1))
        l_3d = (calib['l2i'] @ np.concatenate([l_3d, np.ones_like(l_3d[:, 0:1])], axis=1).T).T
        l_2d = (calib['P2'] @ np.concatenate([l_3d, np.ones_like(l_3d[:, 0:1])], axis=1).T).T
        # 图像坐标系下的点 (N,(x,y))
        l_2d = (l_2d[:, :2] / l_2d[:, 2:3]).astype(int)

        # remove points outside fov 过滤掉视场外的点
        rgb_img = cv.imread(img_path)
        h, w, _ = rgb_img.shape
        valid_ind = (l_2d[:, 0] > 0) & (l_2d[:, 0] < w) & (l_2d[:, 1] > 0) & (l_2d[:, 1] < h) & (l_3d[:, 2] > 0)
        l_3d, l_2d = l_3d[valid_ind], l_2d[valid_ind]

        RoI_box_points = {'bbox2d':[], 'RoI_points':[]}
        if not os.path.exists(seg_bbox_path):
            with open(lidar_save_RoI_points_path, 'wb') as f:
                pickle.dump(RoI_box_points, f)
            continue

        bbox2d = np.loadtxt(seg_bbox_path).reshape(-1, 5)
        if len(bbox2d) < 1:
            with open(lidar_save_RoI_points_path, 'wb') as f:
                pickle.dump(RoI_box_points, f)
            continue

        cls_bbox2d, RoI_points = obtain_cluster_RoI_points_for_raw_kitti(
                    l_3d, l_2d, seg_bbox_path, seg_mask_path, i, calib['P2'])

        RoI_box_points['bbox2d'] = cls_bbox2d
        RoI_box_points['RoI_points'] = RoI_points
        with open(lidar_save_RoI_points_path, 'wb') as f:
            pickle.dump(RoI_box_points, f)







