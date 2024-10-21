import pickle
import numpy as np
from tqdm import tqdm

train_file = np.loadtxt("data/kitti/data_file/split/train_raw.txt", dtype=str)
img_list, velo_list, calib_cam_to_cam_list, calib_velo_to_cam_list = train_file[:, 0], train_file[:, 1], train_file[:, 2], train_file[:, 3]
pkl_file_path_list = [i.replace('image_02', "lidar_RoI_points").replace('png', 'pkl') for i in img_list]
# /lidar_RoI_points/
# /pointSAM/

counter = 0
for pkl_file_path in tqdm(pkl_file_path_list):
    with open(pkl_file_path, 'rb') as f:
        RoI_box_points = pickle.load(f)

    bbox2d = RoI_box_points['bbox2d']
    l_3d = RoI_box_points['RoI_points']

    if len(bbox2d) == 0:
        print(pkl_file_path)
        counter += 1
        exit()


print(counter)










