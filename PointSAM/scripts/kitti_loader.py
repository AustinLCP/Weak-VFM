import numpy as np
import uuid
from scipy.spatial.transform import Rotation as R
import os
from datetime import datetime
from collections import defaultdict

class KittiLoader:
    def __init__(self):

        self.train_file_path = "data/kitti/data_file/split/train_raw_sample.txt"
        self.train_file = np.loadtxt(self.train_file_path, dtype=str)
        self.img_path_list, self.velo_path_list, self.calib_cam_to_cam_list, self.calib_velo_to_cam_list = (
            self.train_file[:, 0], self.train_file[:, 1], self.train_file[:, 2], self.train_file[:,3])

        self.file_dict_cam = self.set_file_dict("cam")  # use dict to present directories structure
        self.file_dict_lidar = self.set_file_dict("lidar")

        self.sample = self.set_samples()
        self.calibrated_sensor = self.set_calibrated_sensor()
        # self.ego_pose = self.set_ego_pose()
        self.sample_data = self.set_sample_data()

        self.dataroot = ""

    # {
    #     token:
    #     data{
    #               CAM_FRONT -> image_02
    #               LIDAR_TOP -> velodyne_points
    #           }
    # }
    def set_samples(self):
        samples = []
        for i in range(len(self.train_file)):
            # token
            token = str(uuid.uuid4())

            # data (use path as data token)
            data = {}
            data["CAM_FRONT"] = self.img_path_list[i]
            data["LIDAR_TOP"] = self.velo_path_list[i]

            sample = {"token": token, "data": data}

            samples.append(sample)
        return samples


    # {
    #     token
    #     translation
    #     rotation
    # }
    def extract_calib_cam_to_cam(self):

        # R_velo_to_cams = [] # 将激光雷达点云数据投影到摄像头视图中
        # T_velo_to_cams = []
        #
        # P_rec_matrics = []  # 用于图像矫正的投影矩阵
        ref_pose = [] # matrix for loading multi-sweep pc

        # for i in range(len(self.calib_velo_to_cam_list)):
        #     with open(self.calib_velo_to_cam_list[i], 'r') as f:
        #         lines = f.readlines()
        #         for line in lines:
        #             if line.startswith('R:'):
        #                 R_velo_to_cam = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 3)
        #                 R_velo_to_cams.append(R_velo_to_cam)
        #             if line.startswith('T:'):
        #                 T_velo_to_cam = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3)
        #                 T_velo_to_cams.append(T_velo_to_cam)

        for i in range(len(self.calib_cam_to_cam_list)):
            with open(self.calib_cam_to_cam_list[i], 'r') as f:
                for line in f:
                    if 'P_rect_02' in line:
                        """P_rect_02"""
                        # 获取P_rect_02矩阵，通常用于左侧相机
                        P_rect_02 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
                        # P_rec_matrics.append(P_rect_02)

                        """ref_pose"""
                        # Extract the rotation (3x3) and translation (3x1) components
                        R_ = P_rect_02[:, :3]  # Rotation matrix
                        t_ = P_rect_02[:, 3]  # Translation vector

                        # Form the 4x4 transformation matrix
                        transformation_matrix = np.eye(4)
                        transformation_matrix[:3, :3] = R_
                        transformation_matrix[:3, 3] = t_
                        ref_pose.append(transformation_matrix)

        # return R_velo_to_cams, T_velo_to_cams, P_rec_matrics, ref_pose
        return ref_pose

    '''车辆(相机)坐标系'''
    def set_calibrated_sensor(self):
        calibrated_sensor_list = []
        ref_pose = self.extract_calib_cam_to_cam()

        for i in range(len(self.calib_cam_to_cam_list)):
            calibrated_sensor = {
                "token": str(uuid.uuid4()),
                # "R_velo_to_cam": R_velo_to_cams[i],
                # "T_velo_to_cam": T_velo_to_cams[i],
                # "P_rec_matric": P_rec_matrics[i],
                "ref_pose": ref_pose[i]
            }

            calibrated_sensor_list.append(calibrated_sensor)

        return calibrated_sensor_list


    # {
    #     token
    #     translation
    #     rotation
    # }
    def extract_oxts(self, oxts_files):
        oxts_data = []
        for filename in oxts_files:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    data = list(map(float, line.strip().split()))
                    oxts_data.append(data)

        translations = []
        rotations = []

        initial_lat = oxts_data[0][0]
        scale = np.cos(initial_lat * np.pi / 180.0)

        for data in oxts_data:
            latitude = data[0]
            longitude = data[1]
            altitude = data[2]
            roll = data[3]
            pitch = data[4]
            yaw = data[5]

            earth_radius = 6378137.0  # Earth radius in meters
            tx = scale * longitude * np.pi * earth_radius / 180.0
            ty = earth_radius * np.log(np.tan((90.0 + latitude) * np.pi / 360.0))
            tz = altitude

            translation = np.array([tx, ty, tz])
            translations.append(translation)

            rotation = R.from_euler('zyx', [yaw, pitch, roll]).as_quat()
            rotations.append(rotation)

        return translations, rotations

    '''世界坐标系'''
    def set_ego_pose(self):
        ego_pose_list = []

        oxts_files = [i.replace('image_02', "oxts").replace('png', 'txt')
                                           for i in self.img_path_list]

        translations, rotations = self.extract_oxts(oxts_files)

        token = uuid.uuid4()

        for i in range(len(self.img_path_list)):
            ego_pose = {
                "token": token,
                "translation": translations[i],
                "rotation": rotations[i],
            }
            ego_pose_list.append(ego_pose)

        return ego_pose_list


    def set_timestamps(self):
        cam_timestamp_path_list = []
        for i in range(len(self.img_path_list)):
            p_cam = self.img_path_list[i].split(r"\image_02")[0]
            cam_timestamp_path = os.path.join(p_cam, "image_02/timestamps.txt")
            cam_timestamp_path_list.append(cam_timestamp_path)

        velo_timestamp_path_list = [i.replace('image_02', "velodyne_points") for i in cam_timestamp_path_list]


        # frame number list (picture/velo number)
        frame_number_list = []
        for i in range(len(self.img_path_list)):
            file_name = self.img_path_list[i].split('\\')[-1]
            frame_number = int(file_name.split('.')[0])
            frame_number_list.append(frame_number)


        cam_timestamp_list = []
        for i in range(len(cam_timestamp_path_list)):
            with open(cam_timestamp_path_list[i], 'r') as file:
                timestamps = file.readlines()

            timestamp_str = timestamps[frame_number_list[i]].strip()
            timestamp_str = timestamp_str[:-3]
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            timestamp_seconds = int(dt.timestamp())
            cam_timestamp_list.append(timestamp_seconds)


        velo_timestamp_list = []
        for i in range(len(velo_timestamp_path_list)):
            with open(velo_timestamp_path_list[i], 'r') as file:
                timestamps = file.readlines()

            timestamp_str = timestamps[frame_number_list[i]].strip()
            timestamp_str = timestamp_str[:-3]
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            timestamp_seconds = int(dt.timestamp())
            velo_timestamp_list.append(timestamp_seconds)

        return cam_timestamp_list, velo_timestamp_list

    def set_file_dict(self, mode):

        file_list = None
        if mode == "cam":
            file_list = self.img_path_list
        elif mode == "lidar":
            file_list = self.velo_path_list

        file_dict = defaultdict(list)

        # Preprocess to categorize files by directory
        for path in file_list:
            dir_key = (path.split("\\")[3], path.split("\\")[4])
            file_dict[dir_key].append(path)

        # Sort each file list once
        for key in file_dict:
            file_dict[key] = sorted(file_dict[key], key=lambda x: os.path.basename(x))

        return file_dict

    def set_prev_next(self, sample_data_token, mode):
        dir_key = (sample_data_token.split("\\")[3], sample_data_token.split("\\")[4])

        same_dir_file_list=None
        if mode == "cam":
            same_dir_file_list = self.file_dict_cam[dir_key]
        elif mode == "lidar":
            same_dir_file_list = self.file_dict_lidar[dir_key]

        # if sample_data_token not in same_dir_file_list:
        #     same_dir_file_list.append(sample_data_token)
        #     same_dir_file_list.sort(key=lambda x: os.path.basename(x))

        next = None
        prev = None
        for i, file in enumerate(same_dir_file_list):
            if file == sample_data_token:
                next = None if i == len(same_dir_file_list) - 1 else same_dir_file_list[i + 1]
                prev = None if i == 0 else same_dir_file_list[i - 1]
                break

        return next, prev

    # {
    #     token:                   -> file path
    #     sample_token:
    #     filename:
    #     channel:
    #     calibrated_sensor_token:
    #     ego_pose_token:
    #     timestamp:
    # }
    def set_sample_data(self):
        sample_data_list = []
        cam_timestamp_list, velo_timestamp_list = self.set_timestamps()
        for i in range(len(self.sample)):

            cam_next, cam_prev = self.set_prev_next(self.sample[i]["data"]["CAM_FRONT"],"cam")

            cam_sample_data = {
                "token": self.sample[i]["data"]["CAM_FRONT"],
                "sample_token": self.sample[i]["token"],
                "filename": self.sample[i]["data"]["CAM_FRONT"],
                "channel": "CAM_FRONT",
                "calibrated_sensor_token": self.calibrated_sensor[i]["token"],
                # "ego_pose_token": self.ego_pose[i]["token"],
                "timestamp": cam_timestamp_list[i],
                "prev": cam_prev,
                "next": cam_next,
                "cam_to_cam_path": self.calib_cam_to_cam_list[i],
                "velo_to_cam_path": self.calib_velo_to_cam_list[i]
            }


            lidar_next, lidar_prev = self.set_prev_next(self.sample[i]["data"]["LIDAR_TOP"], "lidar")

            lidar_sample_data = {
                "token": self.sample[i]["data"]["LIDAR_TOP"],
                "sample_token": self.sample[i]["token"],
                "filename": self.sample[i]["data"]["LIDAR_TOP"],
                "channel": "LIDAR_TOP",
                "calibrated_sensor_token": self.calibrated_sensor[i]["token"],
                # "ego_pose_token": self.ego_pose[i]["token"],
                "timestamp": velo_timestamp_list[i],
                "prev": lidar_prev,
                "next": lidar_next,
            }

            sample_data_list.append(cam_sample_data)
            sample_data_list.append(lidar_sample_data)

        return sample_data_list



    def get(self,data_str, token):

        if data_str == "sample":
            for i in range(len(self.sample)):
                if self.sample[i]["token"] == token:
                    return self.sample[i]
        elif data_str == "sample_data":
            for i in range(len(self.sample_data)):
                if self.sample_data[i]["token"] == token:
                    return self.sample_data[i]
        elif data_str == "ego_pose":
            for i in range(len(self.ego_pose)):
                if self.ego_pose[i]["token"] == token:
                    return self.ego_pose[i]
        elif data_str == "calibrated_sensor":
            for i in range(len(self.calibrated_sensor)):
                if self.calibrated_sensor[i]["token"] == token:
                    return self.calibrated_sensor[i]
        else:
            print("token not found")
            raise ValueError


