import numpy as np
import os

def load_kitti_pointcloud(file_path):
    # 读取 .bin 文件中的点云数据
    pointcloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return pointcloud


def convert_kitti_to_nusc_format(pointcloud):
    # 添加 ring 信息，简单设置为 0
    ring = np.zeros((pointcloud.shape[0], 1), dtype=np.float32)
    # 拼接 ring 维度，形成 NuScenes 格式的点云
    nusc_pointcloud = np.hstack((pointcloud, ring))
    return nusc_pointcloud


def adjust_coordinates(nusc_pointcloud):
    # 反转 z 轴，适配 NuScenes 坐标系
    nusc_pointcloud[:, 2] = -nusc_pointcloud[:, 2]
    return nusc_pointcloud


def save_pointcloud_as_pcd_bin(nusc_pointcloud, output_path):
    # 假设 output_path 是完整的文件路径（包括文件名）
    directory = os.path.dirname(output_path)

    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 保存为 .pcd.bin 文件
    nusc_pointcloud.tofile(output_path)


def kitti_to_nusc_pcd_bin_conversion(kitti_file_path, output_file_path):
    pointcloud = load_kitti_pointcloud(kitti_file_path)
    nusc_pointcloud = convert_kitti_to_nusc_format(pointcloud)
    nusc_pointcloud = adjust_coordinates(nusc_pointcloud)
    save_pointcloud_as_pcd_bin(nusc_pointcloud, output_file_path)


def main():
    train_file_path = "data/kitti/data_file/split/train_raw.txt"
    train_file = np.loadtxt(train_file_path, dtype=str)
    velo_path_list = train_file[:, 1]

    for velo_path in velo_path_list:
        kitti_file_path = velo_path
        output_file_path = velo_path.replace("velodyne_points", "nusc_points_train")
        kitti_to_nusc_pcd_bin_conversion(kitti_file_path, output_file_path)


if __name__ == '__main__':
    main()

















