import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
from mpl_toolkits.mplot3d import Axes3D


def generate_3d_bbox(ax, min_coords, max_coords):
    # 生成8个边界框顶点
    corners = np.array([[min_coords[0], min_coords[1], min_coords[2]],
                        [max_coords[0], min_coords[1], min_coords[2]],
                        [max_coords[0], max_coords[1], min_coords[2]],
                        [min_coords[0], max_coords[1], min_coords[2]],
                        [min_coords[0], min_coords[1], max_coords[2]],
                        [max_coords[0], min_coords[1], max_coords[2]],
                        [max_coords[0], max_coords[1], max_coords[2]],
                        [min_coords[0], max_coords[1], max_coords[2]]])

    # 定义边界框的12条边
    edges = [[corners[0], corners[1], corners[2], corners[3], corners[0]],
             [corners[4], corners[5], corners[6], corners[7], corners[4]],
             [corners[0], corners[4]],
             [corners[1], corners[5]],
             [corners[2], corners[6]],
             [corners[3], corners[7]]]

    # 绘制边界框
    for edge in edges:
        ax.plot3D(*zip(*edge), color='b')

def draw_3d_box(bounding_boxes):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # 绘制点云
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1, c='g', alpha=0.5)
    # 绘制每个边界框
    for bbox in bounding_boxes:
        generate_3d_bbox(ax, bbox['min_coords'], bbox['max_coords'])
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def project_3d_box_to_2d_box(bounding_boxes, cam_to_cam, velo_to_cam):

    P2 = None
    R = None
    T = None
    with open(cam_to_cam, 'r') as f1:
        lines = f1.readlines()
        for line in lines:
            if line.startswith('P_rect_02:'):  # 提取相机2的内参矩阵
                P2 = line.strip().split(' ')[1:]
                P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
                break
    intrinsic_matrix = P2[:3, :3]

    with open(velo_to_cam, 'r') as f2:
        lines = f2.readlines()
        for line in lines:
            if line.startswith('R:'):
                R = line.strip().split(' ')[1:]
                R = np.array(R, dtype=np.float32).reshape(3, 3)

            if line.startswith('T:'):
                T = line.strip().split(' ')[1:]
                T = np.array(T, dtype=np.float32).reshape(3, 1)
    extrinsic_matrix = np.hstack((R, T.reshape(-1, 1)))

    projection_matrix = intrinsic_matrix @ extrinsic_matrix


    bbox_3d_coords = []
    for bbox in bounding_boxes:
        min_coords = bbox["min_coords"]
        max_coords = bbox["max_coords"]
        corners = np.array([[min_coords[0], min_coords[1], min_coords[2],1],
                            [max_coords[0], min_coords[1], min_coords[2],1],
                            [max_coords[0], max_coords[1], min_coords[2],1],
                            [min_coords[0], max_coords[1], min_coords[2],1],
                            [min_coords[0], min_coords[1], max_coords[2],1],
                            [max_coords[0], min_coords[1], max_coords[2],1],
                            [max_coords[0], max_coords[1], max_coords[2],1],
                            [min_coords[0], max_coords[1], max_coords[2],1]])
        bbox_3d_coords.append(corners)

    bbox_2d_list = []
    for bbox_3d_coord in bbox_3d_coords:
        corners_2d = []
        for corner in bbox_3d_coord:
            corner_2d_homogeneous = projection_matrix @ corner.T
            corner_2d = corner_2d_homogeneous[:2] / corner_2d_homogeneous[2]  # 齐次坐标转换到二维坐标
            corners_2d.append(corner_2d)

        corners_2d = np.array(corners_2d)

        # 计算2D边界框的最小值和最大值
        min_2d = corners_2d.min(axis=0)
        max_2d = corners_2d.max(axis=0)

        # 2D bbox 的四个顶点坐标
        width = max_2d[0] - min_2d[0]
        height = max_2d[1] - min_2d[1]

        bbox_2d_corners = [min_2d[0], min_2d[1], width, height]

        bbox_2d_list.append(bbox_2d_corners)

    return bbox_2d_list

def draw_2d_box(image_path, bbox_2d_list):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for bbox in bbox_2d_list:
        plt.gca().add_patch(plt.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            fill=False, edgecolor='red', linewidth=1))

    plt.imshow(image)
    plt.show()




cam_to_cam = "data/kitti/raw_data/2011_09_30/calib_cam_to_cam.txt"
velo_to_cam = "data/kitti/raw_data/2011_09_30/calib_velo_to_cam.txt"
image_path = "data/kitti/raw_data/2011_09_30/2011_09_30_drive_0020_sync/image_02/data/0000000988.png"

# 读取二进制文件
file_path = 'data/kitti/raw_data/2011_09_30/2011_09_30_drive_0020_sync/pointSAM_label/0000000000.bin'
panoptic_label = np.fromfile(file_path, dtype=np.uint16).reshape(-1, 2)

# 提取点云标签
instance_ids = panoptic_label[:, 0]
semantic_ids = panoptic_label[:, 1]

# 假设已经加载了原始点云的3D坐标
# points_3d 是形状为 (N, 3) 的 numpy 数组，N 是点的数量
points_3d = np.fromfile("data/kitti/raw_data/2011_09_30/2011_09_30_drive_0020_sync/velodyne_points/data/0000000000.bin", dtype=np.float32).reshape(-1, 4)
points_3d = points_3d[:, :3]

# 生成3D Bounding Boxes
bounding_boxes = []
unique_instance_ids = np.unique(instance_ids)

for instance_id in unique_instance_ids:
    if instance_id == 65535:  # 忽略背景点或无效点
        continue
    # 找到当前实例的所有点
    instance_mask = (instance_ids == instance_id)
    points_instance = points_3d[instance_mask]  # 对应实例的所有点

    # 计算边界框的最小值和最大值
    min_coords = points_instance.min(axis=0)
    max_coords = points_instance.max(axis=0)

    # 保存边界框
    bounding_boxes.append({
        'instance_id': instance_id,
        'min_coords': min_coords,
        'max_coords': max_coords
    })


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# generate_3d_bbox(ax, min_coords, max_coords)
# draw_3d_box(bounding_boxes)


bbox_2d_list = project_3d_box_to_2d_box(bounding_boxes, cam_to_cam, velo_to_cam)
draw_2d_box(image_path, bbox_2d_list)
