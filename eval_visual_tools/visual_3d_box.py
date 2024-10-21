import cv2
import numpy as np
import matplotlib.pyplot as plt

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

    bbox_3d_in_2d = []
    for bbox_3d_coord in bbox_3d_coords:
        corners_2d = []
        for corner in bbox_3d_coord:
            corner_2d_homogeneous = projection_matrix @ corner.T
            corner_2d = corner_2d_homogeneous[:2] / corner_2d_homogeneous[2]  # 齐次坐标转换到二维坐标
            corners_2d.append(corner_2d)

        corners_2d = np.array(corners_2d)
        bbox_3d_in_2d.append(corners_2d)

    return bbox_3d_in_2d


def draw_3d_bbox(image, box_3d_in_2d):
    edges = [[0, 1], [1, 2], [2, 3], [3, 0],  # 底部四条边
             [4, 5], [5, 6], [6, 7], [7, 4],  # 顶部四条边
             [0, 4], [1, 5], [2, 6], [3, 7]]  # 垂直四条边

    for projected_corners in box_3d_in_2d:
        for edge in edges:
            pt1 = tuple(projected_corners[edge[0]].astype(int))
            pt2 = tuple(projected_corners[edge[1]].astype(int))
            cv2.line(image, pt1, pt2, color=(255, 0, 0), thickness=2)

cam_to_cam = "data/kitti/raw_data/2011_09_30/calib_cam_to_cam.txt"
velo_to_cam = "data/kitti/raw_data/2011_09_30/calib_velo_to_cam.txt"
image_path = "data/kitti/raw_data/2011_09_30/2011_09_30_drive_0028_sync/image_02/data/0000001390.png"

# 读取二进制文件
file_path = 'data/kitti/raw_data/2011_09_30/2011_09_30_drive_0028_sync/pointSAM_label/0000001390.bin'
panoptic_label = np.fromfile(file_path, dtype=np.uint16).reshape(-1, 2)

# 提取点云标签
instance_ids = panoptic_label[:, 0]
semantic_ids = panoptic_label[:, 1]

# 假设已经加载了原始点云的3D坐标
# points_3d 是形状为 (N, 3) 的 numpy 数组，N 是点的数量
points_3d = np.fromfile("data/kitti/raw_data/2011_09_30/2011_09_30_drive_0028_sync/velodyne_points/data/0000001390.bin", dtype=np.float32).reshape(-1, 4)
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

bbox_3d_in_2d = project_3d_box_to_2d_box(bounding_boxes, cam_to_cam, velo_to_cam)
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

draw_3d_bbox(image_rgb, bbox_3d_in_2d)

# 显示结果
plt.imshow(image_rgb)
plt.figure(dpi=300)
plt.axis('off')  # 不显示坐标轴
plt.show()
