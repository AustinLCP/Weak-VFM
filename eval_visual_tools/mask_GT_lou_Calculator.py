import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 读取ground truth文件中的2D bounding box信息
def load_ground_truth(file_path):
    ground_truth = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "DontCare" not in line:
                data = line.split()
                # 提取2D边界框坐标（x_min, y_min, x_max, y_max）
                x_min = float(data[4])
                y_min = float(data[5])
                x_max = float(data[6])
                y_max = float(data[7])

                # 添加到列表中
                ground_truth.append((x_min, y_min, x_max, y_max))
    return ground_truth

# 计算ground truth bounding box的面积
def compute_bbox_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) * (y_max - y_min)

# 读取掩码文件
def load_masks(file_path):
    mask_data = np.load(file_path)
    return mask_data['data']  # 假设掩码文件中的数据以'masks'键存储

# 计算掩码与bounding box的交集区域
def compute_mask_in_bbox(mask, bbox):
    x_min, y_min, x_max, y_max = map(int, bbox)  # 将边界框坐标转换为整数
    # 提取bounding box区域内的掩码
    mask_in_bbox = mask[y_min:y_max, x_min:x_max]
    return np.sum(mask_in_bbox)  # 计算掩码在bounding box内的像素数

# 计算覆盖率
def compute_iou(mask, ground_truth):
    ious = []
    weighted_iou = 0
    gt_area_sum = 0

    for bbox in ground_truth:
        gt_area_sum += compute_bbox_area(bbox)  # 计算bounding box的面积总和

    for bbox in ground_truth:
        gt_area = compute_bbox_area(bbox)  # 计算bounding box的面积
        mask_in_bbox_area = compute_mask_in_bbox(mask, bbox)  # 计算掩码在bounding box内的面积
        iou = mask_in_bbox_area / gt_area  # 计算覆盖率（IoU）
        ious.append(iou)
        weighted_iou += iou * (gt_area / gt_area_sum) / len(ground_truth)

    results = dict(iou=ious, weighted_iou=weighted_iou)
    return results


# 主程序
ground_truth_file = 'data/kitti/raw_data/9999_99_99/9999_99_99_drive_9999_sync/label/data/002544.txt'
mask_file = 'PointSAM/PointSAM_val/CAM_FRONT/instance_9999_99_99_drive_9999_sync_000142.png.npz'
img_file = 'data/kitti/raw_data/9999_99_99/9999_99_99_drive_9999_sync/image_02/data/000142.png'


# 加载ground truth和掩码数据
ground_truth = load_ground_truth(ground_truth_file)
masks = load_masks(mask_file)
image = cv2.imread(img_file)

# 计算每个掩码的覆盖率（IoU）
iou_result = compute_iou(masks, ground_truth)


# 输出结果
print(f'IoU details: {iou_result["iou"]}')
print(" ")
print(f'Weighted IoU: {iou_result["weighted_iou"]}')



fig, ax = plt.subplots(1)
ax.imshow(image)
# ax.imshow(masks, cmap='jet', alpha=0.5)
for bbox in ground_truth:
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
plt.axis('off')
plt.show()