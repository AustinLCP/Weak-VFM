import json
import os
from tqdm import tqdm
import numpy as np


def kitti_labels_to_coco_format():

    # KITTI 数据集路径和图片路径
    kitti_label_dir = 'data/kitti/KITTI3D/training/label_2'  # 标注文件的路径
    image_dir = 'data/kitti/KITTI3D/training/image_2'  # 图片文件的路径

    # 构建 COCO 格式的字典
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 定义类别信息 (根据 KITTI 类别)
    categories = [
        {"id": 0, "name": "Car"},
        {"id": 1, "name": "Pedestrian"},
        {"id": 2, "name": "Cyclist"},
        {"id": 3, "name": "Van"},
        {"id": 4, "name": "Person_sitting"},
        {"id": 5, "name": "Truck"},
        {"id": 6, "name": "Tram"},
    ]

    categories_weakm3d = [
        {"id": 0, "name": "Car"},
    ]

    # 添加类别信息到 COCO 格式
    coco_format["categories"] = categories_weakm3d

    # 图像和标注ID初始化
    image_id = 0
    annotation_id = 0

    # 遍历KITTI标注文件
    for label_file in tqdm(os.listdir(kitti_label_dir), desc="Processing KITTI Labels to gt form"):

        # 获取图像信息
        image_name = label_file.replace('.txt', '.png')  # 假设图像格式为PNG，按需修改
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            continue  # 跳过没有对应图像的标注文件

        # 读取图像尺寸（可以使用opencv或PIL等库）
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size

        # 添加图像信息到COCO格式
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        # 读取标注文件
        with open(os.path.join(kitti_label_dir, label_file), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                category_name = parts[0]

                # 查找类别ID
                category_id = next((cat["id"] for cat in categories_weakm3d if cat["name"].lower() == category_name.lower()),
                                   None)
                if category_id is None:
                    continue

                # 边界框 (xmin, ymin, xmax, ymax)
                xmin, ymin, xmax, ymax = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                width_bbox = xmax - xmin
                height_bbox = ymax - ymin

                # 添加标注信息到COCO格式
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [xmin, ymin, width_bbox, height_bbox],  # COCO格式要求的边界框格式 [x, y, width, height]
                    "area": width_bbox * height_bbox,  # 边界框面积
                    "iscrowd": 0  # 用于检测任务，通常为0
                })
                annotation_id += 1

        image_id += 1
        # pbar.update(1)

    # 将结果保存到JSON文件
    with open('eval_visual_tools/ground_truth/gt.json', 'w') as f:
        json.dump(coco_format, f)


def kitti_labels_in_train_3d_to_coco_format():

    # KITTI 数据集路径和图片路径
    train_path_list = np.loadtxt("data/kitti/data_file/split/train_3d.txt", dtype=str)
    kitti_label_dir = 'data/kitti/KITTI3D/training/label_2'  # 标注文件的路径
    image_dir = 'data/kitti/KITTI3D/training/image_2'

    # 构建 COCO 格式的字典
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 定义类别信息 (根据 KITTI 类别)
    categories = [
        {"id": 0, "name": "Car"},
        {"id": 1, "name": "Pedestrian"},
        {"id": 2, "name": "Cyclist"},
        {"id": 3, "name": "Van"},
        {"id": 4, "name": "Person_sitting"},
        {"id": 5, "name": "Truck"},
        {"id": 6, "name": "Tram"},
    ]

    categories_weakm3d = [
        {"id": 0, "name": "Car"},
    ]

    # 添加类别信息到 COCO 格式
    coco_format["categories"] = categories_weakm3d

    # 图像和标注ID初始化
    image_id = 0
    annotation_id = 0

    # 遍历KITTI标注文件
    for label_file in tqdm(train_path_list, desc="Processing KITTI Labels to gt form"):

        # 获取图像信息
        image_name = label_file + ".png"  # 假设图像格式为PNG，按需修改
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            continue  # 跳过没有对应图像的标注文件

        # 读取图像尺寸（可以使用opencv或PIL等库）
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size

        # 添加图像信息到COCO格式
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        # 读取标注文件
        with open(os.path.join(kitti_label_dir, label_file+".txt"), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                category_name = parts[0]

                # 查找类别ID
                category_id = next((cat["id"] for cat in categories_weakm3d if cat["name"].lower() == category_name.lower()),
                                   None)
                if category_id is None:
                    continue

                # 边界框 (xmin, ymin, xmax, ymax)
                xmin, ymin, xmax, ymax = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                width_bbox = xmax - xmin
                height_bbox = ymax - ymin

                # 添加标注信息到COCO格式
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [xmin, ymin, width_bbox, height_bbox],  # COCO格式要求的边界框格式 [x, y, width, height]
                    "area": width_bbox * height_bbox,  # 边界框面积
                    "iscrowd": 0  # 用于检测任务，通常为0
                })
                annotation_id += 1

        image_id += 1
        # pbar.update(1)

    # 将结果保存到JSON文件
    with open('eval_visual_tools/ground_truth/gt_3d_train.json', 'w') as f:
        json.dump(coco_format, f)


# to test pycocotools AP evaluation method, the output AP should be 1
def kitti_labels_to_dt_format():
    kitti_label_dir = 'data/kitti/KITTI3D/training/label_2'  # 标注文件的路径

    categories = [
        {"id": 0, "name": "Car"},
        # {"id": 1, "name": "Pedestrian"},
        # {"id": 2, "name": "Cyclist"},
        # {"id": 3, "name": "Van"},
        # {"id": 4, "name": "Person_sitting"},
        # {"id": 5, "name": "Truck"},
        # {"id": 6, "name": "Tram"},
    ]

    image_id = 0
    output = []
    # for each image
    for label_file in tqdm(os.listdir(kitti_label_dir), desc="Processing KITTI Labels to dt form"):
        with open(os.path.join(kitti_label_dir, label_file), 'r') as f:
            # for each bbox
            for line in f.readlines():
                parts = line.strip().split()

                category_name = parts[0]
                if category_name != "Car":
                    continue

                category_id = next((cat["id"] for cat in categories if cat["name"].lower() == category_name.lower()),
                                   None)

                if category_id is None:
                    print("category_id is none")
                    exit()

                xmin, ymin, xmax, ymax = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                width_bbox = xmax - xmin
                height_bbox = ymax - ymin

                coco_format = {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [xmin, ymin, width_bbox, height_bbox],
                    "score": 1
                }
                output.append(coco_format)

            image_id += 1

        # 将数据写入到json文件
        with open('eval_visual_tools/ground_truth/gt_to_dt_format.json', 'w') as f:
            json.dump(output, f)


kitti_labels_in_train_3d_to_coco_format()
kitti_labels_to_dt_format()












