import json
import os
from tqdm import tqdm


# Only Car
def weakM3D_pred_result_to_coco_format():
    weakM3D_2d_bbox_path = 'data/kitti/KITTI3D/training/rgb_detections/train'  # 标注文件的路径

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
    for label_file in tqdm(os.listdir(weakM3D_2d_bbox_path), desc="Processing WeakM3D 2d bbox results to dt form"):
        with open(os.path.join(weakM3D_2d_bbox_path, label_file), 'r') as f:
            # for each bbox
            for line in f.readlines():
                parts = line.strip().split()

                category_name = parts[0]
                if category_name.lower() != "car":
                    continue

                category_id = next((cat["id"] for cat in categories if cat["name"].lower() == category_name.lower()),
                                   None)

                xmin, ymin, xmax, ymax = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                width_bbox = xmax - xmin
                height_bbox = ymax - ymin

                coco_format = {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [xmin, ymin, width_bbox, height_bbox],
                    "score": float(parts[5])
                }
                output.append(coco_format)

            image_id += 1

        # 将数据写入到json文件
        with open('eval_visual_tools/weakm3d/dt_weakM3D.json', 'w') as f:
            json.dump(output, f)


weakM3D_pred_result_to_coco_format()