import json
import os
from tqdm import tqdm
import torch
import os.path as osp
import cv2
import numpy as np

from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
import sys
sys.path.append('/YOLO_World/yolo_world')
# from YOLO_World.demo.simple_demo import inference as inference

# Non-Maximum Suppression
def nms(bboxes, scores, iou_threshold):
    """
    :param bboxes: 2D边界框坐标, (N, 4) 每个边界框为 (x1, y1, x2, y2)
    :param scores: 每个边界框对应的置信度得分, (N,)
    :param iou_threshold: IoU阈值, 用于决定是否移除重叠框
    :return: 保留的边界框索引列表
    """
    # 如果没有边界框，直接返回空
    if len(bboxes) == 0:
        return []

    # 初始化一个列表来存储保留的边界框索引
    keep = []

    # 计算每个边界框的面积
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 根据置信度得分对边界框进行排序
    order = scores.argsort()[::-1]

    while order.size > 0:
        # 当前置信度最高的边界框的索引
        i = order[0]
        keep.append(i)

        # 计算当前边界框与其他边界框的IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算重叠区域的宽度和高度
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # 计算IoU
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留 IoU 小于阈值的边界框
        inds = np.where(iou <= iou_threshold)[0]

        # 更新排序索引列表
        order = order[inds + 1]

    return keep


def yolo_inference(model, image_path, texts, test_pipeline, score_thr=0.1, max_dets=100):
    image = cv2.imread(image_path)
    image = image[:, :, [2, 1, 0]]
    data_info = dict(img=image, img_id=0, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    with torch.no_grad():
        output = model.test_step(data_batch)[0]
    pred_instances = output.pred_instances
    # score thresholding
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    # max detections
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()

    boxes = pred_instances['bboxes']
    labels = pred_instances['labels']
    scores = pred_instances['scores']
    label_texts = [texts[x][0] for x in labels]

    keep_indices = nms(boxes, scores, iou_threshold=0.7)
    boxes = boxes[keep_indices]

    result = {
        "image_path": image_path,
        "bboxes": boxes,
        "category_id": labels,
        "scores": scores,
    }
    return result

    # 读取批次中的所有图片并转换为模型的输入格式
    # images = []
    # data_samples = []
    # for img_path in image_paths:
    #     image = cv2.imread(img_path)
    #     image = image[:, :, [2, 1, 0]]
    #     data_info = dict(img=image, img_id=0, texts=texts)
    #     data_info = test_pipeline(data_info)
    #     images.append(data_info['inputs'].unsqueeze(0))  # 使用unsqueeze(0)扩展维度以匹配批次输入
    #     data_samples.append(data_info['data_samples'])
    #
    # # 将批次输入组合成一个张量
    # data_batch = dict(inputs=torch.cat(images, dim=0), data_samples=data_samples)
    #
    # # 模型推理
    # with torch.no_grad():
    #     outputs = model.test_step(data_batch)
    #
    # # 对每个输出结果进行处理
    # results = []
    # for output in outputs:
    #     pred_instances = output.pred_instances
    #     # 按分数阈值筛选结果
    #     pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    #     # 最多检测数量的处理
    #     if len(pred_instances.scores) > max_dets:
    #         indices = pred_instances.scores.float().topk(max_dets)[1]
    #         pred_instances = pred_instances[indices]
    #
    #     pred_instances = pred_instances.cpu().numpy()
    #     boxes = pred_instances['bboxes']
    #     labels = pred_instances['labels']
    #     scores = pred_instances['scores']
    #     label_texts = [texts[x][0] for x in labels]
    #
    #     result = {
    #         "bboxes": boxes,
    #         "category_id": labels,
    #         "scores": scores
    #     }
    #     results.append(result)

    # return results


def yolo_world_pred(batch_size,image_path_list):
    config_file = "YOLO_World/configs/segmentation/yolo_world_seg_m_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis.py"
    checkpoint = "YOLO_World/weights/yolo_world_seg_m_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis-ca465825.pth"

    cfg = Config.fromfile(config_file)
    cfg.work_dir = osp.join('./work_dirs')
    # init model
    cfg.load_from = checkpoint
    model = init_detector(cfg, checkpoint=checkpoint, device='cuda:0')
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)

    texts = [['Car']]
    # image_path_list = os.listdir("data/kitti/KITTI3D/training/image_2")

    results = []
    for image_path in tqdm(image_path_list,desc='YOLO-World predicting'):
        image = os.path.join("data/kitti/KITTI3D/training/image_2", image_path+".png")
        result = yolo_inference(model, image, texts, test_pipeline)
        results.append(result)

    # for i in tqdm(range(0, len(image_path_list), batch_size),desc='YOLO_World Predict'):
    #     # 获取当前批次的图像路径
    #     batch_image_paths = [os.path.join("data/kitti/KITTI3D/training/image_2", img_path+".png")
    #                          for img_path in image_path_list[i:i + batch_size]]
    #     # 批量推理
    #     batch_results = yolo_inference(model, batch_image_paths, texts, test_pipeline)
    #     results.extend(batch_results)

    return results


def yolo_world_pred_result_to_coco_format(kitti_predictions):
    output = []
    image_id = 0
    # for each image
    for result in kitti_predictions:  # 假设kitti_predictions是包含预测结果的列表
        # for each bbox
        for i in range(len(result["bboxes"])):
            box = result["bboxes"][i]
            xmin, ymin, xmax, ymax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            width_bbox = xmax - xmin
            height_bbox = ymax - ymin

            coco_format = {
                "image_id": image_id,
                "category_id": int(result["category_id"][i]),
                "bbox": [xmin, ymin, width_bbox, height_bbox],
                "score": float(result["scores"][i])
            }
            output.append(coco_format)
        image_id += 1

    # 将数据写入到json文件
    with open('AP_calculation/yolo_world/dt_yolo_world_0.1_nms.json', 'w') as f:
        json.dump(output, f)



image_path_list = np.loadtxt("data/kitti/data_file/split/train_3d.txt", dtype=str)
results = yolo_world_pred(16, image_path_list)
yolo_world_pred_result_to_coco_format(results)






