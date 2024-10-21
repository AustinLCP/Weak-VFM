from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import sys
sys.path.append('/YOLO_World/yolo_world')


def AP_eval(dt_file, gt_file):
    # 加载标签数据和预测数据
    coco_gt = COCO(gt_file)  # 真实标签文件
    coco_dt = coco_gt.loadRes(dt_file)  # 预测结果文件

    # 获取类别ID
    car_category_id = coco_gt.getCatIds(catNms=['Car'])[0]

    # 创建评估对象，默认是 AP101
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    # 设置IoU阈值
    # coco_eval.params.iouThrs = [0.5]
    # coco_eval.params.iouThrs = [0.7]

    # AP11: 设置11个固定的召回率点：0.0, 0.1, 0.2, ..., 1.0
    # coco_eval.params.recThrs = np.linspace(0.0, 1.0, 11)

    # AP40: 设置40个固定的召回率点：0.0, 0.025, 0.05, ..., 0.975
    # coco_eval.params.recThrs = np.linspace(0.0, 1.0, 40)

    # 设置仅评估类别为“Car”的AP
    coco_eval.params.catIds = [car_category_id]

    # 运行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


    # Calculate F1-score
    precision = coco_eval.eval['precision'][0, :, 0, 0, 2]  # IoU=0.5~0.9, all categories, area range, max det 100
    recall = coco_eval.eval['recall'][0, 0, 0, 2]  # IoU=0.5~0.9, all categories, area range, max det 100

    # 计算F1 Score
    precision_mean = precision.mean()  # 平均precision
    recall_mean = recall.mean()  # 平均recall
    f1_score = 2 * (precision_mean * recall_mean) / (precision_mean + recall_mean)

    print(f'F1 Score: {f1_score:.4f}')


    print(" ")
    print("Ground Truth Annotations:", len(coco_gt.getAnnIds()))
    print("Detection Annotations:", sum(1 for det in coco_dt.dataset['annotations'] if det['category_id'] == 0))


dt_file = 'AP_calculation/yolo_world/dt_yolo_world_0.1_nms.json'
gt_file = 'AP_calculation/gt/gt_3d_train.json'
AP_eval(dt_file, gt_file)

