import os
import numpy as np
from tqdm import tqdm
import os.path as osp
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from demo.simple_demo import inference


image_path_list = np.loadtxt("H:\\Honours\\WeakM3D\\WeakM3D\\data\\kitti\\data_file\\split\\train_raw.txt", dtype=str)
# image_path_list = np.loadtxt("train_raw_sample.txt", dtype=str)

config_file = "configs/segmentation/yolo_world_seg_m_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis.py"
checkpoint = "weights/yolo_world_seg_m_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis-ca465825.pth"

cfg = Config.fromfile(config_file)
cfg.work_dir = osp.join('./work_dirs')
cfg.load_from = checkpoint
model = init_detector(cfg, checkpoint=checkpoint, device='cuda:0')
test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
test_pipeline = Compose(test_pipeline_cfg)

texts = [['car']]

for image_path in tqdm(image_path_list, desc='Saving 2d bbox and masks'):

    image_path = image_path.tolist()[0]
    image_path = "H:\\Honours\\WeakM3D\\WeakM3D\\" + image_path

    bbox2d_path =image_path.replace('image_02', 'YOLO_seg_bbox').replace('png', 'txt')
    mask_path = image_path.replace('image_02', 'YOLO_seg_mask').replace('png', 'npz')

    seg_mask_dir = os.path.dirname(bbox2d_path)
    seg_bbox_dir = os.path.dirname(mask_path)
    if not os.path.exists(seg_mask_dir):
        os.makedirs(seg_mask_dir)
    if not os.path.exists(seg_bbox_dir):
        os.makedirs(seg_bbox_dir)

    results = inference(model, image_path, texts, test_pipeline, score_thr=0.2)

    boxes = results[0]
    logits = results[3]
    masks = results[4]

    if boxes.shape[0] > 0:
        bbox2d = np.hstack([boxes.reshape(-1, 4), logits.reshape(-1, 1)])
        np.savetxt(bbox2d_path, bbox2d, fmt='%f')
        np.savez(mask_path, masks=masks)



























