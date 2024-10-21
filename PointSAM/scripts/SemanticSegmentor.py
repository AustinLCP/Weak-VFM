import argparse
import torch
import numpy as np
import mmcv
from mmdet.apis import init_detector, inference_detector # show_result_pyplot

from mmdet.registry import VISUALIZERS
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', default='result.jpg', help='Output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


class NuImagesSegmentor:
    def __init__(self, config, checkpoint, device='cuda:0'):
        self.device = device
        self.model = init_detector(config, checkpoint, device=device)
        # self.num_classes = len(self.model.CLASSES) # format for mmdet 2.x
        self.model.dataset_meta['classes'] = ['car','barrier']
        self.num_classes = len(self.model.dataset_meta['classes']) # format for mmdet 3.x

    # used in InstanceSegmentor.generate_semantic_mask()
    def predict(self, img, score_thr=0.3, return_numpy=False):
        h, w = img.shape[:2]
        if return_numpy:
            semantic_mask = np.zeros((h, w), dtype=np.int) + self.num_classes
        else:
            semantic_mask = torch.zeros((h, w), dtype=torch.int, device=self.device) + self.num_classes
        result = inference_detector(self.model, img)

        # bbox_result, segm_result = result # in mmdet=3.3.0, result is a DataDetInstance
        bbox_result_only = result.pred_instances.bboxes  # shape:[82,4],82 是检测到的目标数，4 是 [x1,y1,x2,y2] 两个对角的坐标，不包含置信度
        segm_result = result.pred_instances.masks  # shape:[82,h,w],82 是检测到的目标数，h,w 是图像的尺寸,数组的每一个元素是一个布尔值，表示图像中此处的像素是否属于该目标
        score_result = result.pred_instances.scores  # shape:[82],82 是检测到的目标数，其中的每个元素是一个浮点数，表示目标的置信度
        labels = result.pred_instances.labels  # shape:[82],82 是检测到的目标数，其中的每个元素是一个整数，表示检测到的目标类别

        # integrate bbox_result_only (82,4) with score_result (82,)
        score_result = score_result.unsqueeze(1)
        bbox_result = torch.cat((bbox_result_only, score_result), dim=1)  # shape (82, 5)
        bbox_result = [bbox.cpu().numpy() for bbox in bbox_result]
        # bboxes = np.vstack(bbox_result)

        if bbox_result:  # what if bbox_result is empty
            bboxes = np.vstack(bbox_result)
        else:
            bboxes = np.array([])

        # no need to concat
        # labels = [
        #     np.full(bbox.shape[0], i, dtype=np.int32)
        #     for i, bbox in enumerate(bbox_result)
        # ]
        # labels = np.concatenate(labels)

        if len(labels) > 0:
            # segms = mmcv.concat_list(segm_result) # no need to concat, not support in mmcv=2.1.0
            # segms = np.stack(segm_result, axis=0)
            segms = np.stack([segm.cpu().numpy() for segm in segm_result], axis=0)
            # filter out low score bboxes and masks
            assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            segms = segms[inds, :] # [N,h,w]
            for idx, segm in enumerate(segms):
                labels = labels.to(semantic_mask.dtype)  # 将 labels(Long) 转换为与 semantic_mask(Int) 相同的数据类型
                semantic_mask[segm] = labels[idx]  # 检测到物体的区域 -> 类别编号， 没检测到物体的区域 -> num_classes
        return semantic_mask


if __name__ == '__main__':
    args = parse_args()
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    img = mmcv.imread(args.img)
    result = inference_detector(model, img)
    # show the results
    # show_result_pyplot(model, args.img, result, out_file=args.out)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=0.3,
        show=False)

    img = visualizer.get_image()
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    cv2.imshow('result', img)

    cv2.waitKey(0)
