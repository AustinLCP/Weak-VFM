import os.path as osp
from copy import deepcopy
import os
import cv2
import mmcv
import numpy as np
import torch
import torch_scatter
from scipy.sparse.csgraph import connected_components
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.amg import (MaskData, area_from_rle,
                                        batch_iterator, batched_mask_to_box,
                                        box_xyxy_to_xywh,
                                        calculate_stability_score,
                                        mask_to_rle_pytorch, rle_to_mask)
from torchvision.ops.boxes import batched_nms

import utils
from convert_results import mapping_from_bin_to_npz
from SemanticSegmentor import NuImagesSegmentor

import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

class InstanceSegmentor:
    def __init__(self, nusc, config, device='cuda:0'):
        self.nusc = nusc
        self.generator_cfg = config.generator
        self.sam_cfg = config.sam
        self.device = device
        semantic_segmentor_cfg = config.semantic_segmentor
        self.semantic_segmentor = NuImagesSegmentor(semantic_segmentor_cfg.config,
                                                    semantic_segmentor_cfg.checkpoint,
                                                    device=device)
        self.classes = self.semantic_segmentor.model.dataset_meta['classes']
        assert self.sam_cfg.type in ('vit_h', 'vit_l', 'vit_b')
        sam = sam_model_registry[self.sam_cfg.type](self.sam_cfg.checkpoint).to(self.device)
        self.sam_segmentor = SamPredictor(sam)


    @staticmethod
    def visual_point_cloud(pc):
        point_cloud = pc.T  # 转置为 (N, 4)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
        plt.show()

    @staticmethod
    def visual_mask_on_img(image, mask):
        # Create a figure to display the image and mask
        plt.figure(figsize=(10, 10))

        # Display the original image
        # plt.subplot(2, 1, 1)
        # plt.imshow(image)
        # plt.title("Original Image")
        # plt.axis('off')

        # Display the mask on top of the image
        plt.subplot(1, 1, 1)
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)  # Overlay the mask with transparency
        plt.title("Output of SAM")
        plt.axis('off')

        plt.show()

    @staticmethod
    def save_htc_mask(mask, out_dict, cam_filename):
        f = osp.basename(cam_filename.replace('jpg', 'npz'))
        filename = cam_filename.split("\\")[-4] + "_" + f
        np.savez_compressed(osp.join(out_dict, 'htc_mask', 'semantic_' + filename), data=mask)

    @staticmethod
    def visual_points_on_img(img, proj_coords):
        img = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
        proj_coords = proj_coords.cpu().numpy() if isinstance(proj_coords, torch.Tensor) else proj_coords

        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.imshow(img)
        plt.scatter(proj_coords[:, 0], proj_coords[:, 1], c='r', s=2)
        plt.show()


    # =SemanticSegmentor.predict(), used in save_masks()
    def generate_semantic_mask(self, image, score_thr=0.3, return_numpy=True):
        coarse_semantic_mask = self.semantic_segmentor.predict(image, score_thr, return_numpy=return_numpy)
        return coarse_semantic_mask

    # used in sam_predict()
    def _process_batch(self, points, im_size):
        transformed_points = self.sam_segmentor.transform.apply_coords_torch(points, im_size) # 将点云数据从原始坐标系转换到适合模型预测的坐标系中
        in_points = torch.as_tensor(transformed_points, device=self.device) # 这些点可以用于指导分割模型聚焦于特定的图像区域
        in_labels = torch.ones(in_points.shape[:2], dtype=torch.int, device=self.device) # 输入点的标签，表示每个点是否是前景或背景。通常用0表示背景点，用1表示前景点
        # masks: (N, M, H, W)，其中 N 是输入点的数量，M 是每个点生成的mask数量，H 和 W 分别是mask的高度和宽度
        masks, iou_preds, _ = self.sam_segmentor.predict_torch(
            in_points,
            in_labels,
            multimask_output=True, # 模型会尝试生成多个可能的分割掩码，每个掩码代表一种可能的分割结果
            return_logits=True, # mask 的值是每个点的分数，之后通过阈值筛选，可转化为 true/false 的 mask
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1), # 将第一个和第二个维度合并，以便将多个 mask 平铺在一起
            iou_preds=iou_preds.flatten(0, 1),
            points=points.repeat_interleave(masks.shape[1], dim=0),
        )
        del masks

        # Filter by predicted IoU
        if self.sam_cfg.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.sam_cfg.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], 0.0, self.sam_cfg.stability_score_offset
        )
        if self.sam_cfg.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.sam_cfg.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > 0.0
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Compress to RLE
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]
        return data

    # used in generate_instance_mask()
    def sam_predict(self, image, point_coords):
        # image: (H, W, 3)
        # point_coords: (B, N, 2) [x, y]
        assert len(point_coords) > 0
        im_size = image.shape[:2] # (H,W)

        self.sam_segmentor.set_image(image) # 设置 sam_segmentor 的当前要处理的图像
        # Generate masks in batches
        data = MaskData()
        for (points,) in batch_iterator(self.sam_cfg.points_per_batch, point_coords):
            batch_data = self._process_batch(points, im_size)
            data.cat(batch_data) # 将当前批次处理的掩码数据合并到 data 中
            del batch_data # 删除当前批次数据以释放内存
        self.sam_segmentor.reset_image()

        # Remove duplicates， 对生成的掩码进行非极大值抑制（NMS），以去除重叠度较高的重复掩码
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.sam_cfg.box_nms_thresh,
        )
        data.filter(keep_by_nms) # 根据 NMS 的结果过滤掉重复的掩码
        data.to_numpy()
        data["segmentations"] = [rle_to_mask(rle) for rle in data["rles"]] # 将 RLE 格式的掩码转换为实际的掩码图

        anns = []
        for idx in range(len(data["segmentations"])):
            ann = {
                "segmentation": data["segmentations"][idx], # mask with the same size of the image(值是true 和 false)
                "area": area_from_rle(data["rles"][idx]), # 面积
                "bbox": box_xyxy_to_xywh(data["boxes"][idx]).tolist(), # [x, y, width, height]
                "predicted_iou": data["iou_preds"][idx].item(), # 预测的 IOU 值，用于表示掩码的准确度
                "point_coords": [data["points"][idx].tolist()],
                "stability_score": data["stability_score"][idx].item(), # 掩码的稳定性得分
            }
            anns.append(ann)
        return anns

    # used in save_masks()
    def generate_instance_mask(self, image, point_coords, coarse_semantic_mask, file_path):
        # image: (H, W, 3)
        # point_coords: (B, M, 2) [x, y], B is the number of instances in the current frame, M 是批次，初始化成None
        # coarse_semantic_mask: (H, W)
        assert len(point_coords) > 0
        masks = self.sam_predict(image, point_coords) # masks 是多种可能的实例分割结果的集合

        # mask_list = []
        # for mask in masks:
        #     mask_list.append(mask["segmentation"])
        # all_masks = np.logical_or.reduce(mask_list)
        # InstanceSegmentor.visual_mask_on_img(image, all_masks)
        # InstanceSegmentor.visual_mask_on_img(image, masks[50]['segmentation'])

        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True) # 按照掩码区域大小从大到小排序
        ignore_ids = [self.classes.index(class_name) for class_name in self.generator_cfg.ignore_semantics] # 包含需要忽略的语义类别的索引，这些类别通常是无需进一步处理或掩盖所有实例的类别
        # For example, the "barrier" class, considering that the distance between these instances is close,
        # there will be a mask covering all instances. At this time, it is not necessary to calculate the coverage,
        # and the subsequent instance mask can cover the previous overall mask.
        # For other categories, such as "car", the larger coverage of the subsequent sub-mask is only part of the vehicle instance,
        # so it cannot cover the overall mask.
        fine_semantic_mask = torch.zeros_like(coarse_semantic_mask) + len(self.classes)
        fine_instance_mask = torch.zeros_like(coarse_semantic_mask) - 1
        for i in range(len(sorted_masks)):
            temp_valid_mask = sorted_masks[i]['segmentation'] # 提取掩码的分割区域
            propose_classes_ids = coarse_semantic_mask[temp_valid_mask] # 通过 coarse_semantic_mask 找出这些实例的对应的类别
            class_id = torch.bincount(propose_classes_ids).argmax().item() # 找出出现最多的类别
            # 如果 class_id 合法并且不在忽略类别列表中
            if class_id < len(self.classes):
                if class_id not in ignore_ids:
                    semantics = fine_semantic_mask[temp_valid_mask]
                    coverage = 1 - torch.sum(semantics == len(self.classes)) / len(semantics) # 计算 temp_valid_mask 在 fine_semantic_mask 中的覆盖率 coverage，表示多少比例的像素已经被更新为实际类别
                    # 如果覆盖率超过阈值 and coarse_semantic_mask(HTC 生成的)在该区域的主类别和 temp_valid_mask(SAM 生成的)的一致
                    if coverage > self.generator_cfg.cover_threshold and torch.bincount(semantics).argmax().item() == class_id:
                        continue
                fine_semantic_mask[temp_valid_mask] = class_id
                fine_instance_mask[temp_valid_mask] = i


        # f = osp.basename(file_path).replace('jpg', 'npz')
        # filename = file_path.split("\\")[-4] + "_" + f
        # output_file = filename + '_bbox.txt'
        # bboxes = []
        # # 保存
        # for i in range(len(masks)):
        #     bbox = masks[i]["bbox"]
        #     bboxes.append(bbox)
        # # 使用 numpy 将 bbox 保存到 txt 文件中
        # bboxes = np.array(bboxes)
        # # np.savetxt(output_file, bboxes.reshape(-1, 4), fmt='%.6f', delimiter=' ')
        #
        # fig, ax = plt.subplots(1)
        # ax.imshow(image)
        # for bbox in bboxes:
        #     rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,
        #                              edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        # plt.axis('off')
        # plt.show()


        # InstanceSegmentor.visual_mask_on_img(image, fine_instance_mask)
        return fine_semantic_mask, fine_instance_mask

    # used in segment2D.py
    def save_masks(self, out_dir, sample_idx):
        my_sample = self.nusc.sample[sample_idx]
        pointsensor_token = my_sample['data']['LIDAR_TOP']
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        pc, lag_time, sweep_indices, no_close_masks = utils.load_pointcloud_multisweep_kitti(self.nusc, my_sample, sweeps=self.generator_cfg.multiple_sweeps)
        pc.points = pc.points[:, no_close_masks] # (4,N)

        # InstanceSegmentor.visual_point_cloud(pc.points)

        # camera_list = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        camera_list = ['CAM_FRONT']
        for cam_channel in camera_list:
            camera_token = my_sample['data'][cam_channel]
            cam = self.nusc.get('sample_data', camera_token)
            pc_for_project = deepcopy(pc)
            image = cv2.imread(osp.join(self.nusc.dataroot, cam['filename']))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # proj_points 投影后的点云坐标
            # proj_mask 哪些点被成功投影
            # proj_points, proj_mask = utils.map_proj_to_image(self.nusc, cam, pointsensor, pc_for_project, image)
            # 包含了有效投影点的2D坐标
            # proj_coords = np.concatenate((proj_points[0, proj_mask][:, None], proj_points[1, proj_mask][:, None]), axis=1)
            proj_coords, proj_mask = utils.map_proj_to_image_kitti(pc_for_project.points, image, cam["cam_to_cam_path"],cam["velo_to_cam_path"])
            proj_coords = torch.from_numpy(proj_coords)
            # proj_mask = torch.from_numpy(proj_mask)
            proj_coords_long = proj_coords.long() # (N,2)

            # InstanceSegmentor.visual_points_on_img(image, proj_coords_long)

            # 检测到物体的区域 -> 类别编号， 没检测到物体的区域 -> num_classes
            coarse_semantic_mask = self.generate_semantic_mask(image, self.generator_cfg.coarse_score_thr, return_numpy=False)
            coarse_semantic_mask = coarse_semantic_mask.to(device=proj_coords.device)

            # InstanceSegmentor.save_htc_mask(coarse_semantic_mask, out_dir, cam['filename'])

            # prompt_mask 是一个布尔掩码数组，有检测出物体的地方 true，没有检测出物体的地方 false
            prompt_mask = (coarse_semantic_mask[proj_coords_long[:, 1], proj_coords_long[:, 0]] != len(self.classes))
            unique_proj_coords = torch.unique(proj_coords_long[prompt_mask], dim=0).to(device=self.device)

            # InstanceSegmentor.visual_points_on_img(image, unique_proj_coords)

            # 进行最远点采样，确保选择的点在空间中尽可能分散
            if len(unique_proj_coords) > self.generator_cfg.max_prompts:
                # farthest point sampling
                coords = unique_proj_coords.float()
                indices = utils.farthest_point_sample(
                    coords[None, :, :], self.generator_cfg.max_prompts
                    )[0] .to(device=unique_proj_coords.device, dtype=torch.long)
                filtered_coords_prompt = unique_proj_coords[indices, None, :]

                # points = unique_proj_coords[indices,:]
                # InstanceSegmentor.visual_points_on_img(image,points)
            # 直接保留全部投影点
            elif len(unique_proj_coords) > 0:
                filtered_coords_prompt = unique_proj_coords[:, None, :]
            # 如果没有有效投影点，返回一个空的张量
            else:
                filtered_coords_prompt = unique_proj_coords.new_zeros((0, 1, 2))

            fine_semantic_mask = torch.zeros_like(coarse_semantic_mask) + len(self.classes)
            fine_instance_mask = torch.zeros_like(coarse_semantic_mask) - 1

            # filtered_coords_prompt 是投影在 semantic mask 识别出的物体上的点
            if len(filtered_coords_prompt) > 0:
                fine_semantic_mask, fine_instance_mask = self.generate_instance_mask(image, filtered_coords_prompt, coarse_semantic_mask, cam['filename'])

            fine_semantic_mask = fine_semantic_mask.cpu().numpy().astype(np.int32)
            fine_instance_mask = fine_instance_mask.cpu().numpy().astype(np.int32)
            # filename = osp.basename(cam['filename']).replace('jpg', 'npz')
            f = osp.basename(cam['filename']).replace('jpg', 'npz')
            filename = cam['filename'].split("\\")[-4] + "_" + f
            np.savez_compressed(osp.join(out_dir, cam_channel, 'semantic_' + filename), data=fine_semantic_mask)
            np.savez_compressed(osp.join(out_dir, cam_channel, 'instance_' + filename), data=fine_instance_mask)
            # print(filename)



class PointSAM:
    def __init__(self, nusc, config, device='cuda:0'):
        self.nusc = nusc
        self.cfg = config.PointSAM
        self.cluster_cfg = self.cfg.cluster
        self.device = device
        self.classes = self.cfg.CLASSES

    # used in segment3D.py
    def save_panoptic_labels(self, point_semantic_id, point_instance_id, out_dir, sample_idx, for_eval=False):
        my_sample = self.nusc.sample[sample_idx]
        pointsensor_token = my_sample['data']['LIDAR_TOP']
        valid_mask = point_instance_id != -1
        if valid_mask.any():
            assert point_semantic_id[valid_mask].max() < len(self.classes), 'sample_idx: {}'.format(sample_idx)
        if for_eval:
            filename = f"{pointsensor_token}_panoptic.npz"
            panoptic_label = np.zeros(len(point_semantic_id), dtype=np.uint16)
            panoptic_label[valid_mask] = mapping_from_bin_to_npz[point_semantic_id[valid_mask]] * 1000 + point_instance_id[valid_mask] + 1
            # np.savez_compressed(osp.join(out_dir, 'panoptic/val', filename), data=panoptic_label)
            p = filename.replace("\\", "_")
            np.savez_compressed(osp.join(out_dir, 'panoptic/val', p), data=panoptic_label)
        else:
            # filename = osp.basename(self.nusc.get('sample_data', pointsensor_token)['filename'])
            img_path = my_sample['data']['CAM_FRONT']
            date_path = img_path.split('\\')[3]  # eg 2011_10_03
            sync_path = img_path.split('\\')[4]  # eg 2011_10_03_drive_0027_sync
            file_name = img_path.split('\\')[-1].replace('png', 'bin') # eg 000142.bin
            output_path = osp.join("data\\kitti\\raw_data", date_path, sync_path, "pointSAM_label")
            if not osp.exists(output_path): 
                os.makedirs(output_path)

            panoptic_label = np.zeros((len(point_semantic_id), 2), dtype=np.uint16)
            panoptic_label[:, 0] = 65535
            panoptic_label[:, 1] = len(self.classes)
            panoptic_label[valid_mask, 0] = point_instance_id[valid_mask] # 所有有效点的实例 ID 被填充到 panoptic_label 的第一列
            panoptic_label[valid_mask, 1] = point_semantic_id[valid_mask] # 所有有效点的语义 ID 被填充到 panoptic_label 的第二列
            panoptic_label.tofile(osp.join(output_path, file_name))

    # used in SAR()
    def cluster(self, points, point_semantic_id, point_instance_id):
        # points: (N, 3)
        # point_semantic_id: (N,)
        # point_instance_id: (N,)
        if len(points) == 0:
            return torch.zeros_like(point_instance_id)
        if self.cluster_cfg.type == 'connected_components':
            # Inspired by LESS:http://arxiv.org/abs/2210.08064, set dynamic distance threshold for each point
            ndim = self.cluster_cfg.dim
            dist_mat = points[:, None, :ndim] - points[None, :, :ndim]
            dist_mat = (dist_mat ** 2).sum(2) ** 0.5
            dist_ego = torch.norm(points[:, :ndim], dim=1)
            dynamic_thresh = torch.maximum(dist_ego[:, None], dist_ego[None, :]) * self.cluster_cfg.dist_coef
            min_dist_thresh = torch.zeros_like(dynamic_thresh) + 1000.0
            # for each class, set different minimal distance threshold
            for class_name in self.classes:
                indices = point_semantic_id == self.classes.index(class_name)
                class_min_dist_thresh = torch.tensor(self.cluster_cfg.min_dist_thresh[class_name], device=points.device)
                min_dist_thresh[indices, :] = torch.min(class_min_dist_thresh, min_dist_thresh[indices, :])
                min_dist_thresh[:, indices] = torch.min(class_min_dist_thresh, min_dist_thresh[:, indices])
            dist_thresh = torch.clamp_min(dynamic_thresh, min_dist_thresh)
            adj_mat = dist_mat < dist_thresh
            if self.cluster_cfg.get('partition_different_class', False):
                # forcefully cut-off points with different semantic labels
                vehicle_mask = torch.zeros_like(point_semantic_id, dtype=torch.bool)
                for vehicle_class in self.cluster_cfg.vehicle_class:
                    vehicle_id = self.classes.index(vehicle_class)
                    vehicle_mask = torch.logical_or(vehicle_mask, point_semantic_id == vehicle_id)
                different_class_mask = point_semantic_id[:, None] != point_semantic_id[None, :]
                same_vehicle_mask = vehicle_mask[:, None] & vehicle_mask[None, :]
                adj_mat[different_class_mask & (~same_vehicle_mask)] = False
            for ignore_semantic_name in self.cluster_cfg.ignore_semantics:
                # find the points belonging to the same semantic labels but with different instances
                ignore_id = self.classes.index(ignore_semantic_name)
                mask = torch.logical_and(point_instance_id[:, None] != point_instance_id[None, :],
                                         point_semantic_id[:, None] == point_semantic_id[None, :])
                inds = torch.nonzero(mask)
                # the points belonging to the different masks but with the ignore semantic label should not be connected, such as "barrier"
                adj_mat[inds[:, 0][point_semantic_id[inds[:, 0]] == ignore_id], inds[:, 1][point_semantic_id[inds[:, 1]] == ignore_id]] = False
            adj_mat = adj_mat.cpu().numpy()
            n_components, point_instance_id_3d = connected_components(adj_mat, directed=False)
            point_instance_id_3d = torch.as_tensor(point_instance_id_3d, dtype=torch.int, device=point_instance_id.device)
        else:
            raise NotImplementedError
        return point_instance_id_3d

    # used in SAR()
    def id_merging(self,
                   point_instance_id_2d,
                   point_semantic_id_2d,
                   point_instance_id_3d,
                   min_points=0):
        if len(point_instance_id_2d) == 0:
            return torch.zeros_like(point_semantic_id_2d), torch.zeros_like(point_instance_id_2d)
        # point_instance_id: (N, 2), [point_instance_id_3d, point_instance_id_2d]
        point_instance_id = torch.hstack((point_instance_id_3d[:, None], point_instance_id_2d[:, None]))
        unq_cnt_3d = torch.unique(point_instance_id[:, 0], return_counts=True, dim=0)[1]

        # find the largest 3d instance for each 2d instance
        unq_id, unq_inv, unq_cnt = torch.unique(point_instance_id, return_inverse=True, return_counts=True, dim=0)  # (N1, 2), (N,), (N1,)
        unq_id_2d, unq_inv = torch.unique(unq_id[:, 1], return_inverse=True, dim=0)
        _, max_ind = torch_scatter.scatter_max(unq_cnt.float(), unq_inv)
        map_pairs = unq_id[max_ind]     # [point_instance_id_3d, point_instance_id_2d]
        for map_pair in map_pairs:
            id_3d = map_pair[0].item()
            id_2d = map_pair[1].item()
            mask = (point_instance_id[:, 0] != id_3d) & (point_instance_id[:, 1] == id_2d)
            point_instance_id[mask, 1] = -1

        # find the largest 2d instance for each 3d instance
        unq_id, unq_inv, unq_cnt = torch.unique(point_instance_id, return_inverse=True, return_counts=True, dim=0)  # (N1, 2), (N,), (N1,)
        unq_id_3d, unq_inv = torch.unique(unq_id[:, 0], return_inverse=True, dim=0) # (M_3d,), (N1,)
        max_cnt, max_ind = torch_scatter.scatter_max(unq_cnt.float(), unq_inv)
        map_pairs = unq_id[max_ind]      # [point_instance_id_3d, point_instance_id_2d]

        # if a 2d instance id is assigned to multiple 3d clusters
        # choose the one with the most points
        cluster_cnt_3d = unq_cnt_3d
        selected_map_pairs = dict()     # key: point_instance_id_2d, value: point_instance_id_3d
        for map_pair in map_pairs:
            id_3d = map_pair[0].item()
            id_2d = map_pair[1].item()
            if id_3d < 0 or id_2d < 0 or cluster_cnt_3d[id_3d] < min_points:
                continue
            assert id_2d not in selected_map_pairs
            selected_map_pairs[id_2d] = id_3d

        # merge
        point_semantic_id_merged = torch.zeros_like(point_semantic_id_2d) + len(self.classes)
        point_instance_id_merged = torch.zeros_like(point_instance_id_2d) - 1
        for idx, (id_2d, id_3d) in enumerate(selected_map_pairs.items()):
            assert len(torch.unique(point_semantic_id_2d[point_instance_id_2d == id_2d])) == 1
            point_semantic_id_merged[point_instance_id_3d == id_3d] = point_semantic_id_2d[point_instance_id_2d == id_2d][0]
            point_instance_id_merged[point_instance_id_3d == id_3d] = idx
        return point_semantic_id_merged, point_instance_id_merged

    # used in generate()
    def multiview_merging(self,
                      reference_point_semantic_id,
                      edition_point_semantic_id,
                      reference_point_instance_id,
                      edition_point_instance_id):
        # reference_point_semantic_id: (N,)
        # edition_point_semantic_id: (N,)
        # reference_point_instance_id: (N,)
        # edition_point_instance_id: (N,)
        reference_mask = reference_point_instance_id != -1
        edition_mask = edition_point_instance_id != -1
        edition_point_instance_id[edition_mask] += reference_point_instance_id.max() + 1
        intersection = torch.logical_and(reference_mask, edition_mask)
        intersection_inds = torch.nonzero(intersection).reshape(-1)
        if len(intersection_inds) == 0:
            reference_point_semantic_id[edition_mask] = edition_point_semantic_id[edition_mask]
            reference_point_instance_id[edition_mask] = edition_point_instance_id[edition_mask]
            return reference_point_semantic_id, reference_point_instance_id

        unique_instances, reference_counts = torch.unique(reference_point_instance_id[reference_mask], return_counts=True)
        reference_counts = dict(zip(unique_instances.cpu().numpy(), reference_counts.cpu().numpy()))
        unique_instances, edition_counts = torch.unique(edition_point_instance_id[edition_mask], return_counts=True)
        edition_counts = dict(zip(unique_instances.cpu().numpy(), edition_counts.cpu().numpy()))

        # Calculate the group number correspondence of overlapping points
        overlap = {}
        for inter_idx in intersection_inds:
            edition = edition_point_instance_id[inter_idx].item()
            reference = reference_point_instance_id[inter_idx].item()
            assert edition != -1 and reference != -1
            if edition not in overlap:
                overlap[edition] = {}
            if reference not in overlap[edition]:
                overlap[edition][reference] = 0
            overlap[edition][reference] += 1

        # Update edition_point_instance_id
        for edition, overlap_count in overlap.items():
            max_index = np.argmax(np.array(list(overlap_count.values())))
            reference = list(overlap_count.keys())[max_index]
            count = list(overlap_count.values())[max_index]
            total_count = float(min(reference_counts[reference], edition_counts[edition]))
            if (count / total_count) >= self.cfg.merge_ratio:
                semantic = reference_point_semantic_id[reference_point_instance_id == reference][0]
                edition_point_semantic_id[edition_point_instance_id == edition] = semantic
                edition_point_instance_id[edition_point_instance_id == edition] = reference

        reference_point_semantic_id[edition_mask] = edition_point_semantic_id[edition_mask]
        reference_point_instance_id[edition_mask] = edition_point_instance_id[edition_mask]

        return reference_point_semantic_id, reference_point_instance_id

    # used in segment3D.py
    def generate(self, sample_idx):
        my_sample = self.nusc.sample[sample_idx]
        pointsensor_token = my_sample['data']['LIDAR_TOP']
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        pc = utils.load_pointcloud(self.nusc, pointsensor)
        points = torch.tensor(pc.points[:3, :].T)

        point_semantic_id = torch.zeros(len(points), dtype=torch.int) + len(self.classes)
        point_instance_id = torch.zeros(len(points), dtype=torch.int) - 1
        # camera_list = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        camera_list = ['CAM_FRONT']
        for cam_channel in camera_list:
            camera_token = my_sample['data'][cam_channel]
            cam = self.nusc.get('sample_data', camera_token)
            pc_for_project = deepcopy(pc)
            # try:
            filtered_semantic_id, filtered_instance_id, total_mask = self.generate_single_cam(points,
                                                                                              pc_for_project,
                                                                                              pointsensor,
                                                                                              cam)
            # except:
            #     raise ValueError('sample_idx: {}, cam_channel: {}'.format(sample_idx, cam_channel))
            temp_point_semantic_id = torch.zeros_like(point_semantic_id) + len(self.classes)
            temp_point_instance_id = torch.zeros_like(point_instance_id) - 1
            temp_point_semantic_id[total_mask] = filtered_semantic_id
            temp_point_instance_id[total_mask] = filtered_instance_id
            point_semantic_id, point_instance_id = self.multiview_merging(
                point_semantic_id,
                temp_point_semantic_id,
                point_instance_id,
                temp_point_instance_id
            )
        # refine the multiview merged segmentations
        out_put = self.SAR(points, point_semantic_id, point_instance_id)
        (filtered_points, filtered_semantic_id, filtered_instance_id, filtered_mask) = out_put
        assert torch.isclose(filtered_points, points[filtered_mask]).all()
        point_semantic_id[:] = len(self.classes)
        point_instance_id[:] = -1
        point_semantic_id[filtered_mask] = filtered_semantic_id
        point_instance_id[filtered_mask] = filtered_instance_id

        point_semantic_id = point_semantic_id.cpu().numpy()
        point_instance_id = point_instance_id.cpu().numpy()

        return point_semantic_id, point_instance_id

    # used in generate()
    def generate_single_cam(self, points, pc_for_project, pointsensor_sd, cam_sd):
        # image_filename = osp.basename(cam_sd['filename'])
        f = osp.basename(cam_sd['filename'])
        image_filename = cam_sd['filename'].split("\\")[-4] + "_" + f

        # instance_filename = 'instance_' + image_filename.replace('jpg', 'npz')
        instance_filename = 'instance_' + image_filename + ".npz"
        instance_image = np.load(osp.join(self.cfg.mask_root, cam_sd['channel'], instance_filename))['data']
        instance_image = torch.from_numpy(instance_image)

        # semantic_filename = 'semantic_' + image_filename.replace('jpg', 'npz')
        semantic_filename = 'semantic_' + image_filename + ".npz"
        semantic_image = np.load(osp.join(self.cfg.mask_root, cam_sd['channel'], semantic_filename))['data']
        semantic_image = torch.from_numpy(semantic_image)

        # 把点云投影到 instance mask(2D image) 上
        proj_coords, proj_mask = utils.map_proj_to_image_kitti(pc_for_project.points, instance_image, cam_sd["cam_to_cam_path"], cam_sd["velo_to_cam_path"])
        # proj_coords = np.concatenate((proj_points[0, proj_mask][:, None], proj_points[1, proj_mask][:, None]), axis=1)
        proj_coords = torch.from_numpy(proj_coords)
        proj_mask = torch.from_numpy(proj_mask)
        proj_coords_long = proj_coords.long()

        # retain foreground points
        filtered_mask = (instance_image[proj_coords_long[:, 1], proj_coords_long[:, 0]] != -1)
        total_mask = torch.zeros_like(proj_mask)
        total_mask[proj_mask.nonzero().reshape(-1)] = filtered_mask

        filtered_points = points[total_mask]
        filtered_coords = proj_coords_long[filtered_mask]
        filtered_semantic_id = semantic_image[filtered_coords[:, 1], filtered_coords[:, 0]]
        filtered_instance_id = instance_image[filtered_coords[:, 1], filtered_coords[:, 0]]

        # employ Separability-Aware Refinement (SAR) to refine the instance and semantic segmentations
        out_put = self.SAR(filtered_points,
                           filtered_semantic_id,
                           filtered_instance_id,
                           filtered_coords)
        (output_points, output_semantic_id, output_instance_id, output_mask) = out_put
        total_mask[total_mask.nonzero().reshape(-1)] = output_mask
        return output_semantic_id, output_instance_id, total_mask

    # used in generate() and generate_single_cam()
    def SAR(self,
            points,
            point_semantic_id,
            point_instance_id,
            coords=None):
        # points: (N, 3 + C) [x, y, z, ...]
        # point_semantic_id: (N,), the semantic id of each point
        # point_instance_id: (N,), the instance id of each point
        # point_coords: (N, 2) [x, y], 2D pixel coordinates of the points (for debug)
        if len(points) == 0:
            # no need for further segmentation
            return points[[]], point_semantic_id[[]], point_instance_id[[]], points.new_zeros(len(points), dtype=torch.bool)

        mask = point_instance_id != -1
        points = points[mask]
        point_semantic_id = point_semantic_id[mask]
        point_instance_id = point_instance_id[mask]
        if coords is not None:
            coords = coords[mask]
        if mask.any():
            assert point_semantic_id.max() < len(self.classes)

        # point_instance_id_3d: the connected components ID of each point
        point_instance_id_3d = self.cluster(points, point_semantic_id, point_instance_id)

        point_semantic_id_merged, point_instance_id_merged = self.id_merging(point_instance_id,
                                                                             point_semantic_id,
                                                                             point_instance_id_3d,
                                                                             min_points=self.cluster_cfg.min_points)
        # filter out FP points
        temp_mask = point_instance_id_merged != -1
        points = points[temp_mask]
        point_semantic_id = point_semantic_id_merged[temp_mask]
        point_instance_id = point_instance_id_merged[temp_mask]
        if coords is not None:
            coords = coords[temp_mask]

        mask[mask.nonzero().reshape(-1)] = temp_mask

        return points, point_semantic_id, point_instance_id, mask
