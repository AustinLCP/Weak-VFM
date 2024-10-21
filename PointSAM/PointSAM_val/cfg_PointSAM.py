semantic_segmentor = dict(
    config='PointSAM/configs/nuimages/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim.py',
    checkpoint='PointSAM/ckpt/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim_20201008_211222-0b16ac4b.pth'
)
sam = dict(
    type='vit_h',
    checkpoint='PointSAM/ckpt/sam_vit_h_4b8939.pth',
    points_per_batch=256,
    pred_iou_thresh=0.7,  # SAM 的输出的iou_preds
    stability_score_thresh=0.8,
    stability_score_offset=1.0,
    box_nms_thresh=0.7, # NMS 操作中用于过滤重叠边界框的 IoU 的阈值 (不打算调整)
)
generator = dict(
    multiple_sweeps=[0, 1, -1],
    coarse_score_thr=0.3,
    max_prompts=256,  # SAM 生成的 prompt 个数若超过这个阈值，将进行最远点采样
    ignore_semantics=['barrier'],
    cover_threshold= 0.3
)
PointSAM = dict(
    merge_ratio=0.5,
    # CLASSES=[
    #     'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    #     'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    # ],
    CLASSES=[
            'car', 'barrier'
    ],
    filter_ground=False,
    cluster = dict(
        type='connected_components',
        dist_coef=0.04,
        min_dist_thresh={'car': 1.0,
                         'truck': 1.0,
                         'trailer': 1.0,
                         'bus': 1.0,
                         'construction_vehicle': 1.0,
                         'bicycle': 0.65,
                         'motorcycle': 0.65,
                         'pedestrian': 0.65,
                         'traffic_cone': 0.65,
                         'barrier': 0.65,
                        },
        dim=3,
        min_points=0,
        partition_different_class=True,
        # vehicle_class=['car', 'truck', 'bus', 'construction_vehicle'],
        # ignore_semantics=['barrier', 'pedestrian'],
        vehicle_class=['car'],
        ignore_semantics=['barrier'],
    )
)