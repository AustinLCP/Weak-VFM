dataset_type = 'CocoDataset'
data_root = 'data/nuimages/'
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
# format for mmdet 2.x, format for mmdet 3.x in htc_without_semantic_r50_fpn_1x_nuim.py
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# format for mmdet==2.x
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(
#         type='Resize',
#         img_scale=[(1280, 720), (1920, 1080)],
#         multiscale_mode='range',
#         keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1600, 900),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

# format for mmdet3.x
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        scale=[(1280, 720), (1920, 1080)],
        mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=('img', 'gt_bboxes', 'gt_labels', 'gt_masks')),
    dict(type='Pad', size_divisor=32),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1600, 900), keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=['img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor']),
    # dict(type='ImageToTensor', keys=['img']),
]

# format for mmdet==2.x.
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/nuimages_v1.0-train.json',
#         img_prefix=data_root,
#         classes=class_names,
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/nuimages_v1.0-val.json',
#         img_prefix=data_root,
#         classes=class_names,
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/nuimages_v1.0-val.json',
#         img_prefix=data_root,
#         classes=class_names,
#         pipeline=test_pipeline))

# format for mmdet==3.x
train_dataloader = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/nuimages_v1.0-train.json',
        data_prefix=dict(img=data_root),
        metainfo=dict(classes=class_names),
        pipeline=train_pipeline),
)
val_dataloader = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/nuimages_v1.0-val.json',
        data_prefix=dict(img=data_root),
        metainfo=dict(classes=class_names),
        pipeline=test_pipeline),
)
test_dataloader = val_dataloader


evaluation = dict(metric=['bbox', 'segm'])
