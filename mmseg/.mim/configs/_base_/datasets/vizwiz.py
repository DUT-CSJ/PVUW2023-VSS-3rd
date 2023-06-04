dataset_type = 'CustomDataset'
data_root = '/home/csj/desk2t/Code/Datastes/VizWiz-SalientObject/vizwiz'   # set your path
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='Resize', img_scale=(384, 384), keep_ratio=False),
    dict(type='Resize', img_scale=(853, 480), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(384, 384), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(384, 384), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384, 384),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        classes=('Background', 'Foreground'),
        palette=[[0, 0, 0], [1, 1, 1]],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        classes=('Background', 'Foreground'),
        palette=[[0, 0, 0], [1, 1, 1]],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        classes=('Background', 'Foreground'),
        palette=[[0, 0, 0], [255, 255, 255]],
        pipeline=test_pipeline)
    # test=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     img_dir='images/val',
    #     ann_dir='annotations/val',
    #     classes=('Background', 'Foreground'),
    #     palette=[[0, 0, 0], [1, 1, 1]],
    #     pipeline=test_pipeline),
)
