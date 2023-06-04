_base_ = [
    'upernet_swin_large_patch4_window7_512x512_'
    'pretrain_224x224_22K_160k_ade20k.py'
]
model = dict(
    backbone=dict(
        init_cfg=None,
        pretrain_img_size=384,
        window_size=12),
    decode_head=dict(
        num_classes=124,#),
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
    auxiliary_head=dict(
        num_classes=124)
)

optimizer = dict(
    lr=0.00003,
)

data = dict(samples_per_gpu=8,
            workers_per_gpu=16)
runner = dict(type='IterBasedRunner', max_iters=25000)
checkpoint_config = dict(by_epoch=False, interval=12500)
evaluation = dict(interval=12500, metric='mIoU', pre_eval=True)
