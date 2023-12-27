model = dict(
    type='PWCNet',
    encoder=dict(
        type='PWCNetEncoder',
        in_channels=3,
        net_type='Basic',
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ],
        out_channels=(16, 32, 64, 96, 128, 196),
        strides=(2, 2, 2, 2, 2, 2),
        dilations=(1, 1, 1, 1, 1, 1),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    decoder=dict(
        type='PWCNetDecoder',
        in_channels=dict(
            level6=81, level5=213, level4=181, level3=149, level2=117),
        flow_div=20.0,
        corr_cfg=dict(type='Correlation', max_displacement=4, padding=0),
        warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        scaled=False,
        post_processor=dict(type='ContextNet', in_channels=565),
        flow_loss=dict(
            type='MultiLevelEPE',
            p=2,
            reduction='sum',
            weights=dict(
                level2=0.005,
                level3=0.01,
                level4=0.02,
                level5=0.08,
                level6=0.32))),
    train_cfg=dict(),
    test_cfg=dict(),
    init_cfg=dict(
        type='Kaiming',
        nonlinearity='leaky_relu',
        layer=['Conv2d', 'ConvTranspose2d'],
        mode='fan_in',
        bias=0))
test_dataset_type = 'Sintel'
test_data_root = 'data/Sintel'
img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False)
global_transform = dict(
    translates=(0.05, 0.05),
    zoom=(1.0, 1.5),
    shear=(0.86, 1.16),
    rotate=(-10.0, 10.0))
relative_transform = dict(
    translates=(0.00375, 0.00375),
    zoom=(0.985, 1.015),
    shear=(1.0, 1.0),
    rotate=(-1.0, 1.0))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='ColorJitter',
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='GaussianNoise', sigma_range=(0, 0.04), clamp_range=(0.0, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        global_transform=dict(
            translates=(0.05, 0.05),
            zoom=(1.0, 1.5),
            shear=(0.86, 1.16),
            rotate=(-10.0, 10.0)),
        relative_transform=dict(
            translates=(0.00375, 0.00375),
            zoom=(0.985, 1.015),
            shear=(1.0, 1.0),
            rotate=(-1.0, 1.0))),
    dict(type='RandomCrop', crop_size=(384, 768)),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt'],
        meta_keys=('img_fields', 'ann_fields', 'filename1', 'filename2',
                   'ori_filename1', 'ori_filename2', 'filename_flow',
                   'ori_filename_flow', 'ori_shape', 'img_shape',
                   'img_norm_cfg'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputResize', exponent=6),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=[
            'flow_gt', 'filename1', 'filename2', 'ori_filename1',
            'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
            'scale_factor', 'pad_shape'
        ])
]
flyingthings3d_subset_train = dict(
    type='FlyingThings3DSubset',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            type='ColorJitter',
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5),
        dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(
            type='GaussianNoise',
            sigma_range=(0, 0.04),
            clamp_range=(0.0, 1.0)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.5, direction='vertical'),
        dict(
            type='RandomAffine',
            global_transform=dict(
                translates=(0.05, 0.05),
                zoom=(1.0, 1.5),
                shear=(0.86, 1.16),
                rotate=(-10.0, 10.0)),
            relative_transform=dict(
                translates=(0.00375, 0.00375),
                zoom=(0.985, 1.015),
                shear=(1.0, 1.0),
                rotate=(-1.0, 1.0))),
        dict(type='RandomCrop', crop_size=(384, 768)),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs', 'flow_gt'],
            meta_keys=('img_fields', 'ann_fields', 'filename1', 'filename2',
                       'ori_filename1', 'ori_filename2', 'filename_flow',
                       'ori_filename_flow', 'ori_shape', 'img_shape',
                       'img_norm_cfg'))
    ],
    data_root='data/FlyingThings3D_subset',
    test_mode=False,
    direction='forward',
    scene='left')
test_data_cleanpass = dict(
    type='Sintel',
    data_root='data/Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='InputResize', exponent=6),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=[
                'flow_gt', 'filename1', 'filename2', 'ori_filename1',
                'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                'scale_factor', 'pad_shape'
            ])
    ],
    test_mode=True,
    pass_style='clean')
test_data_finalpass = dict(
    type='Sintel',
    data_root='data/Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='InputResize', exponent=6),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=[
                'flow_gt', 'filename1', 'filename2', 'ori_filename1',
                'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                'scale_factor', 'pad_shape'
            ])
    ],
    test_mode=True,
    pass_style='final')
flyingthings3d_subset_train_x10000 = dict(
    type='RepeatDataset',
    times=10000,
    dataset=dict(
        type='FlyingThings3DSubset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='ColorJitter',
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5),
            dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
            dict(
                type='Normalize',
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0],
                to_rgb=False),
            dict(
                type='GaussianNoise',
                sigma_range=(0, 0.04),
                clamp_range=(0.0, 1.0)),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(
                type='RandomAffine',
                global_transform=dict(
                    translates=(0.05, 0.05),
                    zoom=(1.0, 1.5),
                    shear=(0.86, 1.16),
                    rotate=(-10.0, 10.0)),
                relative_transform=dict(
                    translates=(0.00375, 0.00375),
                    zoom=(0.985, 1.015),
                    shear=(1.0, 1.0),
                    rotate=(-1.0, 1.0))),
            dict(type='RandomCrop', crop_size=(384, 768)),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['imgs', 'flow_gt'],
                meta_keys=('img_fields', 'ann_fields', 'filename1',
                           'filename2', 'ori_filename1', 'ori_filename2',
                           'filename_flow', 'ori_filename_flow', 'ori_shape',
                           'img_shape', 'img_norm_cfg'))
        ],
        data_root='data/FlyingThings3D_subset',
        test_mode=False,
        direction='forward',
        scene='left'))
data = dict(
    train_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=5, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    train=dict(
        type='RepeatDataset',
        times=10000,
        dataset=dict(
            type='FlyingThings3DSubset',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='ColorJitter',
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.5),
                dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
                dict(
                    type='Normalize',
                    mean=[0.0, 0.0, 0.0],
                    std=[255.0, 255.0, 255.0],
                    to_rgb=False),
                dict(
                    type='GaussianNoise',
                    sigma_range=(0, 0.04),
                    clamp_range=(0.0, 1.0)),
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                dict(type='RandomFlip', prob=0.5, direction='vertical'),
                dict(
                    type='RandomAffine',
                    global_transform=dict(
                        translates=(0.05, 0.05),
                        zoom=(1.0, 1.5),
                        shear=(0.86, 1.16),
                        rotate=(-10.0, 10.0)),
                    relative_transform=dict(
                        translates=(0.00375, 0.00375),
                        zoom=(0.985, 1.015),
                        shear=(1.0, 1.0),
                        rotate=(-1.0, 1.0))),
                dict(type='RandomCrop', crop_size=(384, 768)),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['imgs', 'flow_gt'],
                    meta_keys=('img_fields', 'ann_fields', 'filename1',
                               'filename2', 'ori_filename1', 'ori_filename2',
                               'filename_flow', 'ori_filename_flow',
                               'ori_shape', 'img_shape', 'img_norm_cfg'))
            ],
            data_root='data/FlyingThings3D_subset',
            test_mode=False,
            direction='forward',
            scene='left')),
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='Sintel',
                data_root='data/Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True,
                pass_style='clean'),
            dict(
                type='Sintel',
                data_root='data/Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True,
                pass_style='final')
        ],
        separate_eval=True),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='Sintel',
                data_root='data/Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True,
                pass_style='clean'),
            dict(
                type='Sintel',
                data_root='data/Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True,
                pass_style='final')
        ],
        separate_eval=True))
optimizer = dict(
    type='Adam', lr=1e-05, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    by_epoch=False,
    gamma=0.5,
    step=[200000, 300000, 400000, 500000])
runner = dict(type='IterBasedRunner', max_iters=600000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=50000, metric='EPE')
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dir/pwcc_w/latest.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = 'work_dir/pwct_w'
gpu_ids = range(0, 1)