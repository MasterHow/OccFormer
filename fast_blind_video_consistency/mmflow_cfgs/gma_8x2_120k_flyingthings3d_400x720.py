model = dict(
    type='RAFT',
    num_levels=4,
    radius=4,
    cxt_channels=128,
    h_channels=128,
    encoder=dict(
        type='RAFTEncoder',
        in_channels=3,
        out_channels=256,
        net_type='Basic',
        norm_cfg=dict(type='IN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['InstanceNorm2d'], val=1, bias=0)
        ]),
    cxt_encoder=dict(
        type='RAFTEncoder',
        in_channels=3,
        out_channels=256,
        net_type='Basic',
        norm_cfg=dict(type='SyncBN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['SyncBatchNorm2d'], val=1, bias=0)
        ]),
    decoder=dict(
        type='GMADecoder',
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=12,
        corr_op_cfg=dict(type='CorrLookup', align_corners=True),
        gru_type='SeqConv',
        heads=1,
        motion_channels=128,
        position_only=False,
        max_pos_size=None,
        flow_loss=dict(type='SequenceLoss'),
        act_cfg=dict(type='ReLU')),
    freeze_bn=False,
    train_cfg=dict(),
    test_cfg=dict(iters=32))
train_dataset_type = 'FlyingThings3D'
train_data_root = 'data/flyingthings3d'
test_dataset_type = 'Sintel'
test_data_root = 'data/Sintel'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='ColorJitter',
        asymmetric_prob=0.2,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1592356687898089),
    dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
    dict(
        type='SpacialTransform',
        spacial_prob=0.8,
        stretch_prob=0.8,
        crop_size=(400, 720),
        min_scale=-0.4,
        max_scale=0.8,
        max_stretch=0.2),
    dict(type='RandomCrop', crop_size=(400, 720)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.1, direction='vertical'),
    dict(type='Validation', max_flow=1000.0),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=[
            'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
            'filename_flow', 'ori_filename_flow', 'ori_shape', 'img_shape',
            'erase_bounds', 'erase_num', 'scale_factor'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputPad', exponent=3),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=[
            'flow_gt', 'filename1', 'filename2', 'ori_filename1',
            'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
            'scale_factor', 'pad_shape', 'pad'
        ])
]
train_dataset_cleanpass = dict(
    type='FlyingThings3D',
    data_root='data/flyingthings3d',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            type='ColorJitter',
            asymmetric_prob=0.2,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1592356687898089),
        dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
        dict(
            type='SpacialTransform',
            spacial_prob=0.8,
            stretch_prob=0.8,
            crop_size=(400, 720),
            min_scale=-0.4,
            max_scale=0.8,
            max_stretch=0.2),
        dict(type='RandomCrop', crop_size=(400, 720)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.1, direction='vertical'),
        dict(type='Validation', max_flow=1000.0),
        dict(
            type='Normalize',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            to_rgb=False),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs', 'flow_gt', 'valid'],
            meta_keys=[
                'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
                'filename_flow', 'ori_filename_flow', 'ori_shape', 'img_shape',
                'erase_bounds', 'erase_num', 'scale_factor'
            ])
    ],
    test_mode=False,
    pass_style='clean',
    scene='left')
train_dataset_finalpass = dict(
    type='FlyingThings3D',
    data_root='data/flyingthings3d',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            type='ColorJitter',
            asymmetric_prob=0.2,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1592356687898089),
        dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
        dict(
            type='SpacialTransform',
            spacial_prob=0.8,
            stretch_prob=0.8,
            crop_size=(400, 720),
            min_scale=-0.4,
            max_scale=0.8,
            max_stretch=0.2),
        dict(type='RandomCrop', crop_size=(400, 720)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.1, direction='vertical'),
        dict(type='Validation', max_flow=1000.0),
        dict(
            type='Normalize',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            to_rgb=False),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs', 'flow_gt', 'valid'],
            meta_keys=[
                'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
                'filename_flow', 'ori_filename_flow', 'ori_shape', 'img_shape',
                'erase_bounds', 'erase_num', 'scale_factor'
            ])
    ],
    test_mode=False,
    pass_style='final',
    scene='left')
test_data_cleanpass = dict(
    type='Sintel',
    data_root='data/Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='InputPad', exponent=3),
        dict(
            type='Normalize',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=[
                'flow_gt', 'filename1', 'filename2', 'ori_filename1',
                'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                'scale_factor', 'pad_shape', 'pad'
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
        dict(type='InputPad', exponent=3),
        dict(
            type='Normalize',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=[
                'flow_gt', 'filename1', 'filename2', 'ori_filename1',
                'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                'scale_factor', 'pad_shape', 'pad'
            ])
    ],
    test_mode=True,
    pass_style='final')
data = dict(
    train_dataloader=dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        drop_last=True,
        persistent_workers=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    train=[
        dict(
            type='FlyingThings3D',
            data_root='data/flyingthings3d',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='ColorJitter',
                    asymmetric_prob=0.2,
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1592356687898089),
                dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
                dict(
                    type='SpacialTransform',
                    spacial_prob=0.8,
                    stretch_prob=0.8,
                    crop_size=(400, 720),
                    min_scale=-0.4,
                    max_scale=0.8,
                    max_stretch=0.2),
                dict(type='RandomCrop', crop_size=(400, 720)),
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                dict(type='RandomFlip', prob=0.1, direction='vertical'),
                dict(type='Validation', max_flow=1000.0),
                dict(
                    type='Normalize',
                    mean=[127.5, 127.5, 127.5],
                    std=[127.5, 127.5, 127.5],
                    to_rgb=False),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['imgs', 'flow_gt', 'valid'],
                    meta_keys=[
                        'filename1', 'filename2', 'ori_filename1',
                        'ori_filename2', 'filename_flow', 'ori_filename_flow',
                        'ori_shape', 'img_shape', 'erase_bounds', 'erase_num',
                        'scale_factor'
                    ])
            ],
            test_mode=False,
            pass_style='clean',
            scene='left'),
        dict(
            type='FlyingThings3D',
            data_root='data/flyingthings3d',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='ColorJitter',
                    asymmetric_prob=0.2,
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1592356687898089),
                dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
                dict(
                    type='SpacialTransform',
                    spacial_prob=0.8,
                    stretch_prob=0.8,
                    crop_size=(400, 720),
                    min_scale=-0.4,
                    max_scale=0.8,
                    max_stretch=0.2),
                dict(type='RandomCrop', crop_size=(400, 720)),
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                dict(type='RandomFlip', prob=0.1, direction='vertical'),
                dict(type='Validation', max_flow=1000.0),
                dict(
                    type='Normalize',
                    mean=[127.5, 127.5, 127.5],
                    std=[127.5, 127.5, 127.5],
                    to_rgb=False),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['imgs', 'flow_gt', 'valid'],
                    meta_keys=[
                        'filename1', 'filename2', 'ori_filename1',
                        'ori_filename2', 'filename_flow', 'ori_filename_flow',
                        'ori_shape', 'img_shape', 'erase_bounds', 'erase_num',
                        'scale_factor'
                    ])
            ],
            test_mode=False,
            pass_style='final',
            scene='left')
    ],
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='Sintel',
                data_root='data/Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputPad', exponent=3),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape', 'pad'
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
                    dict(type='InputPad', exponent=3),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape', 'pad'
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
                    dict(type='InputPad', exponent=3),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape', 'pad'
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
                    dict(type='InputPad', exponent=3),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape', 'pad'
                        ])
                ],
                test_mode=True,
                pass_style='final')
        ],
        separate_eval=True))
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dir/gma/gma_chairs8x2_120_sync_1227/latest.pth'
resume_from = None
workflow = [('train', 1)]
optimizer = dict(
    type='AdamW',
    lr=0.000125,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-05,
    amsgrad=False)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
lr_config = dict(
    policy='OneCycle',
    max_lr=0.000125,
    total_steps=120100,
    pct_start=0.05,
    anneal_strategy='linear')
runner = dict(type='IterBasedRunner', max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='EPE')
work_dir = 'work_dir/gma/1228_gma_things8x2_120k'
gpu_ids = range(0, 1)