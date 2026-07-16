_base_ = ['./univ2x_multiframe_matchloss_from_ours.py']

load_from = 'projects/work_dirs_e2e_univ2x/univ2x_author_unfreeze_bboxhead_lrmult01/epoch_600iter.pth'
resume_from = None
auto_resume = False

work_dir = 'projects/work_dirs_e2e_univ2x/univ2x_lrmult01_600_multiframe_refine_freeze_bbox'

data = dict(workers_per_gpu=4)

total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)

checkpoint_config = dict(
    by_epoch=False,
    interval=300,
    max_keep_ckpts=10,
    filename_tmpl='freeze_bbox_refine_iter_{}.pth'
)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.0),
            'bev_encoder': dict(lr_mult=0.0),
            'map_head': dict(lr_mult=0.0),
            'motion_head': dict(lr_mult=0.0),
            'planning_head': dict(lr_mult=0.0),
            'cross_agent_query_interaction': dict(lr_mult=0.05),  # 2.5e-6
            'pts_bbox_head': dict(lr_mult=0.0),  # frozen
        }
    )
)
