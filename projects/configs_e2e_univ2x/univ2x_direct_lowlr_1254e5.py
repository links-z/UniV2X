_base_ = ['./univ2x_multiframe_matchloss_from_ours.py']

# Load Stage2 weights only — do NOT restore epoch/iter/scheduler state
load_from = 'ckpts/freeze_bbox_refine_iter_tage2.pth'
resume_from = None
auto_resume = False

work_dir = 'projects/work_dirs_e2e_univ2x/univ2x_direct_lowlr_1254e5'

data = dict(workers_per_gpu=4)

total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)

# Fix lr at exactly 1.254e-05, no cosine decay
lr_config = dict(_delete_=True, policy='fixed')

optimizer = dict(
    type='AdamW',
    lr=1.254e-5,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.0),
            'bev_encoder': dict(lr_mult=0.0),
            'map_head': dict(lr_mult=0.0),
            'motion_head': dict(lr_mult=0.0),
            'planning_head': dict(lr_mult=0.0),
            'cross_agent_query_interaction': dict(lr_mult=0.05),
            'pts_bbox_head': dict(lr_mult=0.02),
        }
    )
)

checkpoint_config = dict(
    by_epoch=False,
    interval=100,
    max_keep_ckpts=10,
    filename_tmpl='direct_lowlr_iter_{}.pth'
)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)
