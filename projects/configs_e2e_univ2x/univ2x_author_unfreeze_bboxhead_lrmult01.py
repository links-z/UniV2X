_base_ = ['./univ2x_coop_e2e_lowlr_finetune.py']

load_from = 'ckpts/univ2x_coop_e2e_stg2.pth'
resume_from = None
auto_resume = False

work_dir = 'projects/work_dirs_e2e_univ2x/univ2x_author_unfreeze_bboxhead_lrmult01'

data = dict(workers_per_gpu=4)

total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)

checkpoint_config = dict(by_epoch=False, interval=300)

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
            'cross_agent_query_interaction': dict(lr_mult=1.0),
            'pts_bbox_head': dict(lr_mult=0.1),
        }
    )
)
