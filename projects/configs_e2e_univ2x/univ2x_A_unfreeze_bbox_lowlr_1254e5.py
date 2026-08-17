_base_ = ['./univ2x_multiframe_matchloss_from_ours.py']

# 在 A_unfreeze_bbox_iter_300iter.pth 基础上，以 cont_from600 epoch3 缩减后的学习率 1.254e-5 继续训练
load_from = 'projects/work_dirs_e2e_univ2x/univ2x_A_continue_unfreeze_bboxhead/A_unfreeze_bbox_iter_300iter.pth'
resume_from = None
auto_resume = False

work_dir = 'projects/work_dirs_e2e_univ2x/univ2x_A_unfreeze_bbox_lowlr_1254e5'

data = dict(workers_per_gpu=4)

total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)

# 不继承 base config 的 lr scheduler，手动固定 lr=1.254e-5
lr_config = dict(_delete_=True, policy='fixed')

checkpoint_config = dict(
    by_epoch=False,
    interval=100,
    max_keep_ckpts=20,
    filename_tmpl='A_unfreeze_bbox_lowlr_iter_{}.pth'
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
            'pts_bbox_head': dict(lr_mult=0.02),   # 1.254e-5 * 0.02 = 2.5e-7
        }
    )
)
