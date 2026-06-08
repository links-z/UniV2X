_base_ = ['./univ2x_coop_e2e_sinkhorn_detadapt_pilot.py']

load_from = 'projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_sinkhorn_detadapt_pilot/epoch_1.pth'
resume_from = None
auto_resume = False

work_dir = 'projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_lowlr_finetune'

total_epochs = 3
runner = dict(type='EpochBasedRunner', max_epochs=3)

optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.0),
            'pts_bbox_head': dict(lr_mult=0.05),
            'map_head': dict(lr_mult=0.0),
            'motion_head': dict(lr_mult=0.0),
            'planning_head': dict(lr_mult=0.0),
            'bev_encoder': dict(lr_mult=0.0),
            'cross_agent_query_interaction': dict(lr_mult=1.0),
        }
    )
)

optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
