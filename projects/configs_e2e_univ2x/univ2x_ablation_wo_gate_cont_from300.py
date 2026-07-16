_base_ = ['./univ2x_ablation_wo_gate.py']

load_from = None
resume_from = 'projects/work_dirs_e2e_univ2x/univ2x_ablation_wo_gate/wo_gate_iter_300.pth'
auto_resume = False

work_dir = 'projects/work_dirs_e2e_univ2x/univ2x_ablation_wo_gate_cont_from300'

data = dict(workers_per_gpu=4)

total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)

checkpoint_config = dict(
    by_epoch=False,
    interval=100,
    max_keep_ckpts=20,
    filename_tmpl='wo_gate_s2_iter_{}.pth'
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
            'cross_agent_query_interaction': dict(lr_mult=0.05),
            'pts_bbox_head': dict(lr_mult=0.02),
        }
    )
)
