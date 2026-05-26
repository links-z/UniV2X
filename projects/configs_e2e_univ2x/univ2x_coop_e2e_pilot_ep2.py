_base_ = ['./univ2x_coop_e2e_sinkhorn_detadapt_pilot.py']

load_from = 'projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_sinkhorn_detadapt_pilot/epoch_1.pth'
resume_from = None

total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.0),
            'bev_encoder': dict(lr_mult=0.0),
            'pts_bbox_head': dict(lr_mult=0.02),
            'cross_agent_query_interaction': dict(lr_mult=0.5),
            'map_head': dict(lr_mult=0.0),
            'motion_head': dict(lr_mult=0.0),
            'planning_head': dict(lr_mult=0.0),
        }
    )
)
