_base_ = ['./univ2x_coop_e2e_compA_ft(ablation).py']

load_from = 'ckpts/univ2x_coop_e2e_stg2.pth'
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
            'pts_bbox_head': dict(lr_mult=0.05),
            'map_head': dict(lr_mult=0.0),
            'motion_head': dict(lr_mult=0.0),
            'planning_head': dict(lr_mult=0.0),
            'bev_encoder': dict(lr_mult=0.0),
            'cross_agent_query_interaction': dict(lr_mult=1.0),
        }
    )
)
