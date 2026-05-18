_base_ = ['./univ2x_coop_e2e.py']

# Fine-tune: unfreeze entire motion_head, freeze backbone/BEV encoder
# Load from original stg2 checkpoint (clean start for motion_head fine-tune)
load_from = "ckpts/univ2x_coop_e2e_stg2.pth"
resume_from = None

model_ego_agent = dict(
    freeze_for_finetune=True,
)

total_epochs = 6
optimizer = dict(
    type="AdamW",
    lr=2e-5,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.0),
        }
    ),
    weight_decay=0.01,
)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
