_base_ = ['./univ2x_coop_e2e.py']

# Fine-tune: only train new inf cross-attention layers (A1+A2+A3)
# Load from stg2 checkpoint (not stg1)
load_from = "projects/work_dirs_e2e_univ2x/finetune_A123/epoch_2.pth"
resume_from = None

model_ego_agent = dict(
    freeze_for_finetune=True,
)

# Continue from epoch_2, train 3 more epochs
total_epochs = 3
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.0),
        }
    ),
    weight_decay=0.01,
)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
