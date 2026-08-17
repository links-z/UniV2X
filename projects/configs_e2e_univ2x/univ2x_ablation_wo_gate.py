_base_ = ['./univ2x_A_continue_unfreeze_bboxhead.py']

load_from = 'projects/work_dirs_e2e_univ2x/univ2x_lrmult01_600_multiframe_refine_freeze_bbox再低学习率600iter的基础上冻结bbox训练300/freeze_bbox_refine_iter_1.pth'
resume_from = None
auto_resume = False

work_dir = 'projects/work_dirs_e2e_univ2x/ablation_wo_gate_stage3_300'

checkpoint_config = dict(
    by_epoch=False,
    interval=100,
    max_keep_ckpts=10,
    filename_tmpl='wo_gate_iter_{}.pth'
)

model = dict(
    ablation_use_sinkhorn=True,
    ablation_use_gate=False,
    ablation_use_complement=True,
)
