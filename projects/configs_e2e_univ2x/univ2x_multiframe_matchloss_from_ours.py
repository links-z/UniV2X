_base_ = ['./univ2x_coop_e2e_lowlr_finetune.py']

load_from = 'projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_sinkhorn_detadapt_pilot/epoch_1.pth'
resume_from = None
auto_resume = False

work_dir = 'projects/work_dirs_e2e_univ2x/univ2x_multiframe_matchloss_from_ours'
