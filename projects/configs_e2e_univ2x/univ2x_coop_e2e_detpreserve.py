_base_ = ['./univ2x_coop_e2e.py']

# Phase-1 validation config: preserve ego detection by skipping current direct
# cross-agent query fusion. Cooperative association will be added separately.
model_ego_agent = dict(
    use_learnable_fusion=False,
    use_detection_preserved_coop=True,
)

resume_from = None
