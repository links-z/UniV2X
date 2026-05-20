"""
Quick sanity check for LearnableAgentQueryFusion.
Run: cd /root/autodl-tmp/UniV2X-zq/UniV2X && python test_learnable_fusion.py
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

# ---- minimal Instances stub (mirrors track_head_plugin.Instances) ----
from projects.mmdet3d_plugin.univ2x.dense_heads.track_head_plugin import Instances

def make_instances(n, embed_dims=256, device='cpu'):
    inst = Instances((1, 1))
    inst.query = torch.randn(n, embed_dims * 2, device=device)
    inst.ref_pts = torch.sigmoid(torch.randn(n, 3, device=device))
    inst.scores = torch.rand(n, device=device)
    inst.obj_idxes = torch.zeros(n, dtype=torch.long, device=device)  # all valid
    inst.pred_boxes = torch.randn(n, 10, device=device)
    inst.matched_gt_idxes = torch.full((n,), -1, dtype=torch.long, device=device)
    inst.disappear_time = torch.zeros(n, dtype=torch.long, device=device)
    inst.iou = torch.zeros(n, device=device)
    inst.track_scores = torch.zeros(n, device=device)
    inst.output_embedding = torch.zeros(n, embed_dims, device=device)
    return inst

def main():
    from projects.mmdet3d_plugin.univ2x.fusion_modules.learnable_agent_fusion import LearnableAgentQueryFusion

    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    module = LearnableAgentQueryFusion(pc_range=pc_range, embed_dims=256, num_sinkhorn_iters=5)
    module.train()

    N_v, N_i = 9, 7
    veh = make_instances(N_v)
    inf = make_instances(N_i)

    # identity transform: ego2other_rt[0] must be [4,4]
    ego2other_rt = [torch.eye(4)]
    other_pc_range = pc_range

    print(f"Input: veh={N_v} queries, inf={N_i} queries")
    out = module(inf, veh, ego2other_rt, other_pc_range)
    print(f"Output: {len(out)} queries (expected >= {N_v})")

    if hasattr(out, '_match_loss'):
        loss = out._match_loss
        print(f"match_loss = {loss.item():.4f}")
        loss.backward()
        print("Backward pass: OK")
    else:
        print("WARNING: no _match_loss found (check training mode)")

    # check no nan
    assert not torch.isnan(out.query).any(), "NaN in output query!"
    print("No NaN: OK")
    print("\nAll checks passed.")

if __name__ == '__main__':
    main()
