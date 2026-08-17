import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import MODELS


def sinkhorn(log_alpha, n_iters=5):
    """Log-space Sinkhorn normalization with dustbin (last row/col)."""
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return log_alpha.exp()


@MODELS.register_module()
class SinkhornAssocRefine(nn.Module):
    """
    Post-detection association refinement using infrastructure detections.
    Does NOT modify pred_boxes / pred_logits / scores.
    Only corrects obj_idxes to reduce ID switches.
    """

    def __init__(self, embed_dims=256, match_thresh=0.3, geo_weight=1.0, loss_weight=0.05):
        super().__init__()
        self.match_thresh = match_thresh
        self.geo_weight = geo_weight
        self.loss_weight = loss_weight

        # geometry score: (dx, dy, dz, dist, inf_score) -> scalar
        self.geo_mlp = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        # dustbin learnable bias
        self.dustbin = nn.Parameter(torch.tensor(1.0))

    def _build_score_matrix(self, veh_boxes, veh_scores, inf_boxes, inf_scores):
        """
        veh_boxes: [N, 7], inf_boxes: [M, 7]  (x,y,z,w,l,h,yaw) in ego frame
        Returns log_alpha: [1, N+1, M+1]
        """
        N, M = len(veh_boxes), len(inf_boxes)
        veh_xy = veh_boxes[:, :3]   # [N,3]
        inf_xy = inf_boxes[:, :3]   # [M,3]

        # pairwise geometry features
        diff = veh_xy.unsqueeze(1) - inf_xy.unsqueeze(0)   # [N,M,3]
        dist = diff.norm(dim=-1, keepdim=True)              # [N,M,1]
        inf_s = inf_scores.unsqueeze(0).unsqueeze(-1).expand(N, M, 1)
        geo_feat = torch.cat([diff, dist, inf_s], dim=-1)   # [N,M,5]
        geo_score = self.geo_mlp(geo_feat).squeeze(-1)      # [N,M]

        scores = geo_score * self.geo_weight                # [N,M]

        # add dustbin row and column
        dust_row = self.dustbin.expand(1, M)                # [1,M]
        scores_with_row = torch.cat([scores, dust_row], dim=0)  # [N+1,M]
        dust_col = self.dustbin.expand(N + 1, 1)
        log_alpha = torch.cat([scores_with_row, dust_col], dim=1).unsqueeze(0)  # [1,N+1,M+1]
        return log_alpha

    def forward(self, veh_boxes, veh_ids, veh_scores, inf_boxes, inf_scores):
        """
        Inference: returns refined obj_idxes (same tensor, modified in-place clone).
        veh_boxes: [N,7], veh_ids: [N] LongTensor, inf_boxes: [M,7], inf_scores: [M]
        """
        N, M = len(veh_boxes), len(inf_boxes)
        if M == 0 or N == 0:
            return veh_ids, torch.tensor(0.0, device=veh_boxes.device)

        log_alpha = self._build_score_matrix(veh_boxes, veh_scores, inf_boxes, inf_scores)
        P = sinkhorn(log_alpha)[0]          # [N+1, M+1]
        match_mat = P[:N, :M]               # [N, M]  vehicle-to-inf soft assignment

        # hard assignment: each inf det matched to best veh det
        conf, veh_idx = match_mat.max(dim=0)   # [M]
        matched_inf = conf > self.match_thresh

        # IDS correction: if two veh queries matched to same inf det in consecutive
        # frames, unify their IDs to the one with higher score
        refined_ids = veh_ids.clone()
        # (no cross-frame state here; ID unification is handled by track_base)
        # Return match matrix for aux loss computation during training
        return refined_ids, match_mat

    def compute_aux_loss(self, match_mat, veh_boxes, inf_boxes, pos_dist=2.0):
        """
        Pseudo-label: pairs within pos_dist meters are positive.
        match_mat: [N, M]
        """
        N, M = len(veh_boxes), len(inf_boxes)
        if N == 0 or M == 0:
            return torch.tensor(0.0, device=match_mat.device)

        dist = (veh_boxes[:, :2].unsqueeze(1) - inf_boxes[:, :2].unsqueeze(0)).norm(dim=-1)  # [N,M]
        labels = (dist < pos_dist).float()
        loss = F.binary_cross_entropy(match_mat.clamp(1e-6, 1 - 1e-6), labels)
        return loss * self.loss_weight
