import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..dense_heads.track_head_plugin import Instances
from .agent_fusion import AgentQueryFusion


class LearnableAgentQueryFusion(AgentQueryFusion):
    """Dustbin Sinkhorn-based differentiable cross-agent query matching and fusion."""

    def __init__(self, pc_range, embed_dims=256, num_sinkhorn_iters=20, temperature=0.1,
                 match_loss_weight=0.05, use_sinkhorn=True, use_gate=True, use_complement=True):
        super().__init__(pc_range=pc_range, embed_dims=embed_dims)

        self.num_iters = num_sinkhorn_iters
        self.match_loss_weight = match_loss_weight
        self.use_sinkhorn = use_sinkhorn
        self.use_gate = use_gate
        self.use_complement = use_complement

        # learnable matching projections
        self.match_proj_q = nn.Linear(embed_dims, embed_dims)
        self.match_proj_k = nn.Linear(embed_dims, embed_dims)

        # geometric compatibility MLP: (dx, dy, dz, dist, inf_score) -> score
        self.geo_mlp = nn.Sequential(
            nn.Linear(5, embed_dims), nn.ReLU(), nn.Linear(embed_dims, 1)
        )

        # fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims), nn.ReLU(), nn.Linear(embed_dims, embed_dims)
        )

        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.dustbin_score = nn.Parameter(torch.tensor(0.0))
        self.new_obj_threshold = nn.Parameter(torch.tensor(0.5))

        # init match projections close to identity so initial behavior resembles L2 matching
        nn.init.eye_(self.match_proj_q.weight)
        nn.init.zeros_(self.match_proj_q.bias)
        nn.init.eye_(self.match_proj_k.weight)
        nn.init.zeros_(self.match_proj_k.bias)

    def _log_sinkhorn(self, log_S):
        # log-domain Sinkhorn: numerically stable, prevents inf/nan
        tau = F.softplus(self.temperature) + 0.03
        log_M = log_S / tau
        for _ in range(self.num_iters):
            log_M = log_M - log_M.logsumexp(dim=1, keepdim=True)
            log_M = log_M - log_M.logsumexp(dim=0, keepdim=True)
        return log_M  # log-domain, call .exp() when needed

    def _dustbin_sinkhorn(self, S):
        """Augment S with dustbin row/col, run Sinkhorn, return (M, M_aug)."""
        N_v, N_i = S.shape
        # dustbin row and col filled with learnable score
        db = self.dustbin_score.expand(1, 1)
        dustbin_row = db.expand(1, N_i)          # [1, N_i]
        dustbin_col = db.expand(N_v + 1, 1)      # [N_v+1, 1]
        S_aug = torch.cat([
            torch.cat([S, dustbin_row], dim=0),
            dustbin_col
        ], dim=1)                                 # [N_v+1, N_i+1]
        log_M_aug = self._log_sinkhorn(S_aug)
        M_aug = log_M_aug.exp()
        return M_aug[:N_v, :N_i], M_aug          # (M, M_aug)

    def _compute_score_matrix(self, veh_query, inf_query, veh_ref_pts, inf_ref_pts,
                               veh_scores, inf_scores):
        # appearance similarity
        q = self.match_proj_q(veh_query[:, self.embed_dims:])
        k = self.match_proj_k(inf_query[:, self.embed_dims:])
        S_app = torch.matmul(q, k.T) / math.sqrt(self.embed_dims)  # [N_v, N_i]

        # geometric compatibility: [dx, dy, dz, dist, inf_score]
        diff = veh_ref_pts[:, None, :3] - inf_ref_pts[None, :, :3]  # [N_v, N_i, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)                # [N_v, N_i, 1]
        if inf_scores is not None:
            inf_s = inf_scores[None, :, None].expand(diff.shape[0], -1, 1)
        else:
            inf_s = torch.zeros(*diff.shape[:2], 1, device=diff.device)
        geo_feat = torch.cat([diff, dist, inf_s], dim=-1)            # [N_v, N_i, 5]
        S_geo = self.geo_mlp(geo_feat).squeeze(-1)                   # [N_v, N_i]

        S = S_app + S_geo
        # both scores have column-discriminating effect
        if veh_scores is not None:
            S = S + torch.log(veh_scores[:, None].clamp(1e-6))
        if inf_scores is not None:
            S = S + torch.log(inf_scores[None, :].clamp(1e-6))
        return S

    def compute_matching_loss(self, M_aug, veh_ref_pts, inf_ref_pts,
                               veh_scores, inf_scores, dist_thr=2.0, score_thr=0.1):
        """Auxiliary matching loss using Hungarian pseudo-labels with quality filtering."""
        N_v, N_i = veh_ref_pts.shape[0], inf_ref_pts.shape[0]
        if N_v == 0 or N_i == 0:
            return M_aug.sum() * 0.0

        # build pseudo GT from center distance (no scipy needed, just greedy on distance)
        with torch.no_grad():
            dist_mat = torch.cdist(veh_ref_pts[:, :3], inf_ref_pts[:, :3])  # [N_v, N_i]
            # quality filter: only supervise confident, close matches
            veh_s = veh_scores if veh_scores is not None else torch.ones(N_v, device=dist_mat.device)
            inf_s = inf_scores if inf_scores is not None else torch.ones(N_i, device=dist_mat.device)
            valid = (dist_mat < dist_thr) & \
                    (veh_s[:, None] > score_thr) & \
                    (inf_s[None, :] > score_thr)

            # build GT matching matrix (dustbin-augmented)
            M_gt = torch.zeros(N_v + 1, N_i + 1, device=dist_mat.device)
            matched_v, matched_i = set(), set()
            # greedy matching on valid pairs sorted by distance
            flat_idx = dist_mat.masked_fill(~valid, 1e6).flatten().argsort()
            for idx in flat_idx:
                vi, ii = idx // N_i, idx % N_i
                vi, ii = vi.item(), ii.item()
                if dist_mat[vi, ii] >= dist_thr:
                    break
                if vi not in matched_v and ii not in matched_i:
                    M_gt[vi, ii] = 1.0
                    matched_v.add(vi)
                    matched_i.add(ii)
            # unmatched → dustbin
            for vi in range(N_v):
                if vi not in matched_v:
                    M_gt[vi, N_i] = 1.0
            for ii in range(N_i):
                if ii not in matched_i:
                    M_gt[N_v, ii] = 1.0

        loss = F.binary_cross_entropy(M_aug.clamp(1e-6, 1 - 1e-6), M_gt)
        return loss

    def forward(self, inf, veh, ego2other_rt, other_agent_pc_range, threshold=0.3,
                gamma=1.0, complement_thr=0.3):
        inf_mask = torch.where(inf.obj_idxes >= 0)
        inf = inf[inf_mask]
        if len(inf) == 0:
            return veh
        inf_mask_new = torch.where(inf.obj_idxes >= 0)

        inf.obj_idxes = torch.ones_like(inf.obj_idxes) * -1

        # coordinate transform
        inf_ref_pts = self._loc_denorm(inf.ref_pts, other_agent_pc_range)
        veh_ref_pts = self._loc_denorm(veh.ref_pts, self.pc_range)

        calib_inf2veh = np.linalg.inv(ego2other_rt[0].cpu().numpy().T)
        calib_inf2veh = inf_ref_pts.new_tensor(calib_inf2veh)
        inf_ref_pts = torch.cat((inf_ref_pts, torch.ones_like(inf_ref_pts[..., :1])), -1).unsqueeze(-1)
        inf_ref_pts = torch.matmul(calib_inf2veh, inf_ref_pts).squeeze(-1)[..., :3]

        # remove ego vehicle from inf queries
        H_B, H_F, W_L, W_R = -2.04, 2.04, -0.92, 0.92
        inf_mask_new = list(inf_mask_new)
        for ii in range(len(inf_ref_pts)):
            xx, yy = inf_ref_pts[ii][0], inf_ref_pts[ii][1]
            if H_B <= xx <= H_F and W_L <= yy <= W_R:
                arr = inf_mask_new[0]
                inf_mask_new[0] = torch.cat([arr[:ii], arr[ii+1:]])
                break
        inf_mask_new = tuple(inf_mask_new)
        inf = inf[inf_mask_new]
        inf_ref_pts = inf_ref_pts[inf_mask_new]

        if len(inf) == 0:
            return veh

        # cross-agent feature alignment
        inf_ref_pts_norm = self._loc_norm(inf_ref_pts.clone(), self.pc_range)
        veh_ref_pts_norm = self._loc_norm(veh_ref_pts.clone(), self.pc_range)
        inf.ref_pts = inf_ref_pts_norm
        veh.ref_pts = veh_ref_pts_norm

        inf2veh_r = calib_inf2veh[:3, :3].reshape(1, 9).repeat(inf.query.shape[0], 1)
        inf.query[..., :self.embed_dims] = self.cross_agent_align_pos(
            torch.cat([inf.query[..., :self.embed_dims], inf2veh_r], -1))
        inf.query[..., self.embed_dims:] = self.cross_agent_align(
            torch.cat([inf.query[..., self.embed_dims:], inf2veh_r], -1))

        veh_scores = veh.scores if hasattr(veh, 'scores') else None
        inf_scores = inf.scores if hasattr(inf, 'scores') else None

        # dustbin Sinkhorn soft matching
        S = self._compute_score_matrix(
            veh.query, inf.query, veh_ref_pts, inf_ref_pts, veh_scores, inf_scores)

        if self.use_sinkhorn:
            M, M_aug = self._dustbin_sinkhorn(S)  # M: [N_v, N_i], M_aug: [N_v+1, N_i+1]
        else:
            # w/o Sinkhorn: simple row-normalized softmax matching
            M = F.softmax(S, dim=1)
            M_aug = M  # no dustbin augmentation

        # reliability-gated fusion: gate by max match confidence per vehicle query
        inf_feat = inf.query[:, self.embed_dims:]                    # [N_i, C]
        inf_fused = torch.matmul(M, inf_feat)                        # [N_v, C]
        veh_feat = veh.query[:, self.embed_dims:].clone()            # [N_v, C]

        if self.use_gate:
            match_conf = M.max(dim=1).values.unsqueeze(-1)           # [N_v, 1]
        else:
            # w/o Gate: no confidence-aware gating
            match_conf = torch.ones(M.shape[0], 1, device=M.device)

        delta = self.fusion_mlp(torch.cat([veh_feat, inf_fused], dim=-1))
        new_query = veh.query.clone()
        new_query[:, self.embed_dims:] = veh_feat + (match_conf ** gamma) * delta
        veh.query = new_query

        # auxiliary matching loss (only during training)
        match_loss = None
        if self.training and self.use_sinkhorn:
            match_loss = self.compute_matching_loss(
                M_aug, veh_ref_pts, inf_ref_pts, veh_scores, inf_scores)

        # complement: inf queries not claimed by any vehicle query
        if self.use_complement:
            inf_accept_idx = [i for i in range(len(inf)) if M[:, i].max().item() > complement_thr]
            veh = self._query_complementation(inf, veh, inf_accept_idx)

        if match_loss is not None:
            # store on veh for the caller to pick up if needed
            veh._match_loss = match_loss * self.match_loss_weight

        return veh
