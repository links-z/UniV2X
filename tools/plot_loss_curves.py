"""UniV2X 3-stage training loss — two-line style: faint raw + bold smooth trend."""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

BASE = "/root/autodl-tmp/UniV2X-zq/UniV2X/projects/work_dirs_e2e_univ2x"

def load(path, global_offset=0, iter_cap=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    train = [r for r in rows if r.get("mode") == "train" and "loss" in r]
    if iter_cap:
        train = [r for r in train if r["iter"] <= iter_cap]
    iters  = np.array([r["iter"] for r in train], dtype=float) + global_offset
    losses = np.array([r["loss"] for r in train], dtype=float)
    return iters, losses

def smooth(y, size=10):
    """Rolling/box average — gives the staircase-with-wiggles look."""
    size = min(size, len(y))
    return uniform_filter1d(y.astype(float), size=size)

# ── Load ──────────────────────────────────────────────────────────────────
s1_i, s1_l   = load(f"{BASE}/stage1_log/20260612_101611.log.json",
                     global_offset=0, iter_cap=600)
s2_i, s2_l   = load(f"{BASE}/stage2_log/20260612_192845.log.json",
                     global_offset=600)

s3a_i, s3a_l = load(f"{BASE}/univ2x_A_continue_unfreeze_bboxhead/20260613_214245.log.json",
                     global_offset=910)
s3b_i, s3b_l = load(f"{BASE}/univ2x_A_continue_unfreeze_bboxhead_from300/20260627_105551.log.json",
                     global_offset=1210)
s3c_i, s3c_l = load(f"{BASE}/univ2x_A_cont_from600/20260627_194808.log.json",
                     global_offset=1510)
s3d_i, s3d_l = load(f"{BASE}/univ2x_A_cont_from900/20260628_105723.log.json",
                     global_offset=1810)

# Trim warmup spikes at sub-segment starts
TRIM = 2
s3b_i, s3b_l = s3b_i[TRIM:], s3b_l[TRIM:]
s3c_i, s3c_l = s3c_i[TRIM:], s3c_l[TRIM:]
s3d_i, s3d_l = s3d_i[TRIM:], s3d_l[TRIM:]

s3_i = np.concatenate([s3a_i, s3b_i, s3c_i, s3d_i])
s3_l = np.concatenate([s3a_l, s3b_l, s3c_l, s3d_l])

# ── Colors: light for raw, dark for trend ────────────────────────────────
C_S1_RAW   = "#90CAF9";  C_S1   = "#1976D2"   # slightly darker blue
C_S2_RAW   = "#EF9A9A";  C_S2   = "#E53935"   # slightly darker red
C_S3_RAW   = "#A5D6A7";  C_S3   = "#388E3C"   # slightly darker green
C_BEST     = "#E65100"

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Stage background fills
ax.axvspan(0,    600,  alpha=0.10, color=C_S1_RAW, zorder=0)
ax.axvspan(600,  910,  alpha=0.12, color=C_S2_RAW, zorder=0)
ax.axvspan(910,  2130, alpha=0.10, color=C_S3_RAW, zorder=0)

# Stage 3 sub-segment dividers
for xv in (1210, 1510, 1810):
    ax.axvline(xv, color="#bbb", lw=0.7, linestyle=":", zorder=1)

# Main stage boundaries
for xv in (600, 910):
    ax.axvline(xv, color="#888", lw=1.0, linestyle="--", zorder=1)

# ── Raw data (faint, shows spikes) ───────────────────────────────────────
ax.plot(s1_i, s1_l, color=C_S1_RAW, lw=0.9, alpha=0.55, zorder=2)
ax.plot(s2_i, s2_l, color=C_S2_RAW, lw=0.9, alpha=0.55, zorder=2)
ax.plot(s3_i, s3_l, color=C_S3_RAW, lw=0.9, alpha=0.55, zorder=2)

# ── Smooth trend (bold rolling average — window=10 gives staircase wiggles)
WIN = 10
ax.plot(s1_i, smooth(s1_l, WIN), color=C_S1, lw=2.8, zorder=3,
        label="Stage 1: unfreeze all (lr×0.1)")
ax.plot(s2_i, smooth(s2_l, WIN), color=C_S2, lw=2.8, zorder=3,
        label="Stage 2: freeze bbox head + multiframe")
ax.plot(s3_i, smooth(s3_l, WIN), color=C_S3, lw=2.8, zorder=3,
        label="Stage 3: unfreeze bbox head")

# ── Best checkpoint marker ────────────────────────────────────────────────
BEST = 1710
ax.axvline(BEST, color=C_BEST, lw=1.3, linestyle="--", zorder=4)
ax.annotate("Best ckpt\n(iter 800)",
            xy=(BEST, 116), xytext=(BEST + 55, 123),
            fontsize=7.5, color=C_BEST, ha="left",
            arrowprops=dict(arrowstyle="->", color=C_BEST, lw=0.9))

# ── Stage top labels ──────────────────────────────────────────────────────
YTOP = 167
ax.text(300,  YTOP, "Stage 1\n(600 iter)",  ha="center", va="top",
        fontsize=8.5, color="#444")
ax.text(755,  YTOP, "Stage 2\n(300 iter)",  ha="center", va="top",
        fontsize=8.5, color="#444")
ax.text(1520, YTOP, "Stage 3",              ha="center", va="top",
        fontsize=8.5, color="#444")

# Stage 3 LR sub-labels — placed at bottom of plot, below the loss curves
s3_lr = [
    (910,  1210, "5.0e-5"),
    (1210, 1510, "2.5e-5"),
    (1510, 1810, "1.25e-5"),
    (1810, 2130, "7.4e-6"),
]
for x0, x1, lr_txt in s3_lr:
    ax.text((x0 + x1) / 2, 92, lr_txt,
            ha="center", va="bottom", fontsize=7, color="#555")

# ── Axes ──────────────────────────────────────────────────────────────────
ax.set_xlim(0, 2200)
ax.set_ylim(90, 172)
ax.set_xticks(range(0, 2201, 200))
ax.set_yticks(range(90, 171, 10))
ax.set_xlabel("Training Iteration (cumulative)", fontsize=11, color="#111")
ax.set_ylabel("Total Loss",                      fontsize=11, color="#111")
ax.set_title("3-Stage Training Loss Curve",      fontsize=12, color="#111")
ax.tick_params(axis="both", labelsize=9.5, colors="#111")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.18, color="#bbb")
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.88,
          edgecolor="#ccc", handlelength=2.0)

plt.tight_layout(pad=1.2)
out = "/root/autodl-tmp/UniV2X-zq/UniV2X/loss_curves_3stage_v2.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
