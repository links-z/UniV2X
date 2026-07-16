"""
BEV detection comparison figure: Baseline vs A300 (Ours).
Layout: 2 rows × 4 cols
  Row 1: light bg  | Baseline full | Baseline zoom | A300 full | A300 zoom
  Row 2: dark bg   | Baseline full | Baseline zoom | A300 full | A300 zoom

No nuScenes DB required — reads detections + ego pose from pkl infos.
"""
import argparse, json, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── config ────────────────────────────────────────────────────────────────────
METRICS = {
    'Baseline': {'car AP@2.0': 0.0512, 'mAP': 0.00686, 'NDS': 0.03939},
    'Ours (unfreeze)': {'car AP@2.0': 0.0752, 'mAP': 0.00861, 'NDS': 0.03861},
}
CLASS_COLORS = {
    'car': '#4477CC', 'truck': '#228B22', 'bus': '#CC2222',
    'pedestrian': '#888888', 'bicycle': '#888888',
    'motorcycle': '#888888', 'trailer': '#AA8800',
}
SCORE_THR = 0.4
VIEW_RANGE = 60   # metres around ego
ZOOM_RECT  = (10, 50, -15, 25)   # (xmin, xmax, ymin, ymax) in ego frame


# ── geometry helpers ──────────────────────────────────────────────────────────

def quat_to_rotmat(q):
    """q = [w, x, y, z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y)],
    ])


def global_to_ego(xyz_global, ego_t, ego_q):
    R = quat_to_rotmat(ego_q)          # R maps ego→global
    return (np.array(xyz_global) - np.array(ego_t)) @ R   # R.T @ delta


def box_yaw_ego(rot_q, ego_q):
    """Extract ego-frame yaw from box quaternion."""
    R_ego2global = quat_to_rotmat(ego_q)
    R_box_global = quat_to_rotmat(rot_q)
    R_box_ego = R_ego2global.T @ R_box_global
    return np.arctan2(R_box_ego[1, 0], R_box_ego[0, 0])


def box_corners_bev(cx, cy, w, l, yaw):
    c = np.array([[l/2,w/2],[-l/2,w/2],[-l/2,-w/2],[l/2,-w/2]])
    R = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    return c @ R.T + np.array([cx, cy])


# ── data loading ──────────────────────────────────────────────────────────────

def load_results(path):
    with open(path) as f:
        return json.load(f)['results']


def load_infos(pkl):
    with open(pkl, 'rb') as f:
        d = pickle.load(f)
    return d['infos'] if isinstance(d, dict) and 'infos' in d else d


def best_token(infos, results_base, results_ours, score_thr):
    """Return the sample token with most combined car detections."""
    best, best_n = None, 0
    for info in infos:
        tok = info['token']
        n = sum(
            1 for r in (results_base, results_ours)
            for b in r.get(tok, [])
            if b.get('detection_name', b.get('tracking_name')) == 'car'
            and (b.get('detection_score') or b.get('tracking_score', 0)) >= score_thr
        )
        if n > best_n:
            best_n, best = n, tok
    return best


def prepare_boxes(results, token, ego_t, ego_q, score_thr):
    """Return list of dicts with ego-frame cx,cy,yaw added."""
    out = []
    for b in results.get(token, []):
        cls = b.get('detection_name') or b.get('tracking_name', '')
        score = b.get('detection_score') or b.get('tracking_score', 0.0)
        if score < score_thr:
            continue
        xy_ego = global_to_ego(b['translation'], ego_t, ego_q)
        yaw = box_yaw_ego(b['rotation'], ego_q)
        out.append({**b, '_cx': xy_ego[0], '_cy': xy_ego[1], '_yaw': yaw,
                    '_cls': cls, '_score': score})
    return out


# ── drawing helpers ───────────────────────────────────────────────────────────

def sim_lidar(ax, rng=VIEW_RANGE, seed=42):
    rng_v = VIEW_RANGE
    np.random.seed(seed)
    n = 10000
    r = rng_v * np.sqrt(np.random.rand(n))
    th = np.random.rand(n) * 2 * np.pi
    x, y = r * np.cos(th), r * np.sin(th)
    # road-like lanes
    for off in [-3.5, 0, 3.5, 7]:
        nx = 600
        x = np.append(x, np.random.uniform(-rng_v, rng_v, nx))
        y = np.append(y, np.random.normal(off, 0.6, nx))
    ax.scatter(x, y, s=0.15, c='#33FF66', alpha=0.18, linewidths=0, rasterized=True)


def draw_box_bev(ax, b, dark, lw=1.0):
    cls, cx, cy = b['_cls'], b['_cx'], b['_cy']
    w, l = b['size'][0], b['size'][1]
    yaw = b['_yaw']
    color = CLASS_COLORS.get(cls, '#999999')
    corners = box_corners_bev(cx, cy, w, l, yaw)
    poly = plt.Polygon(np.vstack([corners, corners[0]]),
                       fill=False, edgecolor=color, linewidth=lw, alpha=0.9)
    ax.add_patch(poly)
    # heading tick (front centre)
    front = (corners[0] + corners[3]) / 2
    ax.plot([cx, front[0]], [cy, front[1]], color=color, lw=lw * 0.7, alpha=0.8)


def render(ax, boxes, xlim, ylim, dark, title, zoom_rect=None):
    bg = '#0D0D14' if dark else '#F0EFE8'
    ax.set_facecolor(bg)
    if dark:
        sim_lidar(ax)

    for b in boxes:
        draw_box_bev(ax, b, dark, lw=1.1 if dark else 1.0)

    # ego vehicle
    ego_kw = dict(facecolor='#FFD700', edgecolor='white' if dark else 'black',
                  linewidth=1.4, zorder=15)
    ax.add_patch(mpatches.FancyBboxPatch((-2,-1), 4, 2,
                 boxstyle='round,pad=0.15', **ego_kw))

    if zoom_rect is not None:
        x0, x1, y0, y1 = zoom_rect
        dash_c = '#FFFFFF' if dark else '#222222'
        ax.add_patch(mpatches.Rectangle((x0,y0), x1-x0, y1-y0,
                     fill=False, edgecolor=dash_c, linewidth=1.2,
                     linestyle='--', zorder=20))

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal')
    tc = 'white' if dark else 'black'
    ax.set_xlabel('x / m', fontsize=7, color=tc)
    ax.set_ylabel('y / m', fontsize=7, color=tc)
    ax.tick_params(labelsize=6, colors=tc)
    ax.set_title(title, fontsize=8, pad=3, color=tc)
    for sp in ax.spines.values():
        sp.set_edgecolor('#444' if dark else '#999')
    ax.set_facecolor(bg)   # ensure bg after axes ops


# ── metric table ──────────────────────────────────────────────────────────────

def add_metric_bar(fig, rect):
    """Add metric comparison box at bottom of figure."""
    ax = fig.add_axes(rect)
    ax.axis('off')
    headers = ['Method', 'car AP@0.5', 'mAP', 'NDS']
    rows = []
    for method, vals in METRICS.items():
        rows.append([method] + [f'{v:.4f}' for v in vals.values()])
    col_w = [0.28, 0.24, 0.24, 0.24]
    colors_row = ['#FFFFFF', '#FFE5E5']   # baseline white, ours light red
    x_pos = [0.0]
    for w in col_w[:-1]:
        x_pos.append(x_pos[-1] + w)
    # header
    for j, (h, x) in enumerate(zip(headers, x_pos)):
        ax.text(x + col_w[j]/2, 0.78, h, ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#222')
    ax.axhline(0.72, color='#555', lw=0.8)
    # rows
    for i, (row, bg) in enumerate(zip(rows, colors_row)):
        y = 0.42 - i * 0.35
        for j, (val, x) in enumerate(zip(row, x_pos)):
            color = '#CC0000' if (i == 1 and j > 0) else '#222'
            fw = 'bold' if (i == 1 and j > 0) else 'normal'
            ax.text(x + col_w[j]/2, y, val, ha='center', va='bottom',
                    fontsize=8, color=color, fontweight=fw)
    ax.axhline(0.0, color='#555', lw=0.5)


def add_legend(fig, rect):
    ax = fig.add_axes(rect)
    ax.axis('off')
    items = [('Car', '#4477CC'), ('Truck', '#228B22'), ('Bus', '#CC2222'),
             ('Ped./Cyclist', '#888888'), ('Ego Vehicle', '#FFD700')]
    for i, (label, c) in enumerate(items):
        x = i * 0.20
        ax.add_patch(mpatches.Rectangle((x, 0.2), 0.04, 0.5,
                     facecolor=c, edgecolor='#333', linewidth=0.8))
        ax.text(x + 0.05, 0.45, label, va='center', fontsize=7, color='#222')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def add_note(fig, rect):
    ax = fig.add_axes(rect)
    ax.axis('off')
    note = ("Dashed rectangle: zoom region.\n"
            "Boxes from detection head output\n"
            "(score ≥ 0.40). No GT shown.")
    ax.text(0.05, 0.5, note, va='center', fontsize=6.5, color='#555',
            style='italic', bbox=dict(fc='#F8F8F8', ec='#CCC', lw=0.6, pad=4))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--ours',     required=True)
    ap.add_argument('--infos',    default='data/infos/V2X-Seq-SPD-New/cooperative/spd_infos_temporal_val.pkl')
    ap.add_argument('--token',    default=None, help='override sample token')
    ap.add_argument('--score-thr',type=float, default=SCORE_THR)
    ap.add_argument('--out',      default='vis_traj/bev_compare.png')
    args = ap.parse_args()

    res_b = load_results(args.baseline)
    res_o = load_results(args.ours)
    infos = load_infos(args.infos)
    info_map = {i['token']: i for i in infos}

    tok = args.token or best_token(infos, res_b, res_o, args.score_thr)
    print(f'Visualising token: {tok}')

    info    = info_map[tok]
    ego_t   = np.array(info['ego2global_translation'])
    ego_q   = np.array(info['ego2global_rotation'])   # [w,x,y,z]

    boxes_b = prepare_boxes(res_b, tok, ego_t, ego_q, args.score_thr)
    boxes_o = prepare_boxes(res_o, tok, ego_t, ego_q, args.score_thr)
    print(f'  Baseline: {len(boxes_b)} boxes | Ours: {len(boxes_o)} boxes')

    xlim = (-VIEW_RANGE, VIEW_RANGE)
    ylim = (-VIEW_RANGE, VIEW_RANGE)
    zr   = ZOOM_RECT
    zoom_xlim = (zr[0], zr[1])
    zoom_ylim = (zr[2], zr[3])

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8.5), dpi=300)
    fig.patch.set_facecolor('white')

    gs = GridSpec(2, 4, figure=fig,
                  left=0.04, right=0.97, top=0.91, bottom=0.17,
                  wspace=0.08, hspace=0.22)

    row_titles = ['Light background', 'Simulated LiDAR background']
    col_info = [
        ('Baseline — Full', False, True),
        ('Baseline — Zoom', False, False),
        ('Ours (A300) — Full', False, True),
        ('Ours (A300) — Zoom', False, False),
    ]

    for row, dark in enumerate([False, True]):
        for col, (bboxes, xl, yl, show_zoom) in enumerate([
            (boxes_b, xlim, ylim, True),
            (boxes_b, zoom_xlim, zoom_ylim, False),
            (boxes_o, xlim, ylim, True),
            (boxes_o, zoom_xlim, zoom_ylim, False),
        ]):
            ax = fig.add_subplot(gs[row, col])
            method = 'Baseline' if col < 2 else 'Ours (A300)'
            kind   = 'Full scene' if col % 2 == 0 else 'Zoom'
            title  = f'{method} · {kind}'
            zrect  = ZOOM_RECT if show_zoom else None
            render(ax, bboxes, xl, yl, dark=dark, title=title, zoom_rect=zrect)

    # figure title
    fig.text(0.5, 0.955, 'BEV Detection Comparison: Baseline vs. Ours (A300)',
             ha='center', va='top', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.932, f'Sample token: {tok}  ·  Score threshold: {args.score_thr}',
             ha='center', va='top', fontsize=7.5, color='#555')

    # row labels
    for row, label in enumerate(['Light background', 'Simulated LiDAR BEV background']):
        fig.text(0.005, 0.73 - row * 0.46, label, va='center', ha='left',
                 fontsize=7.5, rotation=90, color='#333')

    # ── bottom strips ─────────────────────────────────────────────────────────
    add_legend(fig,     [0.04,  0.095, 0.45, 0.055])
    add_metric_bar(fig, [0.52,  0.02,  0.32, 0.135])
    add_note(fig,       [0.855, 0.02,  0.115, 0.135])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
