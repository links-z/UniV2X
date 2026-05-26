"""
Geometry-based Sinkhorn association refinement.
Reads results_nusc.json, re-assigns tracking_id using Sinkhorn soft matching,
writes results_nusc_sinkhorn.json. Detection boxes/scores unchanged.
"""
import json, math, argparse, copy
from collections import defaultdict
import torch
import torch.nn.functional as F

# ---------- Sinkhorn ----------
def log_sinkhorn(log_S, n_iters=20, tau=0.5):
    log_M = log_S / tau
    for _ in range(n_iters):
        log_M = log_M - log_M.logsumexp(dim=1, keepdim=True)
        log_M = log_M - log_M.logsumexp(dim=0, keepdim=True)
    return log_M.exp()

def dustbin_sinkhorn(S, dustbin=-8.0, n_iters=20, tau=0.5):
    N, M = S.shape
    db = torch.full((1,), dustbin)
    S_aug = torch.cat([
        torch.cat([S, db.expand(1, M)], dim=0),
        torch.cat([db.expand(N + 1, 1)], dim=1)
    ], dim=1)  # [N+1, M+1]
    # fix: build augmented properly
    row_db = torch.full((N, 1), dustbin)
    col_db = torch.full((1, M + 1), dustbin)
    S_aug = torch.cat([torch.cat([S, row_db], dim=1), col_db], dim=0)
    M_aug = log_sinkhorn(S_aug, n_iters, tau)
    return M_aug[:N, :M], M_aug

# ---------- cost ----------
CLASS_MAX_DIST = {'car': 4.0, 'truck': 4.0, 'bus': 5.0, 'trailer': 5.0,
                  'construction_vehicle': 3.0, 'pedestrian': 1.5,
                  'motorcycle': 2.5, 'bicycle': 2.5,
                  'traffic_cone': 1.5, 'barrier': 2.0}

DT = 0.5  # seconds between frames (10Hz, gap=5)

def predicted_dist(a, b):
    """Distance from a's velocity-predicted position to b's actual position."""
    px = a['translation'][0] + a['velocity'][0] * DT
    py = a['translation'][1] + a['velocity'][1] * DT
    return math.sqrt((px - b['translation'][0])**2 + (py - b['translation'][1])**2)

def vel_error(a, b):
    va, vb = a['velocity'], b['velocity']
    return math.sqrt((va[0]-vb[0])**2 + (va[1]-vb[1])**2)

def build_cost(prev_tracks, curr_dets, lam_dist=1.0, lam_vel=0.3, lam_score=0.2):
    N, M = len(prev_tracks), len(curr_dets)
    C = torch.zeros(N, M)
    for i, p in enumerate(prev_tracks):
        for j, c in enumerate(curr_dets):
            if p['tracking_name'] != c['tracking_name']:
                C[i, j] = 1e6
                continue
            d = predicted_dist(p, c)
            max_d = CLASS_MAX_DIST.get(c['tracking_name'], 3.0)
            if d > max_d:
                C[i, j] = 1e6
                continue
            v = vel_error(p, c)
            s = c['detection_score']
            C[i, j] = lam_dist * d + lam_vel * v - lam_score * s
    return C

# ---------- main ----------
def refine(input_path, output_path, data_info_path,
           match_thr=0.3, n_iters=20, tau=0.5, dustbin=-8.0):
    with open(input_path) as f:
        data = json.load(f)
    with open(data_info_path) as f:
        data_info = json.load(f)

    # build scene -> ordered sample tokens
    scene_to_tokens = defaultdict(list)
    for entry in data_info:
        scene_to_tokens[entry['vehicle_sequence']].append(entry['vehicle_frame'])
    # sort by frame number within each scene
    for seq in scene_to_tokens:
        scene_to_tokens[seq].sort(key=lambda x: int(x))

    results = data['results']
    new_results = copy.deepcopy(results)

    next_id = [0]  # mutable counter

    def new_track_id():
        tid = next_id[0]
        next_id[0] += 1
        return str(tid)

    for scene_id, tokens in scene_to_tokens.items():
        prev_tracks = {}  # track_id -> last box
        id_map = {}       # old_id -> new_id

        for token in tokens:
            if token not in results:
                prev_tracks = {}
                id_map = {}
                continue

            curr_dets = results[token]
            if not curr_dets:
                new_results[token] = []
                prev_tracks = {}
                id_map = {}
                continue

            if not prev_tracks:
                # first frame: assign new IDs
                new_boxes = []
                for det in curr_dets:
                    nid = new_track_id()
                    box = copy.deepcopy(det)
                    box['tracking_id'] = nid
                    new_boxes.append(box)
                    prev_tracks[nid] = box
                new_results[token] = new_boxes
                continue

            prev_list = list(prev_tracks.values())
            curr_list = list(curr_dets)

            C = build_cost(prev_list, curr_list)
            # convert cost to similarity (negate, clamp inf)
            S = -C.clamp(max=100.0)
            M, _ = dustbin_sinkhorn(S, dustbin=dustbin, n_iters=n_iters, tau=tau)

            match_conf, match_idx = M.max(dim=0)  # [M]
            prev_conf, prev_match = M.max(dim=1)  # [N]

            assigned = {}  # curr_idx -> new_track_id
            for j in range(len(curr_list)):
                i = match_idx[j].item()
                conf = match_conf[j].item()
                d = predicted_dist(prev_list[i], curr_list[j])
                max_d = CLASS_MAX_DIST.get(curr_list[j]['tracking_name'], 3.0)
                if (conf > match_thr and d < max_d and
                        prev_list[i]['tracking_name'] == curr_list[j]['tracking_name']):
                    assigned[j] = list(prev_tracks.keys())[i]
                else:
                    assigned[j] = new_track_id()

            new_boxes = []
            new_prev = {}
            for j, det in enumerate(curr_list):
                box = copy.deepcopy(det)
                box['tracking_id'] = assigned[j]
                new_boxes.append(box)
                new_prev[assigned[j]] = box

            new_results[token] = new_boxes
            prev_tracks = new_prev

    out = {'meta': data['meta'], 'results': new_results}
    with open(output_path, 'w') as f:
        json.dump(out, f)
    print(f'Written to {output_path}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='test/univ2x_coop_e2e/Fri_May_22_19_23_30_2026/results_nusc.json')
    p.add_argument('--output', default='test/sinkhorn_assoc/results_nusc.json')
    p.add_argument('--data-info', default='datasets/V2X-Seq-SPD-New/cooperative/data_info.json')
    p.add_argument('--match-thr', type=float, default=0.3)
    p.add_argument('--tau', type=float, default=0.5)
    p.add_argument('--n-iters', type=int, default=20)
    p.add_argument('--dustbin', type=float, default=-8.0)
    args = p.parse_args()

    import os; os.makedirs(os.path.dirname(args.output), exist_ok=True)
    refine(args.input, args.output, args.data_info,
           match_thr=args.match_thr, n_iters=args.n_iters, tau=args.tau, dustbin=args.dustbin)
