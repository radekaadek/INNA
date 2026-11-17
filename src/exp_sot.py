"""#21: SUTrack (single-object tracker) on SN ball clips. RUN WITH THE SEPARATE VENV: /home/ard/prg/sot_ext/venv/bin/python -I? (needs PYTHONPATH=SUTrack_src,
so use -s not -I) -- see run_sot.sh. Pure numpy/torch/cv2; reads data/sn/<split>.npz + frames, and cands_ball_r18_static.pkl (static-filtered top-10 candidates).
Modes: gt_init (optimistic: init from GT box at the first visible-ball sample, no re-init), hybrid (init/re-init from ball_r18 detections, causal).
Usage: exp_sot.py <split> <mode> [tau hi_thr] [max_clips]  -> results/exp/sot_<split>_<mode>_<tau>_<hi>.npy and prints recall/precision (hit = within 2% width)."""
import os, pickle, sys, time
import cv2, numpy as np, torch

EXT = "/home/ard/prg/sot_ext"; ROOT = "/home/ard/prg/INNA"
sys.path.insert(0, f"{EXT}/SUTrack_src")
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self  # CPU run (GPU busy / 4 GB): patch the hard-coded .cuda()
from lib.config.sutrack.config import cfg, update_config_from_file
from lib.test.utils import TrackerParams
from lib.test.tracker.sutrack import SUTRACK

W, H = 896, 504; AR = 0.5625; BOX = 10.0  # px box used when a detector (not GT) initialises the tracker


def make(model="t224"):
    update_config_from_file(f"{EXT}/SUTrack_src/experiments/sutrack/sutrack_{model}.yaml")
    p = TrackerParams(); p.cfg = cfg; p.template_factor, p.template_size = cfg.TEST.TEMPLATE_FACTOR, cfg.TEST.TEMPLATE_SIZE
    p.search_factor, p.search_size = cfg.TEST.SEARCH_FACTOR, cfg.TEST.SEARCH_SIZE
    p.checkpoint = f"{EXT}/weights/SUTRACK_{model}.pth.tar"; p.save_all_boxes = False; p.debug = 0
    cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT = False  # no text branch at test time (text_src None)
    return SUTRACK(p, "default")


def run(split, mode, tau=0.3, hi=0.6, max_clips=None, model="t224"):
    d = np.load(f"{ROOT}/data/sn/{split}.npz"); names, seq, ball = d["names"], d["seq"], d["ball"]
    cands = pickle.load(open(f"{ROOT}/results/exp/cands_ball_r18_static.pkl", "rb"))[split]
    tr = make(model); est = np.zeros((len(names), 3)); sc_log = []; t0 = time.time()
    clips = list(np.unique(seq))[:max_clips]
    for s in clips:
        tracking = False
        for i in np.flatnonzero(seq == s):
            img = cv2.cvtColor(cv2.imread(f"{ROOT}/data/sn/frames/{split}/{names[i]}"), cv2.COLOR_BGR2RGB)
            cs = [c for c in cands[i] if c[2] >= 0.05]
            def init_at(cx, cy, w=BOX, h=BOX):
                tr.initialize(img, {"init_bbox": [cx * W - w / 2, cy * H - h / 2, w, h]})
            if mode == "gt_init":
                if not tracking and ball[i, 0] >= 0:
                    x1, y1, x2, y2 = ball[i] * [W, H, W, H]; tr.initialize(img, {"init_bbox": [x1, y1, max(x2 - x1, 4), max(y2 - y1, 4)]})
                    tracking = True; est[i] = ((ball[i, 0] + ball[i, 2]) / 2, (ball[i, 1] + ball[i, 3]) / 2, 1.0); continue
                if not tracking:
                    continue
                o = tr.track(img); x, y, w, h = o["target_bbox"]; est[i] = ((x + w / 2) / W, (y + h / 2) / H, float(o["best_score"])); sc_log.append(est[i, 2]); continue
            # hybrid
            if not tracking:
                if cs and cs[0][2] >= hi:
                    init_at(cs[0][0], cs[0][1]); tracking = True; est[i] = (cs[0][0], cs[0][1], cs[0][2])
                continue
            o = tr.track(img); x, y, w, h = o["target_bbox"]; px, py, sc = (x + w / 2) / W, (y + h / 2) / H, float(o["best_score"]); sc_log.append(sc)
            support = any(np.hypot(c[0] - px, (c[1] - py) * AR) < 0.03 and c[2] >= 0.1 for c in cs)
            far_top = bool(cs) and cs[0][2] >= hi and np.hypot(cs[0][0] - px, (cs[0][1] - py) * AR) > 0.03
            if (sc < tau and not support) or (far_top and not support):
                if cs and cs[0][2] >= 0.3:
                    init_at(cs[0][0], cs[0][1]); est[i] = (cs[0][0], cs[0][1], cs[0][2])
                else:
                    tracking = False
                continue
            est[i] = (px, py, max(sc, 1e-3))
        print(f"clip {s} done {time.time() - t0:.0f}s", flush=True)
    sel = np.isin(seq, clips); vis = (ball[:, 0] >= 0) & sel
    gx, gy = (ball[:, 0] + ball[:, 2]) / 2, (ball[:, 1] + ball[:, 3]) / 2
    hit = (est[:, 2] > 0) & (np.hypot(est[:, 0] - gx, (est[:, 1] - gy) * AR) <= 0.02)
    r = dict(recall=float(hit[vis].mean()), precision=float(hit[(est[:, 2] > 0) & sel].sum() / max(((est[:, 2] > 0) & sel).sum(), 1)),
             coverage=float((est[sel, 2] > 0).mean()), score_pcts=np.percentile(sc_log, [5, 25, 50, 75, 95]).round(3).tolist() if sc_log else None, n_clips=len(clips))
    np.save(f"{ROOT}/results/exp/sot_{split}_{mode}_{tau}_{hi}.npy", est); print("RESULT", split, mode, tau, hi, r, flush=True); return r


if __name__ == "__main__":
    a = sys.argv; run(a[1], a[2], float(a[3]) if len(a) > 3 else 0.3, float(a[4]) if len(a) > 4 else 0.6, int(a[5]) if len(a) > 5 else None)
