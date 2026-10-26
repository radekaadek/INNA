"""Mine hard negatives for the ball detector: run a trained ball network over the training frames and keep confident peaks that are
NOT the ball (logos, line crossings, socks, ...). Writes results/sn_hardneg.npz with frame index and pixel position (896x504 frames).
Usage: mine_negatives.py [run=ball_r18] [min_score=0.2]"""
import sys

import numpy as np
import torch

import sn_lib as S
from sn_build import H, W
from sm_data import ROOT

run = sys.argv[1] if len(sys.argv) > 1 else "ball_r18"; min_score = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
d = S.load("train")
model = S.BallNet(pretrained=False).cuda(); model.load_state_dict(torch.load(ROOT / "results" / run / "best.pt")["state"])
cands = S.ball_candidates(model, d["names"], "train", topk=5)
idx, xs, ys, sc = [], [], [], []
for k, cs in enumerate(cands):
    b = d["ball"][k]; has = b[0] >= 0
    gx, gy = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) if has else (None, None)
    for cx, cy, s in cs[:3]:
        if s < min_score or (has and np.hypot(cx - gx, (cy - gy) * 0.5625) <= 0.02):
            continue
        idx.append(k); xs.append(cx * W); ys.append(cy * H); sc.append(s)
np.savez(ROOT / "results" / "sn_hardneg.npz", idx=np.array(idx), x=np.array(xs, np.float32), y=np.array(ys, np.float32), score=np.array(sc, np.float32))
print(len(idx), "mined false positives in", len(set(idx)), "of", len(cands), "training frames; mean score", float(np.mean(sc)) if sc else None)
