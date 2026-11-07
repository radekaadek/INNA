"""Ball network (ResNet-18 heatmap), ball tracker, ball+player crop policy and metrics for SN-Tracking-2023."""
import io
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

import sm_lib as L
from sn_build import SN, W, H

FRAC = L.FRAC
MARGIN = 0.03  # keep the ball at least this far (frame-width fraction) from the window edge when possible


def load(split):
    d = np.load(SN / f"{split}.npz")
    return {k: d[k] for k in d.files}


def load_small(split, size=(448, 252)):
    """Players module input: whole split decoded at 448x252 (cached in an npz, ~3.6 GB for train)."""
    p = SN / f"small_{split}.npz"
    d = load(split)
    if p.exists():
        d["images"] = np.load(p)["images"]
    else:
        d["images"] = np.stack([np.asarray(Image.open(SN / "frames" / split / n).convert("RGB").resize(size, Image.BILINEAR)) for n in d["names"]])
        np.savez(p, images=d["images"])
    d["prev"] = d["images"]  # unused (single-frame models only)
    return d


# ---- ball network -----------------------------------------------------------------------------------------------------
class BallNet(nn.Module):
    """ResNet-18 encoder + light FPN decoder -> stride-4 ball heatmap logits (B,1,H/4,W/4)."""

    def __init__(self, pretrained=True, in_ch=3):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
        if in_ch != 3:  # extra input channels (e.g. a prior map): new conv1 weights start at zero
            w = r.conv1.weight.data; r.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
            r.conv1.weight.data.zero_(); r.conv1.weight.data[:, :3] = w
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in (64, 128, 256, 512)])
        self.out = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 1, 1))
        nn.init.constant_(self.out[-1].bias, -4.6)  # prior p=0.01

    def forward(self, x):
        c1 = self.l1(self.stem(x)); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        p = self.lat[3](c4)
        for lat, c in ((self.lat[2], c3), (self.lat[1], c2), (self.lat[0], c1)):
            p = F.interpolate(p, size=c.shape[-2:], mode="nearest") + lat(c)
        return self.out(p)


def focal_loss(logit, gt):
    """CenterNet penalty-reduced focal loss on a Gaussian heatmap."""
    p = logit.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float(); neg = 1 - pos
    pl = -(1 - p) ** 2 * p.log() * pos
    nl = -((1 - gt) ** 4) * p ** 2 * (1 - p).log() * neg
    n = pos.sum().clamp(min=1)
    return (pl.sum() + nl.sum()) / n


def gaussian_map(h, w, cx, cy, sigma=1.5):
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    if 0 <= int(cx) < w and 0 <= int(cy) < h:
        g[int(cy), int(cx)] = 1.0
    return g


class BallCrops(torch.utils.data.Dataset):
    """Random 512x288 crops of the 896x504 frames (ball inside with prob p_pos), flip + colour jitter."""

    def __init__(self, d, split, n, p_pos=0.75, cw=512, ch=288, seed=0):
        self.d, self.split, self.n, self.p_pos, self.cw, self.ch = d, split, n, p_pos, cw, ch
        self.rng = random.Random(seed)
        self.pos = np.flatnonzero(d["ball"][:, 0] >= 0); self.neg = np.flatnonzero(d["ball"][:, 0] < 0)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(self.rng.random() + i)
        use_ball = len(self.pos) and (rng.random() < self.p_pos or not len(self.neg))
        k = int(rng.choice(self.pos if use_ball else np.arange(len(self.d["names"]))))
        im = np.asarray(Image.open(SN / "frames" / self.split / self.d["names"][k]).convert("RGB"))
        b = self.d["ball"][k]
        x0, y0 = rng.randint(0, W - self.cw), rng.randint(0, H - self.ch)
        if b[0] >= 0 and use_ball:
            bx, by = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H
            x0 = int(min(max(bx - rng.uniform(16, self.cw - 16), 0), W - self.cw)); y0 = int(min(max(by - rng.uniform(16, self.ch - 16), 0), H - self.ch))
        x0, y0 = x0 // 4 * 4, y0 // 4 * 4
        crop = im[y0:y0 + self.ch, x0:x0 + self.cw]
        hm = torch.zeros(self.ch // 4, self.cw // 4)
        if b[0] >= 0:
            bx, by = (b[0] + b[2]) / 2 * W - x0, (b[1] + b[3]) / 2 * H - y0
            if 0 <= bx < self.cw and 0 <= by < self.ch:
                hm = gaussian_map(self.ch // 4, self.cw // 4, bx / 4, by / 4)
        x = torch.from_numpy(crop.copy()).permute(2, 0, 1).float() / 255
        if rng.random() < 0.5:
            x, hm = x.flip(2), hm.flip(1)
        x = (x * rng.uniform(0.75, 1.25) + rng.uniform(-0.1, 0.1)).clamp(0, 1)
        return (x - L.MEAN[0]) / L.STD[0], hm[None]


@torch.no_grad()
def ball_candidates(model, names, split, dev="cuda", bs=8, topk=5):
    """Full-frame inference -> per frame up to `topk` (cx, cy, score) in normalised coordinates."""
    model.eval(); out = []
    for i in range(0, len(names), bs):
        x = torch.stack([torch.from_numpy(np.asarray(Image.open(SN / "frames" / split / n).convert("RGB"))).permute(2, 0, 1) for n in names[i:i + bs]])
        x = ((x.to(dev).float() / 255) - L.MEAN.to(dev)) / L.STD.to(dev)
        hm = model(x).sigmoid()
        pk = (hm == F.max_pool2d(hm, 3, 1, 1)).float() * hm
        B, _, h, w = pk.shape
        sc, ix = pk.view(B, -1).topk(topk, dim=1)
        for b in range(B):
            out.append([((int(j) % w + 0.5) / w, (int(j) // w + 0.5) / h, float(s)) for s, j in zip(sc[b], ix[b])])
    return out


def filter_static(cands, seq, thr=0.3, cell=1 / 48, frac=0.4):
    """Drop candidates that sit in the same image cell (3x3 neighbourhood) in more than `frac` of a clip's samples: broadcast logos
    and score bugs. Offline (uses whole-clip statistics); a stationary real ball (set piece) stays well below `frac` in 30 s clips."""
    out = list(cands)
    for s_ in np.unique(seq):
        idx = np.flatnonzero(seq == s_); cnt = {}
        for i in idx:
            seen = {(int(cx // cell), int(cy // (cell * 1.78))) for cx, cy, sc in cands[i] if sc >= thr}
            for c in seen:
                cnt[c] = cnt.get(c, 0) + 1
        mask = {c for c, n in cnt.items() if n > frac * len(idx)}
        for i in idx:
            out[i] = [(cx, cy, sc) for cx, cy, sc in cands[i]
                      if not any((int(cx // cell) + dx, int(cy // (cell * 1.78)) + dy) in mask for dx in (-1, 0, 1) for dy in (-1, 0, 1))]
    return out


# ---- tracker ----------------------------------------------------------------------------------------------------------
def track_ball(cands, seq, thr=0.3, gate=0.25, hold=8, lam=0.5):
    """Single-ball tracker per clip. Picks, per sample, the candidate maximising score - lam * (distance to last position / gate);
    far jumps to weak candidates are rejected; the last position is held for up to `hold` samples. Returns (N,3): x, y, confidence
    (confidence 0 = no estimate)."""
    out = np.zeros((len(cands), 3))
    for s in np.unique(seq):
        pos, age = None, 10 ** 9
        for i in np.flatnonzero(seq == s):
            best = None
            for cx, cy, sc in cands[i]:
                if sc < thr:
                    continue
                pen = 0.0
                if pos is not None and age <= hold:
                    d = np.hypot(cx - pos[0], (cy - pos[1]) * 0.5625)
                    if d > gate * (1 + age / 4) and sc < 0.8:
                        continue
                    pen = lam * min(d / gate, 1.0)
                if best is None or sc - pen > best[0]:
                    best = (sc - pen, cx, cy, sc)
            if best is not None:
                pos, age = (best[1], best[2]), 0; out[i] = (best[1], best[2], best[3])
            else:
                age += 1
                if pos is not None and age <= hold:
                    out[i] = (pos[0], pos[1], 0.01)
    return out


# ---- crop policy and metrics ------------------------------------------------------------------------------------------
def policy_centers(player_hist, ball_xy, ball_conf, use_ball=True, margin=MARGIN):
    """Window containing the ball (with margin) that covers the most predicted player mass; players-only when no ball estimate."""
    out = np.zeros(len(player_hist))
    k = max(1, int(round(FRAC * player_hist.shape[1])))
    n = player_hist.shape[1]
    for i, h in enumerate(player_hist):
        cs = np.convolve(h, np.ones(k), mode="valid")  # window sums, window j centred at (j + k/2) / n
        cen = (np.arange(len(cs)) + k / 2) / n
        if use_ball and ball_conf[i] > 0:
            bx = ball_xy[i, 0]
            ok = np.abs(cen - bx) <= FRAC / 2 - margin
            if not ok.any():
                ok = np.abs(cen - bx) <= FRAC / 2
            if ok.any():
                cand = np.flatnonzero(ok)
                best = cand[np.flatnonzero(cs[cand] >= cs[cand].max() - 1e-9)]
                out[i] = cen[best[np.argmin(np.abs(cen[best] - bx))]]
                continue
        out[i] = cen[int(np.argmax(cs))]
    return out


def score_sn(d, centers, post="raw"):
    c = np.array([L.clamp_center(x) for x in centers])
    c = np.array([L.clamp_center(x) for x in L.smooth(c, d["seq"], **L.POST[post] if isinstance(post, str) else post)])
    m = L.score(d, c, "raw")
    vis = d["ball"][:, 0] >= 0
    bx = (d["ball"][:, 0] + d["ball"][:, 2]) / 2
    m["ball_in_crop"] = float((np.abs(bx[vis] - c[vis]) <= FRAC / 2).mean())
    m["ball_visible"] = float(vis.mean())
    return m


def ball_hit_rate(d, est, thr=0.02):
    """Fraction of ball-visible frames where the (tracked) ball estimate is within `thr` of the frame width of the GT centre."""
    vis = d["ball"][:, 0] >= 0
    gx, gy = (d["ball"][:, 0] + d["ball"][:, 2]) / 2, (d["ball"][:, 1] + d["ball"][:, 3]) / 2
    dist = np.hypot(est[:, 0] - gx, (est[:, 1] - gy) * 0.5625)
    hit = (est[:, 2] > 0) & (dist <= thr)
    prec = float(hit[est[:, 2] > 0].sum() / max((est[:, 2] > 0).sum(), 1))
    return {"ball_recall": float(hit[vis].mean()), "ball_precision": prec, "ball_coverage": float((est[:, 2] > 0).mean())}
