"""BallHR: ResNet-18 + FPN ball heatmap net with a full-resolution stem branch.
  --out-stride 2|4   heatmap resolution (hr_a: 2, worse on test than BallNet; 4 keeps the BallNet shape)
  --temporal         hr branch also sees |gray(t) - gray(t-1)|, |gray(t) - gray(t-2)| (causal; pretrained trunk sees frame t only)
  --motion           extra head: displacement (dx, dy)/32 px (896 scale) from the ball at t to its position at t-1, trained at the GT cell
Usage: ball_hr.py --run NAME [--train-split train3 --valid-split valid3]. Writes results/<run>/{best.pt, final.pt, train_log.json}."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

import sm_lib as L
import sn_lib as S
from sn_build import SN, W, H

RES = Path(__file__).resolve().parent.parent / "results"
MU = 32.0  # motion offset unit (px at 896 width)


def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class BallHR(nn.Module):
    """Logits (B,1+2*motion,H/os,W/os). Input channels: 3 RGB (+2 temporal diffs); the trunk uses the first 3, the hr branch all of them."""

    def __init__(self, pretrained=True, temporal=False, motion=False, out_stride=2, width=64):
        super().__init__()
        self.os = out_stride
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat = nn.ModuleList([nn.Conv2d(c, width, 1) for c in (64, 128, 256, 512)])
        hr = [cbr(5 if temporal else 3, 16), cbr(16, 32, 2)] + ([cbr(32, 32, 2)] if out_stride == 4 else [cbr(32, 32)])
        self.hr = nn.Sequential(*hr)
        self.lat_hr = nn.Conv2d(32, width, 1)
        self.fuse = cbr(width, width)
        self.out = nn.Sequential(cbr(width, width), nn.Conv2d(width, 3 if motion else 1, 1))
        nn.init.constant_(self.out[-1].bias, 0.0)
        with torch.no_grad():
            self.out[-1].bias[0] = -4.6  # heatmap prior p=0.01

    def forward(self, x):
        c1 = self.l1(self.stem(x[:, :3])); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        p = self.lat[3](c4)
        for lat, c in ((self.lat[2], c3), (self.lat[1], c2), (self.lat[0], c1)):
            p = F.interpolate(p, size=c.shape[-2:], mode="nearest") + lat(c)
        p = self.fuse(p)  # stride 4
        h = self.hr(x)
        p = F.interpolate(p, size=h.shape[-2:], mode="bilinear", align_corners=False) + self.lat_hr(h) if self.os == 2 else p + self.lat_hr(h)
        return self.out(p)


def build_model(a, pretrained=True):
    return BallHR(pretrained, a["temporal"], a["motion"], a["out_stride"])


def prev_index(seq, k, n):
    """Index of the sample n steps before k in the same clip (clamped to the clip start)."""
    j = k
    for _ in range(n):
        if j > 0 and seq[j - 1] == seq[k]:
            j -= 1
    return j


def add_diffs(x, p1, p2):
    """x, p1, p2: (..., 3, h, w) in [0,1] (before normalisation) -> normalised RGB + 2 gray-diff channels."""
    g = lambda t: t.mean(-3, keepdim=True)
    return torch.cat([(x - L.MEAN.to(x.device)) / L.STD.to(x.device) if x.dim() == 4 else (x - L.MEAN[0]) / L.STD[0],
                      (g(x) - g(p1)).abs() * 4, (g(x) - g(p2)).abs() * 4], -3)


class HRCrops(S.BallCrops):
    """512x288 crops (optional scale jitter); target at stride `os`; optional temporal diffs and motion target."""

    def __init__(self, d, split, n, a, **kw):
        super().__init__(d, split, n, **kw)
        self.a = a

    def _img(self, k):
        return np.asarray(Image.open(SN / "frames" / self.split / self.d["names"][k]).convert("RGB"))

    def __getitem__(self, i):
        a, os_ = self.a, self.a["out_stride"]
        rng = random.Random(self.rng.random() + i)
        use_ball = len(self.pos) and (rng.random() < self.p_pos or not len(self.neg))
        k = int(rng.choice(self.pos if use_ball else np.arange(len(self.d["names"]))))
        s = rng.uniform(0.8, 1.25) if a["scale_jit"] else 1.0
        cw, ch = int(self.cw / s) // 4 * 4, int(self.ch / s) // 4 * 4
        b = self.d["ball"][k]
        x0, y0 = rng.randint(0, W - cw), rng.randint(0, H - ch)
        if b[0] >= 0 and use_ball:
            bx, by = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H
            x0 = int(min(max(bx - rng.uniform(16, cw - 16), 0), W - cw)); y0 = int(min(max(by - rng.uniform(16, ch - 16), 0), H - ch))
        x0, y0 = x0 // 4 * 4, y0 // 4 * 4
        ks = [k] + ([prev_index(self.d["seq"], k, 1), prev_index(self.d["seq"], k, 2)] if a["temporal"] else [])
        fr = []
        for kk in ks:
            t = torch.from_numpy(self._img(kk)[y0:y0 + ch, x0:x0 + cw].copy()).permute(2, 0, 1).float() / 255
            if (cw, ch) != (self.cw, self.ch):
                t = F.interpolate(t[None], size=(self.ch, self.cw), mode="bilinear", align_corners=False)[0]
            fr.append(t)
        sx, sy = self.cw / cw, self.ch / ch
        hm = torch.zeros(self.ch // os_, self.cw // os_); mot = torch.zeros(3)
        if b[0] >= 0:
            bx, by = ((b[0] + b[2]) / 2 * W - x0) * sx, ((b[1] + b[3]) / 2 * H - y0) * sy
            if 0 <= bx < self.cw and 0 <= by < self.ch:
                hm = S.gaussian_map(self.ch // os_, self.cw // os_, bx / os_, by / os_, sigma=1.5 * 4 / os_)
                pk = self.d["ball"][prev_index(self.d["seq"], k, 1)]
                if a["motion"] and prev_index(self.d["seq"], k, 1) != k and pk[0] >= 0:
                    px, py = ((pk[0] + pk[2]) / 2 * W - x0) * sx, ((pk[1] + pk[3]) / 2 * H - y0) * sy
                    mot = torch.tensor([1.0, (px - bx) / MU, (py - by) / MU])
        flip = rng.random() < 0.5
        g, o = rng.uniform(0.75, 1.25), rng.uniform(-0.1, 0.1)
        fr = [(t * g + o).clamp(0, 1) for t in fr]
        x = add_diffs(fr[0], fr[1], fr[2]) if a["temporal"] else (fr[0] - L.MEAN[0]) / L.STD[0]
        if flip:
            x, hm = x.flip(2), hm.flip(1); mot[1] = -mot[1]
        return x, hm[None], mot


def loss_fn(out, hm, mot):
    loss = S.focal_loss(out[:, :1], hm)
    if out.shape[1] == 3:
        mask = F.max_pool2d(hm.eq(1).float(), 3, 1, 1) * mot[:, :1, None, None]  # cells around the GT peak, samples with a visible previous ball
        l1 = F.smooth_l1_loss(out[:, 1:], mot[:, 1:, None, None].expand_as(out[:, 1:]), reduction="none")
        loss = loss + 2.0 * (l1 * mask).sum() / (2 * mask.sum()).clamp(min=1)
    return loss


@torch.no_grad()
def candidates(model, a, d, split, dev="cuda", bs=4, topk=10, nms=5):
    """Per frame up to `topk` (cx, cy, score) normalised, plus `extra[i][j] = (bx, by)` the predicted previous position (motion head) or None."""
    model.eval(); out, extra = [], []; os_ = a["out_stride"]; names, seq = d["names"], d["seq"]
    load = lambda k: torch.from_numpy(np.asarray(Image.open(SN / "frames" / split / names[k]).convert("RGB")).copy()).permute(2, 0, 1)
    for i in range(0, len(names), bs):
        ids = range(i, min(i + bs, len(names)))
        x = torch.stack([load(k) for k in ids]).to(dev).float() / 255
        if a["temporal"]:
            p1 = torch.stack([load(prev_index(seq, k, 1)) for k in ids]).to(dev).float() / 255
            p2 = torch.stack([load(prev_index(seq, k, 2)) for k in ids]).to(dev).float() / 255
            x = add_diffs(x, p1, p2)
        else:
            x = (x - L.MEAN.to(dev)) / L.STD.to(dev)
        o = model(x); hm = o[:, :1].sigmoid()
        pk = (hm == F.max_pool2d(hm, nms, 1, nms // 2)).float() * hm
        B, _, h, w = pk.shape; sc, ix = pk.view(B, -1).topk(topk, dim=1)
        for b in range(B):
            cs, ex = [], []
            for s_, j in zip(sc[b], ix[b]):
                yy, xx = int(j) // w, int(j) % w; cx, cy = (xx + 0.5) / w, (yy + 0.5) / h
                cs.append((cx, cy, float(s_)))
                ex.append((cx + float(o[b, 1, yy, xx]) * MU / W, cy + float(o[b, 2, yy, xx]) * MU / H) if o.shape[1] == 3 else None)
            out.append(cs); extra.append(ex)
    return out, extra


def val_hit(model, a, vd, split):
    c, _ = candidates(model, a, vd, split, topk=1)
    return S.ball_hit_rate(vd, np.array([[x[0][0], x[0][1], x[0][2]] for x in c]), 0.02)["ball_recall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--per-epoch", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-stride", type=int, default=4, choices=[2, 4])
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--motion", action="store_true")
    ap.add_argument("--scale-jit", action="store_true")
    ap.add_argument("--train-split", default="train3")
    ap.add_argument("--valid-split", default="valid3")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 160 crops")
    a = vars(ap.parse_args())
    if a["smoke"]:
        a["epochs"], a["per_epoch"] = 1, 160
    out = RES / a["run"]; out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a["seed"]); torch.backends.cudnn.benchmark = True
    tr, vd = S.load(a["train_split"]), S.load(a["valid_split"])
    model = build_model(a).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a["lr"], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a["lr"], total_steps=a["epochs"] * (a["per_epoch"] // a["bs"]), pct_start=0.1)
    log, best = [], -1
    for ep in range(a["epochs"]):
        t0 = time.time(); model.train(); tot, n = 0.0, 0
        ds = HRCrops(tr, a["train_split"], a["per_epoch"], a, seed=a["seed"] * 1000 + ep)
        for x, hm, mot in torch.utils.data.DataLoader(ds, batch_size=a["bs"], num_workers=6, drop_last=True):
            x, hm, mot = x.cuda(non_blocking=True), hm.cuda(non_blocking=True), mot.cuda(non_blocking=True)
            loss = loss_fn(model(x), hm, mot)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10); opt.step(); sched.step()
            tot += loss.item(); n += 1
        h = val_hit(model, a, vd, a["valid_split"])
        rec = {"epoch": ep + 1, "train_loss": tot / n, "val_top1_hit": h, "sec": time.time() - t0, "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        log.append(rec); print(rec, flush=True)
        if h > best:
            best = h; torch.save({"state": model.state_dict(), "args": a}, out / "best.pt")
    torch.save({"state": model.state_dict(), "args": a}, out / "final.pt")
    json.dump({"args": a, "log": log, "best": best}, open(out / "train_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
