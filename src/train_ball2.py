"""Ball heatmap network with optional prior-map input channel (previous estimate), EMA weights, and a longer schedule.
  --prior : 4th input channel = Gaussian (sigma 20 px at 896 px) at the previous sample's ball estimate. Training: 25% none, 50% GT position of the
            previous sampled frame with jitter, 25% a wrong position (mined false positive of that frame, else random). Inference (`prior_candidates`)
            is causal: the prior is the previous frame's own top-1 peak (score >= 0.3), held for up to 4 samples.
Writes results/<run>/{best.pt, best_ema.pt, train_log.json}."""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sm_lib as L
import sn_lib as S
from sn_build import SN, W, H

RES = Path(__file__).resolve().parent.parent / "results"
SIG = 20.0


def prior_map(h, w, pos, x0=0, y0=0):
    """Gaussian prior channel (h,w) at image-pixel position `pos` (896x504 coordinates) inside a window starting at (x0, y0); zeros if None."""
    if pos is None:
        return torch.zeros(h, w)
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    return torch.exp(-((xs + x0 - pos[0]) ** 2 + (ys + y0 - pos[1]) ** 2) / (2 * SIG ** 2))


class PriorCrops(S.BallCrops):
    def __init__(self, d, split, n, hardneg=None, **kw):
        super().__init__(d, split, n, **kw)
        self.prev = np.array([k - 1 if k > 0 and d["seq"][k] == d["seq"][k - 1] else -1 for k in range(len(d["names"]))])
        self.hn = {}
        if hardneg is not None:
            for i, x, y in zip(hardneg["idx"], hardneg["x"], hardneg["y"]):
                self.hn.setdefault(int(i), []).append((float(x), float(y)))

    def __getitem__(self, i):
        # replicate BallCrops.__getitem__ so that the crop window is known
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
                hm = S.gaussian_map(self.ch // 4, self.cw // 4, bx / 4, by / 4)
        # prior from the previous sampled frame
        pos, r, p = None, rng.random(), self.prev[k]
        if p >= 0 and r >= 0.25:
            pb = self.d["ball"][p]
            if r < 0.75:
                if pb[0] >= 0:
                    sd = 10 if rng.random() < 0.8 else 40
                    pos = ((pb[0] + pb[2]) / 2 * W + rng.gauss(0, sd), (pb[1] + pb[3]) / 2 * H + rng.gauss(0, sd))
            else:
                pos = rng.choice(self.hn[p]) if p in self.hn else (rng.uniform(0, W), rng.uniform(0, H))
        pm = prior_map(self.ch, self.cw, pos, x0, y0)
        x = torch.from_numpy(crop.copy()).permute(2, 0, 1).float() / 255
        x = (x * rng.uniform(0.75, 1.25) + rng.uniform(-0.1, 0.1)).clamp(0, 1)
        x = (x - L.MEAN[0]) / L.STD[0]
        x = torch.cat([x, pm[None]], 0)
        if rng.random() < 0.5:
            x, hm = x.flip(2), hm.flip(1)
        return x, hm[None]


@torch.no_grad()
def prior_candidates(model, d, split, topk=10, bs=8, dev="cuda", use_prior=True):
    """Causal clip-by-clip inference, clips in lockstep. Returns per frame [(cx, cy, score)] (normalised)."""
    model.eval(); out = [None] * len(d["names"]); seqs = [np.flatnonzero(d["seq"] == s) for s in np.unique(d["seq"])]
    state = [None] * len(seqs)  # (x_px, y_px, age)
    for t in range(max(len(q) for q in seqs)):
        act = [j for j, q in enumerate(seqs) if len(q) > t]
        for a in range(0, len(act), bs):
            js = act[a:a + bs]; ks = [seqs[j][t] for j in js]
            x = torch.stack([torch.from_numpy(np.asarray(Image.open(SN / "frames" / split / d["names"][k]).convert("RGB"))).permute(2, 0, 1) for k in ks])
            x = ((x.to(dev).float() / 255) - L.MEAN.to(dev)) / L.STD.to(dev)
            if use_prior:
                pm = torch.stack([prior_map(H, W, (state[j][0], state[j][1]) if state[j] and state[j][2] <= 4 else None) for j in js]).to(dev)
                x = torch.cat([x, pm[:, None]], 1)
            hm = model(x).sigmoid(); pk = (hm == F.max_pool2d(hm, 3, 1, 1)).float() * hm
            B, _, h, w = pk.shape; sc, ix = pk.view(B, -1).topk(topk, dim=1)
            for b, (j, k) in enumerate(zip(js, ks)):
                cs = [((int(q) % w + 0.5) / w, (int(q) // w + 0.5) / h, float(s)) for s, q in zip(sc[b], ix[b])]
                out[k] = cs
                if cs[0][2] >= 0.3:
                    state[j] = (cs[0][0] * W, cs[0][1] * H, 0)
                elif state[j]:
                    state[j] = (state[j][0], state[j][1], state[j][2] + 1)
    return out


def val_hit(model, vd, use_prior, split="valid"):
    c = prior_candidates(model, vd, split, use_prior=use_prior, topk=1)
    return S.ball_hit_rate(vd, np.array([[x[0][0], x[0][1], x[0][2]] for x in c]), 0.02)["ball_recall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--per-epoch", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-min", type=float, default=60)
    ap.add_argument("--prior", action="store_true")
    ap.add_argument("--hardneg", default=str(RES / "sn_hardneg.npz"))
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--valid-split", default="valid")
    a = ap.parse_args()
    out = RES / a.run; out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); torch.backends.cudnn.benchmark = True
    tr, vd = S.load(a.train_split), S.load(a.valid_split)
    model = S.BallNet(in_ch=4 if a.prior else 3).cuda(); ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * (a.per_epoch // a.bs), pct_start=0.1)
    hn = dict(np.load(a.hardneg)) if a.prior else None
    log, best, best_e, t_start, start = [], -1, -1, time.time(), 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt"); model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        log, best, best_e, start = ck["log"], ck["best"], ck["best_e"], len(ck["log"]); print("resumed from epoch", start)
    for ep in range(start, a.epochs):
        t0 = time.time(); model.train(); tot, n = 0.0, 0
        ds = PriorCrops(tr, a.train_split, a.per_epoch, hardneg=hn, seed=ep) if a.prior else S.BallCrops(tr, a.train_split, a.per_epoch, seed=ep)
        for x, hm in torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=6, drop_last=True):
            x, hm = x.cuda(non_blocking=True), hm.cuda(non_blocking=True)
            loss = S.focal_loss(model(x), hm)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10); opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            tot += loss.item(); n += 1
        h, he = val_hit(model, vd, a.prior, a.valid_split), val_hit(ema, vd, a.prior, a.valid_split)
        rec = {"epoch": ep + 1, "train_loss": tot / n, "val_hit": h, "val_hit_ema": he, "sec": time.time() - t0}
        log.append(rec); print(rec, flush=True)
        if h > best:
            best = h; torch.save({"state": model.state_dict(), "prior": a.prior}, out / "best.pt")
        if he > best_e:
            best_e = he; torch.save({"state": ema.state_dict(), "prior": a.prior}, out / "best_ema.pt")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "log": log, "best": best, "best_e": best_e}, out / "last.pt")
        if (time.time() - t_start) / 60 > a.max_min:
            print("time cap"); break
    torch.save({"state": model.state_dict(), "prior": a.prior}, out / "final.pt"); torch.save({"state": ema.state_dict(), "prior": a.prior}, out / "final_ema.pt")
    json.dump({"args": vars(a), "log": log, "best": best, "best_ema": best_e}, open(out / "train_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
