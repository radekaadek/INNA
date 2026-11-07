"""StreamBall: StreamYOLO (arXiv 2203.12338) adapted to single-class ball detection.
  * shared ResNet-18 + FPN on frames t-1 and t (levels: stride 4 and 8; the ball is ~4-12 px wide at 896 px, YOLOX's stride 8/16/32 is too coarse)
  * DFP (Dual-Flow Perception): per level a shared 1x1 conv-BN-SiLU reduces F_{t-1} and F_t to C/2, they are concatenated (dynamic flow) and
    added to the current feature F_t (static flow / residual)   [--no-dfp: single-frame control, same head]
  * YOLOX-style decoupled anchor-free head (obj, cls, reg), shared over the two levels
  * TAL (Trend-Aware Loss): regression loss per positive weighted by 1/mIoU(target box, box of the reference frame) if mIoU >= tau else 1/nu,
    renormalised so the summed regression loss is unchanged (tau 0.3, nu 1.4 as in the paper)
  * targets: --target cur  ball at t (inputs t-1, t)      --target next  ball at t+1 (inputs t-1, t; the paper's forecasting setup)
Deviations from the paper: ResNet-18 instead of CSPDarknet-53, center-prior assignment (3x3 cells around the GT centre) instead of SimOTA,
focal BCE for objectness, shared (not per-level) head, shared flip/colour jitter for both frames (they stay aligned).
Writes results/<run>/{best.pt, final.pt, train_log.json}."""
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
STRIDES = (4, 8)
SMALL = 12.0  # balls with max(w, h) <= SMALL px (896 scale) are assigned to the stride-4 level
TAU, NU = 0.3, 1.4


def cbs(i, o, k=1):
    return nn.Sequential(nn.Conv2d(i, o, k, 1, k // 2, bias=False), nn.BatchNorm2d(o), nn.SiLU(True))


class StreamBall(nn.Module):
    def __init__(self, pretrained=True, dfp=True, width=64):
        super().__init__()
        self.dfp, self.w = dfp, width
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat = nn.ModuleList([nn.Conv2d(c, width, 1) for c in (64, 128, 256, 512)])
        self.smooth = nn.ModuleList([cbs(width, width, 3) for _ in STRIDES])  # P2 (stride 4), P3 (stride 8)
        if dfp:
            self.reduce = nn.ModuleList([cbs(width, width // 2) for _ in STRIDES])  # shared by both frames
        # decoupled head
        self.stem_h = cbs(width, width)
        self.cls_br = nn.Sequential(cbs(width, width, 3), cbs(width, width, 3))
        self.reg_br = nn.Sequential(cbs(width, width, 3), cbs(width, width, 3))
        self.cls_p, self.obj_p, self.reg_p = nn.Conv2d(width, 1, 1), nn.Conv2d(width, 1, 1), nn.Conv2d(width, 4, 1)
        for p in (self.cls_p, self.obj_p):
            nn.init.constant_(p.bias, -4.6)

    def features(self, x):
        """FPN features [P2, P3] of a batch of frames."""
        c1 = self.l1(self.stem(x)); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        p = self.lat[3](c4); outs = []
        for lat, c in ((self.lat[2], c3), (self.lat[1], c2), (self.lat[0], c1)):
            p = F.interpolate(p, size=c.shape[-2:], mode="nearest") + lat(c)
            outs.append(p)
        return [self.smooth[0](outs[2]), self.smooth[1](outs[1])]

    def fuse(self, f_cur, f_prev):
        if not self.dfp:
            return f_cur
        return [c + torch.cat([self.reduce[i](c), self.reduce[i](p)], 1) for i, (c, p) in enumerate(zip(f_cur, f_prev))]

    def head(self, f):
        h = self.stem_h(f)
        c, r = self.cls_br(h), self.reg_br(h)
        return self.cls_p(c), self.obj_p(r), self.reg_p(r)

    def forward(self, x_prev, x_cur):
        """x_prev, x_cur: (B,3,h,w) normalised. Returns per level (cls, obj, reg) maps."""
        if self.dfp:
            B = x_cur.shape[0]
            f = self.features(torch.cat([x_cur, x_prev]))
            fc, fp = [t[:B] for t in f], [t[B:] for t in f]
        else:
            fc, fp = self.features(x_cur), None
        return [self.head(t) for t in self.fuse(fc, fp)]


def decode_boxes(reg, stride):
    """reg (B,4,h,w) -> boxes (B,h*w,4) x1,y1,x2,y2 in px (YOLOX parametrisation: centre = (grid + d) * s, size = exp(.) * s)."""
    B, _, h, w = reg.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=reg.device), torch.arange(w, device=reg.device), indexing="ij")
    g = torch.stack([xs, ys], -1).float().view(1, h * w, 2)
    r = reg.flatten(2).transpose(1, 2)
    c = (r[..., :2] + g) * stride; s = r[..., 2:].clamp(max=6).exp() * stride
    return torch.cat([c - s / 2, c + s / 2], -1)


def iou_xyxy(a, b):
    lt, rb = torch.max(a[..., :2], b[..., :2]), torch.min(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    ua = (a[..., 2:] - a[..., :2]).prod(-1) + (b[..., 2:] - b[..., :2]).prod(-1) - inter
    return inter / ua.clamp(min=1e-6)


def tal_weights(iou_ref, ref_ok, tau=TAU, nu=NU):
    """Per-sample trend factor: 1/mIoU if the ball was matched in the reference frame with mIoU >= tau else 1/nu (new / fast object)."""
    w = torch.full_like(iou_ref, 1.0 / nu)
    m = ref_ok & (iou_ref >= tau)
    w[m] = 1.0 / iou_ref[m].clamp(min=tau)
    return w


def stream_loss(outs, gt, ref, use_tal=True):
    """outs: model output; gt, ref: (B,5) = [valid, x1,y1,x2,y2] px in the crop (target box and reference-frame box).
    Returns (loss, parts). Positives: 3x3 cells around the GT centre at the level chosen by box size."""
    B = gt.shape[0]; dev = gt.device
    obj_t = [torch.zeros_like(o[1]) for o in outs]
    pos_pred, pos_gt, pos_w, pos_b = [], [], [], []
    iou_ref = iou_xyxy(gt[:, 1:], ref[:, 1:]); tal = tal_weights(iou_ref, ref[:, 0] > 0) if use_tal else torch.ones(B, device=dev)
    cls_logit = []
    for b in range(B):
        if gt[b, 0] <= 0:
            continue
        box = gt[b, 1:]; cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        li = 0 if max(box[2] - box[0], box[3] - box[1]) <= SMALL else 1
        s = STRIDES[li]; _, _, h, w = outs[li][1].shape
        gx, gy = int(cx // s), int(cy // s)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x, y = gx + dx, gy + dy
                if 0 <= x < w and 0 <= y < h:
                    obj_t[li][b, 0, y, x] = 1.0
                    pos_b.append((b, li, y * w + x))
    n_pos = max(len(pos_b), 1)
    l_obj = sum(torchvision.ops.sigmoid_focal_loss(o[1], t, alpha=0.25, gamma=2.0, reduction="sum") for o, t in zip(outs, obj_t)) / n_pos
    l_cls = l_reg_sum = torch.zeros((), device=dev)
    if pos_b:
        dec = [decode_boxes(o[2], s) for o, s in zip(outs, STRIDES)]
        bi = torch.tensor([p[0] for p in pos_b], device=dev); li = torch.tensor([p[1] for p in pos_b], device=dev); ci = torch.tensor([p[2] for p in pos_b], device=dev)
        pb = torch.zeros(len(pos_b), 4, device=dev); cl = torch.zeros(len(pos_b), device=dev)
        for l in range(len(outs)):
            m = li == l
            if m.any():
                pb[m] = dec[l][bi[m], ci[m]]; cl[m] = outs[l][0].flatten(2)[bi[m], 0, ci[m]]
        tb = gt[bi, 1:]
        l_i = 1 - iou_xyxy(pb, tb) ** 2  # YOLOX IoU loss
        w = tal[bi]
        w = w * l_i.detach().sum() / (w * l_i.detach()).sum().clamp(min=1e-6)  # keep total regression loss unchanged
        l_reg_sum = (w * l_i).sum() / n_pos
        l_cls = F.binary_cross_entropy_with_logits(cl, torch.ones_like(cl), reduction="sum") / n_pos
    loss = 5.0 * l_reg_sum + l_obj + l_cls  # YOLOX regression weight 5
    return loss, {"obj": float(l_obj), "cls": float(l_cls), "reg": float(l_reg_sum)}


# ---- data ----------------------------------------------------------------------------------------------------------------
def step_index(seq, k, n):
    """Index of the sample n steps before k in the same clip, clamped to the clip start (n may be negative: forward, clamped to the clip end)."""
    j = k
    for _ in range(abs(n)):
        nj = j - 1 if n > 0 else j + 1
        if 0 <= nj < len(seq) and seq[nj] == seq[k]:
            j = nj
    return j


def ball_px(d, k):
    b = d["ball"][k]
    if b[0] < 0:
        return np.zeros(5, np.float32)
    return np.array([1, b[0] * W, b[1] * H, b[2] * W, b[3] * H], np.float32)


class StreamCrops(torch.utils.data.Dataset):
    """512x288 crop windows applied identically to frames (k-1, k); target ball at k (cur) or k+1 (next); reference box = box in the frame before the target."""

    def __init__(self, d, split, n, target="cur", cw=512, ch=288, p_pos=0.75, seed=0):
        self.d, self.split, self.n, self.target, self.cw, self.ch, self.p_pos = d, split, n, target, cw, ch, p_pos
        self.rng = random.Random(seed); seq = d["seq"]; N = len(seq)
        self.tidx = np.array([step_index(seq, k, -1) if target == "next" else k for k in range(N)])  # target sample for input index k
        ok = (self.tidx != np.arange(N)) if target == "next" else np.ones(N, bool)
        self.valid = np.flatnonzero(ok)
        vis = d["ball"][self.tidx, 0] >= 0
        self.pos = np.flatnonzero(ok & vis)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(self.rng.random() + i)
        use_ball = len(self.pos) and rng.random() < self.p_pos
        k = int(rng.choice(self.pos if use_ball else self.valid))
        t = int(self.tidx[k]); kp = step_index(self.d["seq"], k, 1)
        gt = ball_px(self.d, t); ref = ball_px(self.d, k if self.target == "next" else kp)
        x0, y0 = rng.randint(0, W - self.cw), rng.randint(0, H - self.ch)
        if gt[0] > 0 and use_ball:
            bx, by = (gt[1] + gt[3]) / 2, (gt[2] + gt[4]) / 2
            x0 = int(min(max(bx - rng.uniform(16, self.cw - 16), 0), W - self.cw)); y0 = int(min(max(by - rng.uniform(16, self.ch - 16), 0), H - self.ch))
        x0, y0 = x0 // 4 * 4, y0 // 4 * 4
        g, o = rng.uniform(0.75, 1.25), rng.uniform(-0.1, 0.1); flip = rng.random() < 0.5
        fr = []
        for kk in (kp, k):
            im = np.asarray(Image.open(SN / "frames" / self.split / self.d["names"][kk]).convert("RGB"))[y0:y0 + self.ch, x0:x0 + self.cw]
            x = torch.from_numpy(im.copy()).permute(2, 0, 1).float() / 255
            x = ((x * g + o).clamp(0, 1) - L.MEAN[0]) / L.STD[0]
            fr.append(x.flip(2) if flip else x)

        def shift(b):  # to crop coordinates (+ flip); drop boxes whose centre is outside the crop
            b = b.copy()
            if b[0] > 0:
                b[[1, 3]] -= x0; b[[2, 4]] -= y0
                cx, cy = (b[1] + b[3]) / 2, (b[2] + b[4]) / 2
                if not (0 <= cx < self.cw and 0 <= cy < self.ch):
                    return np.zeros(5, np.float32)
                if flip:
                    b[[1, 3]] = self.cw - b[[3, 1]]
            return b
        return fr[0], fr[1], torch.from_numpy(shift(gt)), torch.from_numpy(shift(ref))


# ---- inference -----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def decode_candidates(outs, topk=10, nms_px=8.0):
    """Per image: up to `topk` (cx, cy, score) in px of the input; score = sigmoid(obj) * sigmoid(cls); distance NMS across both levels."""
    B = outs[0][0].shape[0]; allc = [[] for _ in range(B)]
    for (cls, obj, reg), s in zip(outs, STRIDES):
        sc = obj.sigmoid() * cls.sigmoid()
        pk = (sc == F.max_pool2d(sc, 3, 1, 1)).float() * sc
        _, _, h, w = pk.shape
        v, ix = pk.view(B, -1).topk(topk, dim=1); bx = decode_boxes(reg, s)
        for b in range(B):
            for vv, j in zip(v[b], ix[b]):
                x1, y1, x2, y2 = bx[b, int(j)].tolist()
                allc[b].append(((x1 + x2) / 2, (y1 + y2) / 2, float(vv)))
    out = []
    for cs in allc:
        keep = []
        for c in sorted(cs, key=lambda c: -c[2]):
            if all(np.hypot(c[0] - k[0], c[1] - k[1]) > nms_px for k in keep):
                keep.append(c)
        out.append(keep[:topk])
    return out


@torch.no_grad()
def candidates(model, a, d, split, dev="cuda", bs=4, topk=10, W_=W, H_=H):
    """Candidates for every sample of `d`, indexed by TARGET sample. cur: inputs (k-1, k). next: inputs (k-2, k-1), none for the first sample of a clip."""
    model.eval(); seq, names = d["seq"], d["names"]; out = [[] for _ in names]
    load = lambda k: torch.from_numpy(np.asarray(Image.open(SN / "frames" / split / names[k]).convert("RGB")).copy()).permute(2, 0, 1)
    prep = lambda x: ((x.to(dev).float() / 255) - L.MEAN.to(dev)) / L.STD.to(dev)
    for i in range(0, len(names), bs):
        ids = []
        for k in range(i, min(i + bs, len(names))):
            inp = step_index(seq, k, 1) if a["target"] == "next" else k
            if a["target"] == "next" and inp == k:
                continue  # first sample of a clip: nothing to predict from
            ids.append((k, step_index(seq, inp, 1), inp))
        if not ids:
            continue
        xp = prep(torch.stack([load(p) for _, p, _ in ids])); xc = prep(torch.stack([load(c) for _, _, c in ids]))
        for (k, _, _), cs in zip(ids, decode_candidates(model(xp, xc), topk)):
            out[k] = [(cx / W_, cy / H_, sc) for cx, cy, sc in cs]
    return out


def val_hit(model, a, vd, split, dev="cuda"):
    c = candidates(model, a, vd, split, dev=dev, topk=1)
    est = np.array([[x[0][0], x[0][1], x[0][2]] if x else [0, 0, 0] for x in c])
    return S.ball_hit_rate(vd, est, 0.02)["ball_recall"]


def build_model(a, pretrained=True):
    return StreamBall(pretrained, dfp=not a["no_dfp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--target", default="cur", choices=["cur", "next"])
    ap.add_argument("--no-dfp", action="store_true", help="single-frame control (frame t only)")
    ap.add_argument("--no-tal", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--per-epoch", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-split", default="train3")
    ap.add_argument("--valid-split", default="valid3")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 160 crops")
    ap.add_argument("--device", default="cuda")
    a = vars(ap.parse_args())
    if a["smoke"]:
        a["epochs"], a["per_epoch"] = 1, 160
    dev = a["device"]; out = RES / a["run"]; out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a["seed"]); torch.backends.cudnn.benchmark = dev == "cuda"
    tr, vd = S.load(a["train_split"]), S.load(a["valid_split"])
    if a["smoke"]:
        vd = {k: v[:60] for k, v in vd.items() if hasattr(v, "__len__") and len(v) == len(vd["names"])}
    model = build_model(a).to(dev)
    import copy
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=a["lr"], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a["lr"], total_steps=a["epochs"] * (a["per_epoch"] // a["bs"]), pct_start=0.1)
    log, best = [], -1
    for ep in range(a["epochs"]):
        t0 = time.time(); model.train(); tot, n = 0.0, 0
        ds = StreamCrops(tr, a["train_split"], a["per_epoch"], a["target"], seed=a["seed"] * 1000 + ep)
        for xp, xc, gt, ref in torch.utils.data.DataLoader(ds, batch_size=a["bs"], num_workers=6 if dev == "cuda" else 0, drop_last=True):
            xp, xc, gt, ref = (t.to(dev, non_blocking=True) for t in (xp, xc, gt, ref))
            loss, parts = stream_loss(model(xp, xc), gt, ref, use_tal=not a["no_tal"])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10); opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(a["ema"]).add_(pm.detach(), alpha=1 - a["ema"])
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            tot += loss.item(); n += 1
        h = val_hit(ema, a, vd, a["valid_split"], dev) if not a["smoke"] else float("nan")
        if a["smoke"]:
            c = candidates(ema, a, vd, a["valid_split"], dev=dev, topk=10); print("smoke candidates per frame:", {len(x) for x in c})
        rec = {"epoch": ep + 1, "train_loss": tot / n, "val_top1_hit_ema": h, "sec": time.time() - t0,
               "mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0}
        log.append(rec); print(rec, flush=True)
        if h > best or a["smoke"]:
            best = h; torch.save({"state": ema.state_dict(), "args": a}, out / "best.pt")
    torch.save({"state": ema.state_dict(), "args": a}, out / "final.pt")
    json.dump({"args": a, "log": log, "best": best}, open(out / "train_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
