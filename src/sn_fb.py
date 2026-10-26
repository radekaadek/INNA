"""FootAndBall-style joint ball + player network (re-implementation after Komorowski et al., VISAPP 2020, from the public description):
FPN over a small backbone; ball cell-classification head at stride 4, player head at stride 16; 3x3-cell positive blocks around object
centres, hard-negative mining (3:1) and ball:player loss weights 5:1.  Backbones: 'tiny' (their 5-module VGG-style, from scratch) or 'r18'.
No player box regression: only the player confidence map is used (its column sums give the x-density for the 9:16 window)."""
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.transforms import ColorJitter

import sm_lib as L
from sm_data import BINS
from sn_build import SN, W, H

BALL_DELTA = PLAYER_DELTA = 3
CW, CH = 512, 288


def conv_bn(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class FBNet(nn.Module):
    def __init__(self, backbone="tiny", pretrained=True, in_ch=3):
        super().__init__()
        self.kind, self.in_ch = backbone, in_ch
        if backbone == "tiny":
            cfg = [[16], [32, 32], [32, 32], [64, 64], [64, 64]]; chans = []; mods = []; cin = in_ch
            for c in cfg:
                layers = []
                for o in c:
                    layers.append(conv_bn(cin, o)); cin = o
                layers.append(nn.MaxPool2d(2)); mods.append(nn.Sequential(*layers)); chans.append(cin)
            self.mods, lat = nn.ModuleList(mods), 32
        else:
            r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.mods = nn.ModuleList([nn.Sequential(r.conv1, r.bn1, r.relu), nn.Sequential(r.maxpool, r.layer1), r.layer2, r.layer3, r.layer4])
            if in_ch == 6:  # [frame, frame - previous frame]; the difference channels start at zero
                r.conv1 = L.CropNet._widen(r.conv1)
                self.mods[0] = nn.Sequential(r.conv1, r.bn1, r.relu)
            chans, lat = [64, 64, 128, 256, 512], 64
        self.lat = nn.ModuleList([nn.Conv2d(c, lat, 1) for c in chans])
        head = lambda: nn.Sequential(nn.Conv2d(lat, lat, 3, padding=1), nn.ReLU(True), nn.Conv2d(lat, 2, 3, padding=1))
        self.ball_head, self.player_head = head(), head()

    def backbone_parameters(self):
        return self.mods.parameters()

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("mods.")]

    def forward(self, x):
        feats = []
        for m in self.mods:
            x = m(x); feats.append(x)
        p = self.lat[4](feats[4]); outs = [p]
        for i in (3, 2, 1, 0):
            li = self.lat[i](feats[i])
            p = F.interpolate(p, size=li.shape[-2:], mode="bilinear", align_corners=False) + li
            outs.append(p)
        outs = outs[::-1]  # strides 2, 4, 8, 16, 32
        return self.ball_head(outs[1]), self.player_head(outs[3])


def hard_neg_ce(logits, labels, ratio=3):
    """Cross-entropy over all positive cells and the `ratio`*#pos hardest negatives of each image, divided by the batch #pos."""
    B = logits.shape[0]
    ce = F.cross_entropy(logits.flatten(2), labels.flatten(1), reduction="none")  # (B, N)
    pos = labels.flatten(1) > 0
    npos = pos.sum(1).clamp(min=1)
    neg_ce = ce.masked_fill(pos, -1.0)
    rank = neg_ce.argsort(1, descending=True).argsort(1)
    sel = pos | (rank < (ratio * npos)[:, None])
    return (ce * sel).sum() / pos.sum().clamp(min=1)


def fb_loss(out, ball_lab, pl_lab, wb=5.0, wp=1.0):
    lb, lp = hard_neg_ce(out[0], ball_lab), hard_neg_ce(out[1], pl_lab)
    return (wb * lb + wp * lp) / (wb + wp), lb.item(), lp.item()


def block_labels(points, shape, stride, delta):
    lab = torch.zeros(shape, dtype=torch.long); r = delta // 2
    for x, y in points:
        cx, cy = int(x // stride), int(y // stride)
        if 0 <= cx < shape[1] and 0 <= cy < shape[0]:
            lab[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1] = 1
    return lab


def prev_names(d):
    """Name of the previous sampled frame of the same clip (the frame itself for the first one)."""
    n, seq = d["names"], d["seq"]
    return [n[k - 1] if k > 0 and seq[k] == seq[k - 1] else n[k] for k in range(len(n))]


def read_rgb(split, name):
    return cv2.cvtColor(cv2.imread(str(SN / "frames" / split / name)), cv2.COLOR_BGR2RGB)


def to_input(cur, prev=None):
    """float (3,H,W) in [0,1] -> normalised frame, plus (frame - previous) / std when `prev` is given."""
    x = (cur - L.MEAN[0]) / L.STD[0]
    return x if prev is None else torch.cat([x, (cur - prev) / L.STD[0]], 0)


class FBCrops(torch.utils.data.Dataset):
    """512x288 crops of the 896x504 frames with colour jitter, scale 0.8-1.2, rotation +-5 deg, flip (FootAndBall-style augmentation).
    The ball is inside the crop with prob p_pos; with prob p_hn the crop is centred on a mined false-positive location (hard negative:
    logos, line crossings, ...). two_frame: the previous sampled frame is warped/jittered identically and its difference is appended.
    Targets are cell-label maps (ball: stride 4, players: stride 16)."""

    def __init__(self, d, split, n, p_pos=0.75, seed=0, two_frame=False, hardneg=None, p_hn=0.25):
        self.d, self.split, self.n, self.p_pos, self.seed, self.two_frame = d, split, n, p_pos, seed, two_frame
        self.pos = np.flatnonzero(d["ball"][:, 0] >= 0)
        self.prev = prev_names(d) if two_frame else None
        self.hn = None if hardneg is None else {k: hardneg[k] for k in ("idx", "x", "y")}
        self.p_hn = p_hn if self.hn is not None and len(self.hn["idx"]) else 0.0
        self.cj = ColorJitter(0.4, 0.4, 0.4, 0.1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        hn = rng.random() < self.p_hn
        use_ball = (not hn) and rng.random() < self.p_pos
        if hn:
            e = rng.randrange(len(self.hn["idx"])); k = int(self.hn["idx"][e])
        else:
            k = int(rng.choice(self.pos)) if use_ball else rng.randrange(len(self.d["names"]))
        im = read_rgb(self.split, self.d["names"][k])
        pim = read_rgb(self.split, self.prev[k]) if self.two_frame else None
        b = self.d["ball"][k]; has_ball = b[0] >= 0
        s, th, flip = rng.uniform(0.8, 1.2), np.deg2rad(rng.uniform(-5, 5)), rng.random() < 0.5
        A = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([-1.0 if flip else 1.0, 1.0])
        if hn:
            cs = np.array([self.hn["x"][e] + rng.uniform(-1, 1) * (CW / 2 - 24) / s, self.hn["y"][e] + rng.uniform(-1, 1) * (CH / 2 - 24) / s])
        elif has_ball and use_ball:
            bx, by = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H
            cs = np.array([bx + rng.uniform(-1, 1) * (CW / 2 - 24) / s, by + rng.uniform(-1, 1) * (CH / 2 - 24) / s])
        else:
            cs = np.array([rng.uniform(0, W), rng.uniform(0, H)])
        t = np.array([CW / 2, CH / 2]) - A @ cs
        M = np.concatenate([A, t[:, None]], 1).astype(np.float32)
        warp = lambda img: cv2.warpAffine(img, M, (CW, CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        cur = torch.from_numpy(warp(im)).permute(2, 0, 1).float() / 255
        if self.two_frame:  # identical colour jitter for both frames: jitter them side by side as one image
            pr = torch.from_numpy(warp(pim)).permute(2, 0, 1).float() / 255
            both = self.cj(torch.cat([cur, pr], 2)); cur, pr = both[:, :, :CW], both[:, :, CW:]
        else:
            cur, pr = self.cj(cur), None
        tf = lambda p: A @ np.asarray(p, np.float64) + t
        ball_pts = [tf([(b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H])] if has_ball else []
        pb = self.d["boxes"][k]; pb = pb[pb[:, 0] >= 0]
        pl_pts = [tf([(q[0] + q[2]) / 2 * W, (q[1] + q[3]) / 2 * H]) for q in pb]
        return (to_input(cur, pr), block_labels(ball_pts, (CH // 4, CW // 4), 4, BALL_DELTA),
                block_labels(pl_pts, (CH // 16, CW // 16), 16, PLAYER_DELTA))


@torch.no_grad()
def fb_outputs(model, names, split, dev="cuda", bs=8, topk=5, prev=None):
    """Full-frame inference -> per-frame ball candidates [(cx, cy, p)] (normalised) and player x-histograms (BINS) from the player map."""
    model.eval(); cands, hists = [], []
    for i in range(0, len(names), bs):
        rd = lambda ns: torch.stack([torch.from_numpy(read_rgb(split, n)).permute(2, 0, 1) for n in ns]).to(dev).float() / 255
        cur = rd(names[i:i + bs])
        x = (cur - L.MEAN.to(dev)) / L.STD.to(dev)
        if prev is not None:
            x = torch.cat([x, (cur - rd(prev[i:i + bs])) / L.STD.to(dev)], 1)
        lb, lp = model(x)
        pb = lb.softmax(1)[:, 1]; pk = (pb == F.max_pool2d(pb[:, None], 3, 1, 1)[:, 0]).float() * pb
        B, h, w = pk.shape
        sc, ix = pk.view(B, -1).topk(topk, dim=1)
        for b in range(B):
            cands.append([((int(j) % w + 0.5) / w, (int(j) // w + 0.5) / h, float(s)) for s, j in zip(sc[b], ix[b])])
        col = lp.softmax(1)[:, 1].sum(1)  # (B, w16)
        hists.append(F.interpolate(col[:, None], size=BINS, mode="linear", align_corners=False)[:, 0].cpu().numpy())
    return cands, np.concatenate(hists)
