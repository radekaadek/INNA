"""Models, crop selection, smoothing and metrics for player-driven 9:16 cropping (SportsMOT football)."""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from crop import CROP_W_FRAC
from sm_data import BINS, OUT

FRAC = CROP_W_FRAC
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_split(name):
    d = np.load(OUT / f"{name}.npz")
    return {k: d[k] for k in d.files}


class CropNet(nn.Module):
    """Backbone -> column features (mean over height) -> head.
    head='hist': logits for a BINS-bin player-density histogram along x; head='reg': crop centre in (0,1).
    two_frame: input is [frame, frame - previous frame] (6 channels)."""

    def __init__(self, backbone="resnet18", head="hist", two_frame=False, pretrained=True):
        super().__init__()
        self.head_kind, self.two_frame = head, two_frame
        cin = 6 if two_frame else 3
        if backbone == "resnet18":
            r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
            if two_frame:
                r.conv1 = self._widen(r.conv1)
            self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
            ch = 512
        elif backbone == "mobilenetv3":
            m = torchvision.models.mobilenet_v3_large(
                weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            if two_frame:
                m.features[0][0] = self._widen(m.features[0][0])
            self.body, ch = m.features, 960
        elif backbone == "mobilenetv4s":
            import timm
            m = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=pretrained, features_only=True,
                                  out_indices=(4,), in_chans=cin)
            self.body, ch = m, m.feature_info.channels()[-1]
        else:
            raise ValueError(backbone)
        self.backbone = backbone
        self.col = nn.Sequential(nn.Conv2d(ch, 128, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        if head == "hist":
            self.out = nn.Conv1d(128, 1, 3, padding=1)
        else:
            self.out = nn.Sequential(nn.Flatten(), nn.LazyLinear(128), nn.ReLU(True), nn.Linear(128, 1))

    @staticmethod
    def _widen(conv):
        """3->6 input channels; the extra (frame-difference) channels start at zero."""
        new = nn.Conv2d(6, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:, :3] = conv.weight
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
        return new

    def forward(self, x):
        f = self.body(x)
        f = f[-1] if isinstance(f, (list, tuple)) else f
        f = self.col(f)
        if self.head_kind == "hist":
            c = f.mean(2)                                   # (B, 128, w)
            return F.interpolate(self.out(c), size=BINS, mode="linear", align_corners=False)[:, 0]  # (B, BINS)
        return torch.sigmoid(self.out(f.mean(2, keepdim=True))[:, 0])  # (B,)


def prep(cur, prev=None, two_frame=False, dev="cuda"):
    """uint8 (B,H,W,3) numpy -> normalised float tensor (optionally with the frame-difference channels)."""
    x = torch.from_numpy(cur).to(dev).permute(0, 3, 1, 2).float() / 255
    x = (x - MEAN.to(dev)) / STD.to(dev)
    if two_frame:
        p = torch.from_numpy(prev).to(dev).permute(0, 3, 1, 2).float() / 255
        p = (p - MEAN.to(dev)) / STD.to(dev)
        x = torch.cat([x, x - p], 1)
    return x


# ---- crop selection -------------------------------------------------------------------------------------------------
def center_from_hist(h):
    """h: (BINS,) non-negative mass along x -> centre (0..1) of the 9:16 window covering the most mass."""
    h = np.asarray(h, np.float64)
    k = max(1, int(round(FRAC * len(h))))
    cs = np.convolve(h, np.ones(k), mode="valid")  # window sums
    i = int(np.argmax(cs))
    return (i + k / 2) / len(h)


def clamp_center(c):
    return float(min(max(c, FRAC / 2), 1 - FRAC / 2))


def smooth(centers, seq, alpha=1.0, max_step=None):
    """Per-video exponential smoothing with optional pan-speed limit (fraction of frame width per sample)."""
    out = np.array(centers, np.float64)
    for s in np.unique(seq):
        idx = np.flatnonzero(seq == s)
        prev = out[idx[0]]
        for i in idx[1:]:
            c = alpha * out[i] + (1 - alpha) * prev
            if max_step is not None:
                c = prev + np.clip(c - prev, -max_step, max_step)
            out[i] = prev = c
    return out


POST = {"raw": dict(alpha=1.0), "ema": dict(alpha=0.5), "ema_lim": dict(alpha=0.5, max_step=0.02)}


# ---- metrics ----------------------------------------------------------------------------------------------------------
def coverage(boxes, centers):
    """Mean fraction of players (box centre) inside the crop window, per frame."""
    cx = (boxes[..., 0] + boxes[..., 2]) / 2
    valid = boxes[..., 0] >= 0
    inside = (np.abs(cx - np.asarray(centers)[:, None]) <= FRAC / 2) & valid
    return inside.sum(1) / np.maximum(valid.sum(1), 1)


def score(d, centers, post="raw"):
    """Metrics for predicted centres on split `d`, after post-processing."""
    c = np.array([clamp_center(x) for x in centers])
    c = np.array([clamp_center(x) for x in smooth(c, d["seq"], **POST[post])])
    cov = coverage(d["boxes"], c)
    err = np.abs(c - d["center"])
    jit = np.concatenate([np.abs(np.diff(c[d["seq"] == s])) for s in np.unique(d["seq"])])
    return {"players_in_crop": float(cov.mean()), "full_crop_frac": float((cov >= 0.999).mean()),
            "center_mae": float(err.mean()), "jitter": float(jit.mean())}


def baselines(d):
    cx = (d["boxes"][..., 0] + d["boxes"][..., 2]) / 2
    valid = d["boxes"][..., 0] >= 0
    mean = np.array([cx[i][valid[i]].mean() for i in range(len(cx))])
    med = np.array([np.median(cx[i][valid[i]]) for i in range(len(cx))])
    from sm_data import oracle_center
    return {"center": np.full(len(cx), 0.5), "mean_players": mean, "median_players": med, "oracle": d["center"]}


@torch.no_grad()
def predict_centers(model, d, bs=32, dev="cuda"):
    model.eval()
    out = []
    for i in range(0, len(d["images"]), bs):
        x = prep(d["images"][i:i + bs], d["prev"][i:i + bs], model.two_frame, dev)
        y = model(x).float()
        if model.head_kind == "hist":
            p = y.softmax(-1).cpu().numpy()
            out += [center_from_hist(h) for h in p]
        else:
            out += y.cpu().tolist()
    return np.array(out)


@torch.no_grad()
def latency(model, dev, n=60, warm=10, shape=(252, 448)):
    model.to(dev).eval()
    x = torch.rand(1, 6 if model.two_frame else 3, *shape, device=dev)
    ts = []
    for _ in range(n + warm):
        t = time.perf_counter(); model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts[warm:]) * 1000)


def n_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
