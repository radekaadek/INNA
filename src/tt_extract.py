"""TeamTrack (static fish-eye panorama, 6500x1000) -> per-sample detections from the models trained on broadcast footage.
Every 3rd frame (0.12 s) is processed tile by tile (native scale, 1056-px wide tiles over the pitch band).  Output: results/tt/<clip>.pkl.
Usage: tt_extract.py <clip> [<clip> ...]      (clip = folder name in data/teamtrack, e.g. F_1680_1710)"""
import os
import pickle
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from ultralytics import YOLO

import sm_lib as L
import sn_fb as FB
import sn_lib as S
from evaluate_sn import load_cropnet
from sm_data import BINS, ROOT

DATASET = os.environ.get("TT_DATA", "teamtrack")  # teamtrack | soccertrack
TT = ROOT / "data" / DATASET
OUT = ROOT / "results" / ("tt" if DATASET == "teamtrack" else "st")
IW, IH = 6500, 1000
TW = 1056
Y0, Y1 = 288, 928            # pitch band (all GT players lie inside it); 640 px tall
XS = np.linspace(0, IW - TW, 8).astype(int)
GRID = 650                   # panorama histogram resolution (10 px bins)


def read_frames(clip, stride=3):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(TT / clip / "img1.mp4"), "-vf", f"select='not(mod(n\\,{stride}))'", "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(IW * IH * 3)
        if len(b) < IW * IH * 3:
            return
        yield np.frombuffer(b, np.uint8).reshape(IH, IW, 3)


def yolo_boxes(model, tiles_bgr, conf):
    """Run on all tiles, return panorama boxes (x1, y1, x2, y2, conf, cls) after cross-tile NMS (per class)."""
    res = model.predict(tiles_bgr, imgsz=TW, conf=conf, verbose=False, device=0)
    allb = []
    for r, x0 in zip(res, XS):
        b = r.boxes
        if len(b) == 0:
            continue
        xy = b.xyxy.cpu().numpy(); xy[:, [0, 2]] += x0; xy[:, [1, 3]] += Y0
        allb.append(np.concatenate([xy, b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1))
    if not allb:
        return np.zeros((0, 6), np.float32)
    a = np.concatenate(allb); keep = []
    for c in np.unique(a[:, 5]):
        idx = np.flatnonzero(a[:, 5] == c)
        k = nms(torch.from_numpy(a[idx, :4]), torch.from_numpy(a[idx, 4]), 0.5)
        keep += list(idx[k.numpy()])
    return a[keep].astype(np.float32)


def to_grid(vals, x0, width):
    """Accumulate a 1-D profile covering panorama x in [x0, x0 + width) into (sum, count) arrays of GRID bins."""
    n = len(vals); acc = np.zeros(GRID); cnt = np.zeros(GRID)
    b0, b1 = int(x0 / IW * GRID), int((x0 + width) / IW * GRID)
    xi = np.linspace(0, n - 1, b1 - b0)
    acc[b0:b1] = np.interp(xi, np.arange(n), vals); cnt[b0:b1] = 1
    return acc, cnt


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    yolo = {"yolo_sn": (YOLO(str(ROOT / "results/yolo11n_sn/train/weights/best.pt")), 0.2, 0),
            "yolo_sm": (YOLO(str(ROOT / "results/yolo11n_sm/train/weights/best.pt")), 0.2, 0),
            "yolo_coco": (YOLO("yolo11n.pt"), 0.2, 0)}
    r18 = load_cropnet("players_sn_r18")
    bn = S.BallNet(pretrained=False).cuda(); bn.load_state_dict(torch.load(ROOT / "results/ball_r18/best.pt")["state"]); bn.eval()
    ck = torch.load(ROOT / "results/fb_tiny/best.pt", map_location="cuda")
    fb = FB.FBNet("tiny", pretrained=False).cuda(); fb.load_state_dict(ck["state"]); fb.eval()
    mean, std = L.MEAN.cuda(), L.STD.cuda()
    for clip in sys.argv[1:]:
        rec = {k: [] for k in ["yolo_sn", "yolo_sm", "yolo_coco", "yolo_sn_ball", "hist_r18", "hist_fb", "ball_r18", "ball_fb"]}
        for fi, frame in enumerate(read_frames(clip)):
            band = frame[Y0:Y1]
            tiles = [band[:, x0:x0 + TW] for x0 in XS]
            tb = [np.ascontiguousarray(t[:, :, ::-1]) for t in tiles]  # BGR for ultralytics
            for k, (m, conf, _) in yolo.items():
                b = yolo_boxes(m, tb, conf if k != "yolo_sn" else 0.05)
                if k == "yolo_sn":  # 2-class: players (cls 0) and ball (cls 1)
                    rec["yolo_sn_ball"].append(b[b[:, 5] == 1]); b = b[(b[:, 5] == 0) & (b[:, 4] >= 0.2)]
                rec[k].append(b[b[:, 5] == 0] if k != "yolo_sn" else b)
            x = torch.from_numpy(np.stack(tiles)).cuda().permute(0, 3, 1, 2).float() / 255
            xn = (x - mean) / std
            # SN-trained ResNet-18 player histogram: each tile resized to 448x252 (broadcast-like), stitched on the panorama grid
            small = F.interpolate(x, size=(252, 448), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).mul(255).byte().cpu().numpy()
            h = r18(L.prep(small, None, False)).float().softmax(-1).cpu().numpy()
            acc = np.zeros(GRID); cnt = np.zeros(GRID)
            for hh, x0 in zip(h, XS):
                a, c = to_grid(hh, x0, TW); acc += a; cnt += c
            rec["hist_r18"].append(acc / np.maximum(cnt, 1))
            # ball networks at native scale
            hm = bn(xn).sigmoid(); lb, lp = fb(xn)
            for name, prob in (("ball_r18", hm[:, 0]), ("ball_fb", lb.softmax(1)[:, 1])):
                pk = (prob == F.max_pool2d(prob[:, None], 3, 1, 1)[:, 0]).float() * prob
                T, hh_, ww_ = pk.shape
                sc, ix = pk.view(T, -1).topk(5, dim=1); c = []
                for t in range(T):
                    for s, j in zip(sc[t].tolist(), ix[t].tolist()):
                        c.append((XS[t] + (j % ww_ + .5) / ww_ * TW, Y0 + (j // ww_ + .5) / hh_ * (Y1 - Y0), s))
                rec[name].append(sorted(c, key=lambda q: -q[2])[:8])
            col = lp.softmax(1)[:, 1].sum(1).cpu().numpy(); acc = np.zeros(GRID); cnt = np.zeros(GRID)
            for cc, x0 in zip(col, XS):
                a, c = to_grid(cc, x0, TW); acc += a; cnt += c
            rec["hist_fb"].append(acc / np.maximum(cnt, 1))
            if fi % 25 == 0:
                print(clip, "sample", fi, flush=True)
        pickle.dump(rec, open(OUT / f"{clip}.pkl", "wb"))
        print(clip, "done", len(rec["yolo_sn"]), "samples", flush=True)


if __name__ == "__main__":
    main()
