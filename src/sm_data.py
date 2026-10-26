"""SportsMOT football -> cached low-res frames + player-based 9:16 crop targets.

Official train -> train/valid (split by video), official val -> test (SportsMOT test labels are hidden).
Output: results/sm/{train,valid,test}.npz with images (N,H,W,3 uint8), prev (frame `GAP` frames earlier), per-frame player
boxes (normalised xyxy, padded with -1), sequence id / frame index, oracle crop centre, player-count histogram."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from crop import CROP_W_FRAC

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "sportsmot"
OUT = ROOT / "results" / "sm"
W, H = 448, 252  # cached resolution (16:9)
STRIDE = 3       # keep every 3rd frame (8.3 fps)
GAP = 6          # previous-frame offset (in original frames) for the 2-frame model
BINS = 64        # resolution of the player-density histogram along x
MAXP = 40        # max boxes kept per frame
VALID_SEQS = 3   # last 3 official-train football videos form the internal validation split


def football(split):
    names = set((SRC / "football.txt").read_text().split())
    return sorted(s.name for s in (SRC / split).iterdir() if s.name in names)


def oracle_center(cx_players, frac=CROP_W_FRAC, grid=200):
    """Centre (0..1) of the window covering the most players; ties -> the plateau middle nearest to the median."""
    lo, hi = frac / 2, 1 - frac / 2
    cs = np.linspace(lo, hi, grid)
    if len(cx_players) == 0:
        return 0.5
    cnt = np.array([(np.abs(cx_players - c) <= frac / 2).sum() for c in cs])
    best = np.flatnonzero(cnt == cnt.max())
    med = np.median(cx_players)
    return float(cs[best[np.argmin(np.abs(cs[best] - med))]])


def density(cx_players, bins=BINS):
    h, _ = np.histogram(cx_players, bins=bins, range=(0, 1))
    return h.astype(np.float32)


def read_gt(seq_dir):
    gt = np.loadtxt(seq_dir / "gt" / "gt.txt", delimiter=",")
    ini = dict(l.strip().split("=") for l in (seq_dir / "seqinfo.ini").read_text().splitlines() if "=" in l)
    iw, ih = float(ini["imWidth"]), float(ini["imHeight"])
    per = {}
    for f, _, x, y, w, h, *_ in gt:
        per.setdefault(int(f), []).append([x / iw, y / ih, (x + w) / iw, (y + h) / ih])
    return per, int(ini["seqLength"])


def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR))


def build(split, seqs, out_name):
    imgs, prev, boxes, seq_id, frame, ctr, dens = [], [], [], [], [], [], []
    for si, s in enumerate(seqs):
        d = SRC / split / s
        per, n = read_gt(d)
        for f in range(1 + GAP, n + 1, STRIDE):
            b = np.array(per.get(f, []), dtype=np.float32).reshape(-1, 4)
            if len(b) == 0:
                continue
            cx = (b[:, 0] + b[:, 2]) / 2
            imgs.append(load(d / "img1" / f"{f:06d}.jpg")); prev.append(load(d / "img1" / f"{f - GAP:06d}.jpg"))
            pad = -np.ones((MAXP, 4), np.float32); pad[: min(len(b), MAXP)] = b[:MAXP]
            boxes.append(pad); seq_id.append(si); frame.append(f)
            ctr.append(oracle_center(cx)); dens.append(density(cx))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{out_name}.npz", images=np.stack(imgs), prev=np.stack(prev), boxes=np.stack(boxes),
             seq=np.array(seq_id), frame=np.array(frame), center=np.array(ctr, np.float32), density=np.stack(dens),
             seqs=np.array(seqs))
    print(out_name, len(imgs), "frames,", len(seqs), "videos", flush=True)


def main():
    argparse.ArgumentParser().parse_args()
    tr = football("train")
    build("train", tr[:-VALID_SEQS], "train")
    build("train", tr[-VALID_SEQS:], "valid")
    build("val", football("val"), "test")


if __name__ == "__main__":
    main()
