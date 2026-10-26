"""SoccerNet SN-Tracking-2023 -> cached frames (896x504 JPEG) + ball / player annotations + ball-aware 9:16 crop target.

Official train -> train / valid (last N_VALID clips), official test -> test.  Output: data/sn/{train,valid,test}.npz and
data/sn/frames/<split>/<clip>_<frame>.jpg.  Reference ("oracle") crop: a window that contains the ball when it is visible,
among those the one covering most players; without a visible ball the densest-player window (as for SportsMOT)."""
import configparser
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from crop import CROP_W_FRAC
from sm_data import BINS, MAXP, density

ROOT = Path(__file__).resolve().parent.parent
SN = ROOT / "data" / "sn"
W, H = 896, 504
STRIDE = {"train": 4, "valid": 4, "test": 3}
N_VALID = 6


def oracle_center_ball(cx_players, ball_cx=None, frac=CROP_W_FRAC, grid=200):
    lo, hi = frac / 2, 1 - frac / 2
    cs = np.linspace(lo, hi, grid)
    if ball_cx is not None:
        ok = np.abs(cs - ball_cx) <= frac / 2
        cs = cs[ok] if ok.any() else np.array([min(max(ball_cx, lo), hi)])
    if len(cx_players) == 0:
        return float(cs[np.argmin(np.abs(cs - (ball_cx if ball_cx is not None else 0.5)))])
    cnt = np.array([(np.abs(cx_players - c) <= frac / 2).sum() for c in cs])
    best = np.flatnonzero(cnt == cnt.max())
    ref = ball_cx if ball_cx is not None else np.median(cx_players)
    return float(cs[best[np.argmin(np.abs(cs[best] - ref))]])


def parse_clip(z, split, clip):
    gi = configparser.ConfigParser(); gi.read_string(z.read(f"{split}/{clip}/gameinfo.ini").decode())
    role = {}
    for k, v in gi["Sequence"].items():
        if k.startswith("trackletid_"):
            v = v.strip()
            role[int(k.split("_")[1])] = "ball" if v.startswith("ball") else "player" if v.startswith(("player", "goalkeeper")) else "other"
    si = configparser.ConfigParser(); si.read_string(z.read(f"{split}/{clip}/seqinfo.ini").decode())
    iw, ih, n = int(si["Sequence"]["imWidth"]), int(si["Sequence"]["imHeight"]), int(si["Sequence"]["seqLength"])
    ball, players = {}, {}
    for line in z.read(f"{split}/{clip}/gt/gt.txt").decode().splitlines():
        p = line.split(",")
        f, tid = int(p[0]), int(p[1]); x, y, w, h = (float(v) for v in p[2:6])
        box = [x / iw, y / ih, (x + w) / iw, (y + h) / ih]
        r = role.get(tid, "other")
        if r == "ball":
            ball[f] = box
        elif r == "player":
            players.setdefault(f, []).append(box)
    return n, ball, players


def build_clip(zip_path, split, clip, stride, frames_dir):
    z = zipfile.ZipFile(zip_path)
    n, ball, players = parse_clip(z, split, clip)
    rows = []
    for f in range(1, n + 1, stride):
        name = f"{clip}_{f:06d}.jpg"
        im = Image.open(io.BytesIO(z.read(f"{split}/{clip}/img1/{f:06d}.jpg"))).convert("RGB").resize((W, H), Image.BILINEAR)
        im.save(frames_dir / name, quality=90)
        b = np.array(players.get(f, []), np.float32).reshape(-1, 4)
        cx = (b[:, 0] + b[:, 2]) / 2
        bb = np.array(ball[f], np.float32) if f in ball else -np.ones(4, np.float32)
        pad = -np.ones((MAXP, 4), np.float32); pad[: min(len(b), MAXP)] = b[:MAXP]
        rows.append((name, f, bb, pad, oracle_center_ball(cx, (bb[0] + bb[2]) / 2 if f in ball else None), density(cx)))
    return clip, rows


def build(split_name, zip_split, clips, zip_path):
    frames_dir = SN / "frames" / split_name; frames_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda c: build_clip(zip_path, zip_split, c, STRIDE[split_name], frames_dir), clips))
    names, frame, ball, boxes, ctr, dens, seq = [], [], [], [], [], [], []
    for si, (clip, rows) in enumerate(res):
        for name, f, bb, pad, c, d in rows:
            names.append(name); frame.append(f); ball.append(bb); boxes.append(pad); ctr.append(c); dens.append(d); seq.append(si)
    np.savez(SN / f"{split_name}.npz", names=np.array(names), frame=np.array(frame), ball=np.stack(ball), boxes=np.stack(boxes),
             center=np.array(ctr, np.float32), density=np.stack(dens), seq=np.array(seq), seqs=np.array([c for c, _ in res]))
    vis = float((np.stack(ball)[:, 0] >= 0).mean())
    print(split_name, len(names), "frames,", len(res), "clips, ball visible in", f"{100 * vis:.1f}%", flush=True)


def clips_of(zip_path, split):
    z = zipfile.ZipFile(zip_path)
    return sorted({n.split("/")[1] for n in z.namelist() if n.startswith(f"{split}/") and n.endswith("gt/gt.txt")})


def main():
    tr = clips_of(SN / "train.zip", "train")
    build("train", "train", tr[:-N_VALID], SN / "train.zip")
    build("valid", "train", tr[-N_VALID:], SN / "train.zip")
    build("test", "test", clips_of(SN / "test.zip", "test"), SN / "test.zip")


if __name__ == "__main__":
    main()
