"""Tiled ball detection on a video. Each 16:9 tile is stretched to 640x640 like the training images
(so the ball keeps the training scale/aspect); detections are mapped back to frame coordinates and cached."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import models
from video_demo import reader

RES = Path(__file__).resolve().parent.parent / "results"


def tiles(W, H, tw, th, nx, ny):
    xs = np.linspace(0, W - tw, nx).round().astype(int)
    ys = np.linspace(0, H - th, ny).round().astype(int)
    return [(x, y) for y in ys for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cn_mnv3")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", default="640:3:2,1280:1:1,2560:1:1",
                    help="tile_w:nx:ny list; tile height = 9/16 tile_w, tiles larger than the frame are grey-padded")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ck = torch.load(RES / a.run / "best.pt", map_location="cpu")
    model = models.build(ck["model"], pretrained=False)
    model.load_state_dict(ck["state"])
    model.to(a.device).eval()
    model.tta = a.tta

    pr = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json",
         a.video]))["streams"][0]
    W, H = pr["width"], pr["height"]
    PX, PY = 0, 0  # grey padding so tiles larger than the frame can be cut
    for sc in a.scales.split(","):
        tw, nx, ny = map(int, sc.split(":"))
        th = round(tw * 9 / 16)
        PX, PY = max(PX, (tw - W) // 2 + 1), max(PY, (th - H) // 2 + 1)
    grid = []  # (tile_w, tile_h, x0, y0) in frame coordinates (may be negative)
    for sc in a.scales.split(","):
        tw, nx, ny = map(int, sc.split(":"))
        th = round(tw * 9 / 16)
        ox, oy = (W - tw) / 2, (H - th) / 2  # offsets of the tile grid when the tile is larger than the frame
        if tw >= W:
            grid.append((tw, th, round(ox), round(oy)))
        else:
            grid += [(tw, th, xo, yo) for xo, yo in tiles(W, H, tw, th, nx, ny)]

    allf = []
    for n, frame in enumerate(reader(a.video, W, H)):
        if a.max_frames and n >= a.max_frames:
            break
        x = torch.from_numpy(frame.copy()).to(a.device).permute(2, 0, 1).float().div(255)
        x = F.pad(x, (PX, PX, PY, PY), value=0.5)
        batch = [F.interpolate(x[None, :, y + PY:y + PY + th, xo + PX:xo + PX + tw], size=(640, 640), mode="bilinear",
                               align_corners=False)[0] for tw, th, xo, y in grid]
        outs = [o for i in range(0, len(batch), 3) for o in models.predict(model, batch[i:i + 3])]
        dets = []
        for (tw, th, xo, y), o in zip(grid, outs):
            for b, s in zip(o["boxes"].cpu().tolist(), o["scores"].cpu().tolist()):
                if s < 0.05:
                    continue
                sx, sy = tw / 640, th / 640
                dets.append([xo + b[0] * sx, y + b[1] * sy, xo + b[2] * sx, y + b[3] * sy, s])
        allf.append(dets)
        if n % 250 == 0:
            print(n, flush=True)
    json.dump({"W": W, "H": H, "dets": allf}, open(a.out, "w"))
    print("done", len(allf))


if __name__ == "__main__":
    main()
