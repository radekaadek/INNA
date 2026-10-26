"""Run a detector on a video: left = frame with detection + 9:16 window, right = resulting vertical crop."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import crop
import models

RES = Path(__file__).resolve().parent.parent / "results"


def reader(path, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    n = w * h * 3
    while len(buf := p.stdout.read(n)) == n:
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cn_mnv3")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.3, help="EMA factor for the crop centre (1 = no smoothing)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ck = torch.load(RES / a.run / "best.pt", map_location="cpu")
    model = models.build(ck["model"], pretrained=False)
    model.load_state_dict(ck["state"])
    model.to(a.device).eval()
    thr = json.load(open(RES / a.run / "test.json"))["thr"]

    pr = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "json", a.video]))["streams"][0]
    W, H = pr["width"], pr["height"]
    num, den = map(int, pr["r_frame_rate"].split("/"))
    fps = num / den
    cw = int(round(H * 9 / 16 / 2)) * 2
    sx, sy = W / 640, H / 640

    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W + cw}x{H}", "-r", str(fps),
         "-i", "-", "-i", a.video, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", a.out], stdin=subprocess.PIPE)

    prev, n, found = None, 0, 0
    for frame in reader(a.video, W, H):
        img = Image.fromarray(frame)
        x = torch.from_numpy(np.asarray(img.resize((640, 640), Image.BILINEAR))).permute(2, 0, 1).float().div(255)
        out = models.predict(model, [x.to(a.device)])[0]
        dets = [[*b.tolist(), s.item()] for b, s in zip(out["boxes"].cpu(), out["scores"].cpu())]
        best = max(dets, key=lambda d: d[4])
        hit = best[4] >= thr
        c = crop.choose_center(dets, thr, 640, prev, a.alpha)
        prev = c
        x0 = int(round(min(max(c * sx - cw / 2, 0), W - cw)))
        found += hit

        left = img.copy()
        d = ImageDraw.Draw(left)
        d.rectangle([x0, 0, x0 + cw - 1, H - 1], outline=(0, 200, 255), width=4)
        if hit:
            b = best
            d.rectangle([b[0] * sx - 6, b[1] * sy - 6, b[2] * sx + 6, b[3] * sy + 6], outline=(255, 40, 40), width=3)
            d.text((b[0] * sx, b[1] * sy - 20), f"{b[4]:.2f}", fill=(255, 40, 40))
        canvas = Image.new("RGB", (W + cw, H))
        canvas.paste(left, (0, 0))
        canvas.paste(img.crop((x0, 0, x0 + cw, H)), (W, 0))
        enc.stdin.write(np.asarray(canvas).tobytes())
        n += 1
        if n % 250 == 0:
            print(f"{n} frames, ball found in {found / n:.0%}", flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"done: {n} frames, detection above threshold ({thr}) in {found / n:.1%} -> {a.out}")


if __name__ == "__main__":
    main()
