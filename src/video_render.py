"""Post-process cached tiled detections (static-overlay removal + tracking) and render the 9:16 crop video."""
import argparse
import json
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from video_demo import reader


def static_mask(F, thr, cell=32, frac=0.10):
    """Cells where a detection >= thr appears in more than `frac` of frames are overlays (logos, score bug)."""
    cnt = {}
    for f in F:
        seen = {(int((b[0] + b[2]) / 2 // cell), int((b[1] + b[3]) / 2 // cell)) for b in f if b[4] >= thr}
        for c in seen:
            cnt[c] = cnt.get(c, 0) + 1
    return {c for c, n in cnt.items() if n > frac * len(F)}


def track(F, thr, mask, cell=32, gate=250, hold=25):
    """Pick one ball per frame: score, minus a distance penalty to the last confirmed position."""
    pos, age, out = None, 10 ** 9, []
    for f in F:
        cands = []
        for b in f:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if b[4] < thr or any((int(cx // cell) + dx, int(cy // cell) + dy) in mask
                                 for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                continue
            pen = 0.0
            if pos is not None and age <= hold:
                dist = np.hypot(cx - pos[0], cy - pos[1])
                if dist > gate * (1 + age / 5) and b[4] < 0.8:  # implausible jump to a weak detection
                    continue
                pen = 0.5 * min(dist / gate, 1.0)
            cands.append((b[4] - pen, cx, cy, b[4]))
        if cands:
            _, cx, cy, s = max(cands)
            pos, age = (cx, cy), 0
            out.append((cx, cy, s))
        else:
            age += 1
            out.append(None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--out")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--hold", type=int, default=25)
    a = ap.parse_args()

    d = json.load(open(a.dets))
    W, H, F = d["W"], d["H"], d["dets"]
    mask = static_mask(F, a.thr)
    print("static cells:", sorted(mask))
    tr = track(F, a.thr, mask, hold=a.hold)
    print(f"ball tracked in {np.mean([t is not None for t in tr]):.1%} of frames")
    if not a.out:
        return

    cw = int(round(H * 9 / 16 / 2)) * 2
    ptr = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                              "stream=r_frame_rate", "-of", "json", a.video]))["streams"][0]
    num, den = map(int, ptr["r_frame_rate"].split("/"))
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W + cw}x{H}", "-r", str(num / den),
         "-i", "-", "-i", a.video, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", a.out], stdin=subprocess.PIPE)
    c, last_hit, lost = W / 2, None, 0
    for frame, t in zip(reader(a.video, W, H), tr):
        if t is not None:
            c = t[0] if last_hit is None else a.alpha * t[0] + (1 - a.alpha) * c
            last_hit = t
        x0 = int(round(min(max(c - cw / 2, 0), W - cw)))
        img = Image.fromarray(frame)
        left = img.copy()
        dr = ImageDraw.Draw(left)
        dr.rectangle([x0, 0, x0 + cw - 1, H - 1], outline=(0, 200, 255), width=4)
        if t is not None:
            dr.ellipse([t[0] - 14, t[1] - 14, t[0] + 14, t[1] + 14], outline=(255, 40, 40), width=3)
            dr.text((t[0] + 16, t[1] - 22), f"{t[2]:.2f}", fill=(255, 40, 40))
        canvas = Image.new("RGB", (W + cw, H))
        canvas.paste(left, (0, 0))
        canvas.paste(img.crop((x0, 0, x0 + cw, H)), (W, 0))
        enc.stdin.write(np.asarray(canvas).tobytes())
    enc.stdin.close()
    enc.wait()
    print("saved", a.out)


if __name__ == "__main__":
    main()
