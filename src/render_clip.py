"""Run a trained CropNet on a 16:9 video and write (1) the 9:16 crop and (2) a preview with the crop window drawn.
Usage: render_clip.py <run> <in.mp4> <out_prefix>"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sm_lib as L
import sm_data as D

run, src, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
ck = torch.load(D.ROOT / "results" / run / "best.pt", map_location="cuda")
c = ck["cfg"]
model = L.CropNet(c["backbone"], c["head"], c["two_frame"], pretrained=False).cuda()
model(L.prep(np.zeros((2, D.H, D.W, 3), np.uint8), np.zeros((2, D.H, D.W, 3), np.uint8), c["two_frame"]))
model.load_state_dict(ck["state"]); model.eval()

info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                       "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip().split(",")
IW, IH = int(info[0]), int(info[1]); num, den = info[2].split("/"); fps = float(num) / float(den)
raw = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
frames = []
while True:
    b = raw.stdout.read(IW * IH * 3)
    if len(b) < IW * IH * 3:
        break
    frames.append(np.frombuffer(b, np.uint8).reshape(IH, IW, 3))
n = len(frames); print(n, "frames", fps, "fps", flush=True)

# predict on the training sampling grid (every 3rd frame, previous frame 6 frames earlier), then interpolate to every frame
small = lambda f: np.asarray(Image.fromarray(f).resize((D.W, D.H), Image.BILINEAR))
idx = list(range(D.GAP, n, D.STRIDE))
cur = np.stack([small(frames[i]) for i in idx]); prv = np.stack([small(frames[i - D.GAP]) for i in idx])
out = []
with torch.no_grad():
    for i in range(0, len(idx), 32):
        x = L.prep(cur[i:i + 32], prv[i:i + 32], model.two_frame)
        y = model(x).float()
        out += [L.center_from_hist(h) for h in y.softmax(-1).cpu().numpy()] if model.head_kind == "hist" else y.cpu().tolist()
raw_c = np.array([L.clamp_center(v) for v in out])
sm_c = np.array([L.clamp_center(v) for v in L.smooth(raw_c, np.zeros(len(raw_c), int), **L.POST["ema_lim"])])
full = np.interp(np.arange(n), idx, sm_c)  # centre (0..1) for every frame
np.save(f"{prefix}_centers.npy", full)

cw = int(round(L.FRAC * IW)) // 2 * 2; OW, OH = 540, 960
def enc(path, w, h):
    return subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                             "-i", src, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac",
                             "-shortest", path], stdin=subprocess.PIPE)
vert, prev = enc(f"{prefix}_vertical.mp4", OW, OH), enc(f"{prefix}_preview.mp4", IW, IH)
for t, f in enumerate(frames):
    x0 = int(min(max(full[t] * IW - cw / 2, 0), IW - cw))
    vert.stdin.write(np.asarray(Image.fromarray(f[:, x0:x0 + cw]).resize((OW, OH), Image.LANCZOS)).tobytes())
    p = (f * 0.35).astype(np.uint8); p[:, x0:x0 + cw] = f[:, x0:x0 + cw]
    p[:4, x0:x0 + cw] = (255, 0, 0); p[-4:, x0:x0 + cw] = (255, 0, 0); p[:, x0:x0 + 4] = (255, 0, 0); p[:, x0 + cw - 4:x0 + cw] = (255, 0, 0)
    prev.stdin.write(p.tobytes())
for p in (vert, prev):
    p.stdin.close(); p.wait()
print("done", f"{prefix}_vertical.mp4", f"{prefix}_preview.mp4")
