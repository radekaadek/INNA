"""Same as render_clip_ball.py but with the offline Viterbi ball tracker (exp_topk.viterbi).
9:16 crop of a 16:9 video driven by the tracked ball (ResNet-18 heatmap + tracker) and predicted player density.
Usage: render_clip_ball.py <in.mp4> <out_prefix> [post=ema_fast]
Writes <prefix>_vertical.mp4 (9:16 crop with audio), <prefix>_preview.mp4 (full frame, crop window in red, ball estimate in yellow)."""
import json
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

import sm_lib as L
import sn_lib as S
import sm_data as D
from exp_topk import viterbi
from evaluate_sn import POSTS, load_cropnet

src, prefix = sys.argv[1], sys.argv[2]
post = POSTS[sys.argv[3] if len(sys.argv) > 3 else "ema_fast"]
ev = json.load(open(D.ROOT / "results" / "sn_eval.json"))
trk = next(v["tracker"] for k, v in ev["systems"].items() if k.startswith("ball_r18|"))

info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", src],
                      capture_output=True, text=True).stdout.strip().split(",")
IW, IH = int(info[0]), int(info[1]); a, b = info[2].split("/"); fps = float(a) / float(b)
raw = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
frames = []
while True:
    buf = raw.stdout.read(IW * IH * 3)
    if len(buf) < IW * IH * 3:
        break
    frames.append(np.frombuffer(buf, np.uint8).reshape(IH, IW, 3))
n = len(frames); print(n, "frames", fps, "fps", flush=True)

idx = list(range(0, n, 3))  # same sampling grid as the evaluation (0.12 s at 25 fps)
big = [np.asarray(Image.fromarray(frames[i]).resize((S.W, S.H), Image.BILINEAR)) for i in idx]
bn = S.BallNet(pretrained=False).cuda(); bn.load_state_dict(torch.load(D.ROOT / "results" / "ball_r18" / "best.pt")["state"]); bn.eval()
cands = []
with torch.no_grad():
    for i in range(0, len(big), 8):
        x = torch.from_numpy(np.stack(big[i:i + 8])).cuda().permute(0, 3, 1, 2).float() / 255
        hm = bn(((x - L.MEAN.cuda()) / L.STD.cuda())).sigmoid()
        pk = (hm == torch.nn.functional.max_pool2d(hm, 3, 1, 1)).float() * hm
        B, _, h, w = pk.shape
        sc, ix = pk.view(B, -1).topk(5, dim=1)
        for k in range(B):
            cands.append([((int(j) % w + 0.5) / w, (int(j) // w + 0.5) / h, float(s)) for s, j in zip(sc[k], ix[k])])
seq0 = np.zeros(len(idx), int)
cands = S.filter_static(cands, seq0)
trk = dict(thr=0.1, lam=1.0, miss=0.2, hold=4)  # tuned on validation, see notes_viterbi.md
est = viterbi(cands, seq0, topk=5, **trk)  # offline (uses look-ahead)
pm = load_cropnet("players_sn_r18")
small = np.stack([np.asarray(Image.fromarray(f).resize((448, 252), Image.BILINEAR)) for f in big])
hists = []
with torch.no_grad():
    for i in range(0, len(small), 32):
        hists.append(pm(L.prep(small[i:i + 32], None, False)).float().softmax(-1).cpu().numpy())
hists = np.concatenate(hists)
c = S.policy_centers(hists, est[:, :2], est[:, 2])
c = np.array([L.clamp_center(v) for v in L.smooth(np.array([L.clamp_center(v) for v in c]), seq0, **post)])
full = np.interp(np.arange(n), idx, c)
bxy = np.stack([np.interp(np.arange(n), idx, est[:, 0]), np.interp(np.arange(n), idx, est[:, 1])], 1)
bconf = np.interp(np.arange(n), idx, est[:, 2]) > 0.05
np.save(f"{prefix}_centers.npy", full)
print("ball estimate on", f"{100 * (est[:, 2] > 0).mean():.0f}% of sampled frames; tracker", trk, flush=True)

cw = int(round(L.FRAC * IW)) // 2 * 2; OW, OH = 540, 960
def enc(path, w, h):
    return subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-i", src,
                             "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path], stdin=subprocess.PIPE)
vert, prev = enc(f"{prefix}_vertical.mp4", OW, OH), enc(f"{prefix}_preview.mp4", IW, IH)
for t, f in enumerate(frames):
    x0 = int(min(max(full[t] * IW - cw / 2, 0), IW - cw))
    vert.stdin.write(np.asarray(Image.fromarray(f[:, x0:x0 + cw]).resize((OW, OH), Image.LANCZOS)).tobytes())
    p = f.copy(); p[:, :x0] = (p[:, :x0] * 0.55).astype(np.uint8); p[:, x0 + cw:] = (p[:, x0 + cw:] * 0.55).astype(np.uint8)
    p[:5, x0:x0 + cw] = (255, 0, 0); p[-5:, x0:x0 + cw] = (255, 0, 0); p[:, x0:x0 + 5] = (255, 0, 0); p[:, x0 + cw - 5:x0 + cw] = (255, 0, 0)
    if bconf[t]:
        bx, by = int(bxy[t, 0] * IW), int(bxy[t, 1] * IH); r, th = 24, 3
        y0_, y1_, x0_, x1_ = max(by - r, 0), min(by + r, IH), max(bx - r, 0), min(bx + r, IW)
        p[y0_:y0_ + th, x0_:x1_] = (255, 255, 0); p[y1_ - th:y1_, x0_:x1_] = (255, 255, 0); p[y0_:y1_, x0_:x0_ + th] = (255, 255, 0); p[y0_:y1_, x1_ - th:x1_] = (255, 255, 0)
    prev.stdin.write(p.tobytes())
for q in (vert, prev):
    q.stdin.close(); q.wait()
print("done", f"{prefix}_vertical.mp4", f"{prefix}_preview.mp4")
