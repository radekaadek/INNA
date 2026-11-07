"""Backlog #6/#13: high-resolution second pass. For each frame take the top-10 ball_r18 candidates, cut a window around each from the ORIGINAL 1080p frame
(read from the SoccerNet zips), run ball_r18 on it at scale r (r=1: native, ball ~2.1x larger than in the 896-wide frames it was trained on) and keep the
heatmap peak near the window centre as a new score. Caches results/exp/hires_<split>_<r>.pkl (list per frame of 10 scores aligned with cands_ball_r18)."""
import io, pickle, sys, zipfile
import numpy as np, torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import sn_lib as S
from sm_data import ROOT

RES = ROOT / "results"; SN = ROOT / "data" / "sn"; WIN = 192; SCALES = (0.5, 0.75, 1.0)


def main(split):
    d = S.load(split); zp = SN / ("test.zip" if split == "test" else "train.zip"); zs = "test" if split == "test" else "train"
    cands = pickle.load(open(RES / "exp" / "cands_ball_r18.pkl", "rb"))[split]
    m = S.BallNet(pretrained=False).cuda(); m.load_state_dict(torch.load(RES / "ball_r18" / "best.pt")["state"]); m.eval()
    mean, std = S.L.MEAN.cuda(), S.L.STD.cuda()
    out = {r: [None] * len(cands) for r in SCALES}
    clips = sorted(set(n.rsplit("_", 1)[0] for n in d["names"]))
    z = zipfile.ZipFile(zp)
    for ci, clip in enumerate(clips):
        idx = [i for i, n in enumerate(d["names"]) if n.rsplit("_", 1)[0] == clip]
        def read(i):
            f = int(d["names"][i].rsplit("_", 1)[1][:6])
            return Image.open(io.BytesIO(z.read(f"{zs}/{clip}/img1/{f:06d}.jpg"))).convert("RGB")
        with ThreadPoolExecutor(4) as ex:
            ims = ex.map(read, idx)
            for i, im in zip(idx, ims):
                iw, ih = im.size
                for r in SCALES:
                    L = WIN / r; crops = []
                    for cx, cy, sc in cands[i]:
                        x0, y0 = cx * iw - L / 2, cy * ih - L / 2
                        crops.append(np.asarray(im.crop((int(x0), int(y0), int(x0 + L), int(y0 + L))).resize((WIN, WIN), Image.BILINEAR)))
                    if not crops:
                        out[r][i] = []; continue
                    x = torch.from_numpy(np.stack(crops)).cuda().permute(0, 3, 1, 2).float() / 255
                    with torch.no_grad():
                        hm = m((x - mean) / std).sigmoid()
                    c = hm.shape[-1] // 2
                    out[r][i] = [float(v) for v in hm[:, 0, c - 4:c + 4, c - 4:c + 4].amax((1, 2)).cpu()]
        print(split, clip, ci + 1, "/", len(clips), flush=True)
    for r in SCALES:
        pickle.dump(out[r], open(RES / "exp" / f"hires_{split}_{r}.pkl", "wb"))


if __name__ == "__main__":
    main(sys.argv[1])
