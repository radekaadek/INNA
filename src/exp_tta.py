"""Test-time augmentation for ball_r18: flip, 1.25x upscale, both. Candidates -> results/exp/cands_ball_r18_<mode>.pkl; Viterbi tuned on valid -> results/exp/tta.json"""
import itertools, json, pickle
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
import sm_lib as L
import sn_lib as S
from exp_topk import viterbi, objective, oracle
from sm_data import ROOT

RES = ROOT / "results"; SN = S.SN


@torch.no_grad()
def cands(model, names, split, mode, bs=4, topk=10):
    out = []
    for i in range(0, len(names), bs):
        x = torch.stack([torch.from_numpy(np.asarray(Image.open(SN / "frames" / split / n).convert("RGB"))).permute(2, 0, 1) for n in names[i:i + bs]])
        x = ((x.cuda().float() / 255) - L.MEAN.cuda()) / L.STD.cuda()
        hs = []
        for flip in ([False, True] if "flip" in mode else [False]):
            for sc in ([1.0, 1.25] if "scale" in mode else [1.0]):
                xi = x.flip(-1) if flip else x
                if sc != 1.0:
                    xi = F.interpolate(xi, scale_factor=sc, mode="bilinear", align_corners=False)
                h = model(xi).sigmoid()
                if sc != 1.0:
                    h = F.interpolate(h, size=(x.shape[2] // 4, x.shape[3] // 4), mode="bilinear", align_corners=False)
                hs.append(h.flip(-1) if flip else h)
        hm = torch.stack(hs).mean(0)
        pk = (hm == F.max_pool2d(hm, 3, 1, 1)).float() * hm
        B, _, h_, w_ = pk.shape; sc_, ix = pk.view(B, -1).topk(topk, dim=1)
        for b in range(B):
            out.append([((int(j) % w_ + 0.5) / w_, (int(j) // w_ + 0.5) / h_, float(s)) for s, j in zip(sc_[b], ix[b])])
    return out


def main():
    va, te = S.load("valid"), S.load("test")
    m = S.BallNet(pretrained=False).cuda(); m.load_state_dict(torch.load(RES / "ball_r18" / "best.pt")["state"]); m.eval()
    res = {}
    for mode in ["flip", "scale", "flip+scale"]:
        f = RES / "exp" / f"cands_ball_r18_{mode}.pkl"
        if f.exists():
            c = pickle.load(open(f, "rb"))
        else:
            c = {"valid": cands(m, va["names"], "valid", mode), "test": cands(m, te["names"], "test", mode)}; pickle.dump(c, open(f, "wb"))
        vc = S.filter_static(c["valid"], va["seq"]); tc = S.filter_static(c["test"], te["seq"])
        grid = list(itertools.product([0.05, 0.1, 0.2], [0.3, 0.6, 1.0], [0.1, 0.2, 0.3], [4, 8, 16]))
        bv = max((objective(va, viterbi(vc, va["seq"], thr=a, lam=b, miss=mm, hold=h)), a, b, mm, h) for a, b, mm, h in grid)
        kw = dict(thr=bv[1], lam=bv[2], miss=bv[3], hold=bv[4])
        res[mode] = {"oracle_valid": oracle(va, vc), "oracle_test": oracle(te, tc), "params": kw, "valid": S.ball_hit_rate(va, viterbi(vc, va["seq"], **kw)),
                     "test": S.ball_hit_rate(te, viterbi(tc, te["seq"], **kw)), "test_causal": S.ball_hit_rate(te, viterbi(tc, te["seq"], lag=0, **kw))}
        print(mode, json.dumps(res[mode]), flush=True)
    json.dump(res, open(RES / "exp" / "tta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
