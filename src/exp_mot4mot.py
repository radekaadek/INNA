"""Ball post-processing of the SoccerNet-2023 3rd-place report (arXiv 2308.16651) applied to our candidates: smooth the top-1 centres with a 3rd-order
polynomial (window 51 frames at 25 fps = 17 of our 0.12 s samples), keep the candidate closest to the smoothed path if within 100 px (1080p => 0.052 of the frame width),
linearly interpolate the gaps. Offline (non-causal). Fixed parameters, nothing tuned. -> results/exp/mot4mot.json"""
import json, pickle
import numpy as np
import sn_lib as S
from sm_data import ROOT

RES = ROOT / "results"


def savgol_filter(y, w, order):
    # local polynomial fit over a window of w samples (shifted at the edges), like scipy's savgol_filter(mode='interp')
    n = len(y); h = w // 2; out = np.empty(n); x = np.arange(n)
    for t in range(n):
        a = min(max(t - h, 0), n - w); sl = slice(a, a + w)
        out[t] = np.polyval(np.polyfit(x[sl] - t, y[sl], order), 0)
    return out


def mot4mot(cands, seq, win=17, order=3, dmax=100 / 1920):
    out = np.zeros((len(cands), 3))
    for s in np.unique(seq):
        idx = np.flatnonzero(seq == s); n = len(idx)
        top = np.array([[c[0][0], c[0][1]] if c else [0.5, 0.5] for c in (cands[i] for i in idx)])
        w = min(win, n if n % 2 else n - 1)
        sm = np.stack([savgol_filter(top[:, k], w, min(order, w - 1)) for k in range(2)], 1)
        sel = np.full((n, 3), np.nan)
        for t, i in enumerate(idx):
            best = None
            for cx, cy, sc in cands[i]:
                d = np.hypot(cx - sm[t, 0], (cy - sm[t, 1]) * 0.5625)
                if d <= dmax and (best is None or d < best[0]):
                    best = (d, cx, cy, sc)
            if best:
                sel[t] = best[1:]
        ok = ~np.isnan(sel[:, 0]); tt = np.arange(n)
        if ok.sum() >= 2:
            for k in range(2):
                sel[~ok, k] = np.interp(tt[~ok], tt[ok], sel[ok, k])
            sel[~ok, 2] = 0.01
            out[idx] = sel
    return out


def main():
    res = {}
    for run in ["ball_r18", "b2_e20", "b2_prior"]:
        c = pickle.load(open(RES / "exp" / f"cands_{run}.pkl", "rb")); res[run] = {}
        for sp in ("valid", "test"):
            d = S.load(sp); cc = S.filter_static(c[sp], d["seq"])
            res[run][sp] = S.ball_hit_rate(d, mot4mot(cc, d["seq"]))
        print(run, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res[run].items()}, flush=True)
    json.dump(res, open(RES / "exp" / "mot4mot.json", "w"), indent=1)


if __name__ == "__main__":
    main()
