"""Top-K oracle recall and Viterbi (DP) tracker over ball candidates. Tuned on validation only; test reported once for the tuned settings.
Usage: python exp_topk.py   (CPU)  -> results/exp/topk.json"""
import itertools
import json
import pickle
import sys
import numpy as np
import sn_lib as S
from sm_data import ROOT

RES = ROOT / "results"
RUNS = ["ball_r18", "fb_tiny", "fb_r18", "fb_r18_v2"]


def gt_xy(d):
    return (d["ball"][:, 0] + d["ball"][:, 2]) / 2, (d["ball"][:, 1] + d["ball"][:, 3]) / 2


def oracle(d, cands, ks=(1, 2, 3, 5, 10), thr=0.02, min_sc=0.05):
    vis = d["ball"][:, 0] >= 0; gx, gy = gt_xy(d); out = {}
    for k in ks:
        hit = np.array([any(sc >= min_sc and np.hypot(cx - gx[i], (cy - gy[i]) * 0.5625) <= thr for cx, cy, sc in cands[i][:k]) for i in range(len(cands))])
        out[k] = float(hit[vis].mean())
    return out


def viterbi(cands, seq, thr=0.1, lam=0.5, gate=0.25, miss=0.15, hold=8, topk=5, lag=None, mw=0.0):
    """Offline max-sum path per clip. State = candidate (score>=thr) or MISS (carries the last real position of its best path).
    Gain: candidate score; MISS: `miss`. Move penalty: lam*min(d/(gate*(1+age/4)),1) from the last real position, age = frames since it.
    Output like track_ball: (N,3) x, y, conf (0 = none); a MISS run longer than `hold` is output as no estimate, shorter as held position."""
    out = np.zeros((len(cands), 3))
    for s in np.unique(seq):
        idx = np.flatnonzero(seq == s)
        # states per frame: list of (x, y, score) + final MISS at index -1
        sts = [[c for c in cands[i][:topk] if c[2] >= thr] for i in idx]
        score = None; back = []; arg = []  # last[j] = (x, y, age) for each state
        for t, st in enumerate(sts):
            n = len(st) + 1
            sc = np.full(n, -1e9); bk = np.zeros(n, int); lp = [None] * n
            for j in range(n):
                gain = st[j][2] if j < len(st) else miss
                if t == 0:
                    sc[j] = gain; lp[j] = (st[j][0], st[j][1], 0) if j < len(st) else None; continue
                for pj in range(len(last_prev)):
                    pl = last_prev[pj]
                    pen = 0.0
                    if j < len(st) and pl is not None:
                        d = np.hypot(st[j][0] - pl[0], (st[j][1] - pl[1]) * 0.5625)
                        if mw and pl[2] == 0 and len(st[j]) > 4:  # candidate carries a predicted previous position (x, y, score, bx, by)
                            d = (1 - mw) * d + mw * np.hypot(st[j][3] - pl[0], (st[j][4] - pl[1]) * 0.5625)
                        pen = lam * min(d / (gate * (1 + pl[2] / 4)), 1.0)
                    v = score[pj] + gain - pen
                    if v > sc[j]:
                        sc[j] = v; bk[j] = pj
                        lp[j] = (st[j][0], st[j][1], 0) if j < len(st) else (None if pl is None else (pl[0], pl[1], pl[2] + 1))
            back.append(bk); score = sc; last_prev = lp; arg.append(int(np.argmax(sc)))
        T = len(sts)
        if lag is None:  # offline: whole-clip optimum
            j = arg[-1]; path = []
            for t in range(T - 1, -1, -1):
                path.append(j); j = back[t][j]
            path = path[::-1]
        else:  # fixed lag: frame t is decoded from the best path at t+lag (lag=0: causal)
            path = []
            for t in range(T):
                u = min(t + lag, T - 1); j = arg[u]
                for v in range(u, t, -1):
                    j = back[v][j]
                path.append(j)
        pos = None; age = 10 ** 9
        for t, j in enumerate(path):
            i = idx[t]
            if j < len(sts[t]):
                pos = (sts[t][j][0], sts[t][j][1]); age = 0; out[i] = (*pos, sts[t][j][2])
            else:
                age += 1
                if pos is not None and age <= hold:
                    out[i] = (*pos, 0.01)
    return out


def objective(d, est):
    c = np.where(est[:, 2] > 0, est[:, 0], 0.5)
    return S.ball_hit_rate(d, est)["ball_recall"] + 0.5 * S.score_sn(d, c, "raw")["ball_in_crop"]


def main():
    va, te = S.load("valid"), S.load("test"); res = {}
    for run in RUNS:
        if run != "ball_r18" and len(sys.argv) > 1 and sys.argv[1] == "r18only":
            continue
        c = pickle.load(open(RES / "exp" / f"cands_{run}.pkl", "rb"))
        vc = S.filter_static(c["valid"], va["seq"]); tc = S.filter_static(c["test"], te["seq"])
        r = {"oracle_valid": oracle(va, vc), "oracle_test_info": oracle(te, tc)}
        # greedy tracker (current), tuned on valid with the same grid as evaluate_sn
        best = max(((objective(va, S.track_ball(vc, va["seq"], thr=t, hold=h)), t, h) for t, h in itertools.product([0.1, 0.2, 0.3, 0.5], [4, 8, 16])))
        g = S.track_ball(vc, va["seq"], thr=best[1], hold=best[2]); gt_ = S.track_ball(tc, te["seq"], thr=best[1], hold=best[2])
        r["greedy"] = {"params": best[1:], "valid": S.ball_hit_rate(va, g), "test": S.ball_hit_rate(te, gt_)}
        # viterbi, tuned on valid
        grid = list(itertools.product([0.05, 0.1, 0.2], [0.3, 0.6, 1.0], [0.1, 0.2, 0.3], [4, 8, 16]))
        sc = [(objective(va, viterbi(vc, va["seq"], thr=a, lam=b, miss=m, hold=h)), a, b, m, h) for a, b, m, h in grid]
        bv = max(sc)
        kw = dict(thr=bv[1], lam=bv[2], miss=bv[3], hold=bv[4])
        r["viterbi"] = {"params": kw, "valid": S.ball_hit_rate(va, viterbi(vc, va["seq"], **kw)), "test": S.ball_hit_rate(te, viterbi(tc, te["seq"], **kw))}
        r["viterbi_lag"] = {str(L): {"valid": S.ball_hit_rate(va, viterbi(vc, va["seq"], lag=L, **kw)), "test": S.ball_hit_rate(te, viterbi(tc, te["seq"], lag=L, **kw))} for L in (0, 2, 4, 8)}
        res[run] = r
        print(run, json.dumps({k: (v if k.startswith("oracle") else {kk: vv for kk, vv in v.items()}) for k, v in r.items()}, default=str), flush=True)
    json.dump(res, open(RES / "exp" / "topk.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
