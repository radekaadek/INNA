"""#19: ByteTrack-style (two-stage score split) single-ball association on cached candidates, optionally with a constant-velocity predictor
(OC-SORT/BoT-SORT-like motion model). Centre-distance cost (box IoU is meaningless for 5-8 px balls). Causal by construction.
Usage: exp_mot.py <cands_run> [sanity]  -> results/exp/mot_<run>.json   (tuned on valid, test once)"""
import itertools, json, pickle, sys
import numpy as np
import sn_lib as S
from sm_data import ROOT
from exp_topk import viterbi, objective, oracle

RES = ROOT / "results"
AR = 0.5625


def dist(a, b):
    return float(np.hypot(a[0] - b[0], (a[1] - b[1]) * AR))


def bytetrack(cands, seq, hi=0.3, lo=0.05, gate=0.2, hold=8, kf=False, vk=0.5, grow=4.0, topk=10):
    """Per clip: one track. Stage 1 matches the best (score - dist/gate) HIGH detection inside the gate, stage 2 the nearest LOW one.
    Gate grows with samples since the last match (1 + age/grow). Track dies after `hold` unmatched samples; a new track starts
    from the top-scoring high detection. Returns (N,3) x, y, conf (0 = no estimate); unmatched-but-alive samples output the predicted position."""
    out = np.zeros((len(cands), 3))
    for s in np.unique(seq):
        pos, vel, age = None, np.zeros(2), 0
        for i in np.flatnonzero(seq == s):
            dets = [c for c in cands[i][:topk] if c[2] >= lo]
            highs = [c for c in dets if c[2] >= hi]; lows = [c for c in dets if c[2] < hi]
            if pos is not None:
                pred = np.array(pos) + (vel * (age + 1) if kf else 0)
                g = gate * (1 + age / grow)
                m = None
                for pool, sel in ((highs, lambda c, d: c[2] - d / g), (lows, lambda c, d: -d)):
                    best = None
                    for c in pool:
                        d = dist(c, pred)
                        if d <= g and (best is None or sel(c, d) > best[0]):
                            best = (sel(c, d), c)
                    if best:
                        m = best[1]; break
                if m is not None:
                    new = np.array(m[:2])
                    if kf:
                        vel = vk * vel + (1 - vk) * (new - np.array(pos)) / (age + 1)
                    pos, age = (m[0], m[1]), 0
                    out[i] = (m[0], m[1], m[2])
                    continue
                age += 1
                if age <= hold:
                    out[i] = (pred[0], pred[1], 1e-3)
                    pos = (pred[0], pred[1]) if kf else pos
                    continue
                pos = None  # track lost
            if highs:
                c = max(highs, key=lambda c: c[2])
                pos, vel, age = (c[0], c[1]), np.zeros(2), 0
                out[i] = c
    return out


def main(run, sanity=False):
    va, te = S.load("valid"), S.load("test")
    c = pickle.load(open(RES / "exp" / f"cands_{run}.pkl", "rb"))
    vc = S.filter_static(c["valid"], va["seq"]); tc = S.filter_static(c["test"], te["seq"])
    if sanity:  # lo=hi, huge gate, no motion model: should be close to greedy
        e = bytetrack(vc, va["seq"], hi=0.1, lo=0.1, gate=0.25, hold=16)
        print("sanity (valid) bytetrack~greedy", S.ball_hit_rate(va, e), "greedy", S.ball_hit_rate(va, S.track_ball(vc, va["seq"], thr=0.1, hold=16))); return
    grid = list(itertools.product([0.15, 0.3, 0.5], [0.05, 0.1], [0.1, 0.2, 0.3], [4, 8, 16], [False, True]))
    sc = [(objective(va, bytetrack(vc, va["seq"], hi=h, lo=l, gate=g, hold=o, kf=k)), h, l, g, o, k) for h, l, g, o, k in grid]
    res = {"valid_top5": sorted(sc, reverse=True)[:5]}
    for name, ks in (("any", (False, True)), ("no_kf", (False,)), ("kf", (True,))):
        b = max(x for x in sc if x[5] in ks); kw = dict(hi=b[1], lo=b[2], gate=b[3], hold=b[4], kf=b[5])
        res[name] = {"params": kw, "valid": S.ball_hit_rate(va, bytetrack(vc, va["seq"], **kw)),
                     "test": S.ball_hit_rate(te, bytetrack(tc, te["seq"], **kw))}
    res["greedy_test"] = S.ball_hit_rate(te, S.track_ball(tc, te["seq"], thr=0.1, hold=16))
    res["oracle_test"] = oracle(te, tc)
    print(json.dumps(res, default=str, indent=1)); json.dump(res, open(RES / "exp" / f"mot_{run}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == "sanity")
