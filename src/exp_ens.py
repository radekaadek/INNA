"""Candidate-level ensemble of cached ball networks + Viterbi; tuned on valid, test reported once per ensemble. -> results/exp/ens.json"""
import itertools, json, pickle
import numpy as np
import sn_lib as S
from exp_topk import viterbi, objective, oracle
from sm_data import ROOT

RES = ROOT / "results"


def fuse(lists, radius=0.012):
    """lists: per model, per frame [(x,y,s)]. Greedy clustering across models; score = sum of member scores / n_models."""
    n = len(lists); out = []
    for fr in zip(*lists):
        pool = sorted([(s, x, y, m) for m, cs in enumerate(fr) for x, y, s in cs], reverse=True); used = [False] * len(pool); res = []
        for i, (s, x, y, m) in enumerate(pool):
            if used[i]: continue
            tot = {m: s}; used[i] = True
            for j in range(i + 1, len(pool)):
                s2, x2, y2, m2 = pool[j]
                if not used[j] and m2 not in tot and np.hypot(x2 - x, (y2 - y) * 0.5625) <= radius:
                    tot[m2] = s2; used[j] = True
            res.append((x, y, sum(tot.values()) / n))
        out.append(sorted(res, key=lambda c: -c[2])[:10])
    return out


def main():
    va, te = S.load("valid"), S.load("test"); C = {r: pickle.load(open(RES / "exp" / f"cands_{r}.pkl", "rb")) for r in ["ball_r18", "fb_tiny", "fb_r18", "fb_r18_v2"]}
    res = {}
    for name, runs in {"r18+v2": ["ball_r18", "fb_r18_v2"], "r18+fb_r18+v2": ["ball_r18", "fb_r18", "fb_r18_v2"], "all4": list(C)}.items():
        vc = S.filter_static(fuse([C[r]["valid"] for r in runs]), va["seq"]); tc = S.filter_static(fuse([C[r]["test"] for r in runs]), te["seq"])
        grid = list(itertools.product([0.05, 0.1, 0.2], [0.3, 0.6, 1.0], [0.1, 0.2, 0.3], [4, 8, 16]))
        bv = max((objective(va, viterbi(vc, va["seq"], thr=a, lam=b, miss=m, hold=h)), a, b, m, h) for a, b, m, h in grid)
        kw = dict(thr=bv[1], lam=bv[2], miss=bv[3], hold=bv[4])
        res[name] = {"oracle_valid": oracle(va, vc), "oracle_test": oracle(te, tc), "params": kw,
                     "valid": S.ball_hit_rate(va, viterbi(vc, va["seq"], **kw)), "test": S.ball_hit_rate(te, viterbi(tc, te["seq"], **kw)),
                     "test_causal": S.ball_hit_rate(te, viterbi(tc, te["seq"], lag=0, **kw))}
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(RES / "exp" / "ens.json", "w"), indent=1)


if __name__ == "__main__":
    main()
