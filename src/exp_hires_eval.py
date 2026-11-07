"""Evaluate the high-res rescoring (exp_hires.py): new score = a*old + (1-a)*hires(r); Viterbi tuned on valid (r, a), test once. -> results/exp/hires.json"""
import itertools, json, pickle
import numpy as np
import sn_lib as S
from sm_data import ROOT
from exp_topk import viterbi, objective, oracle

RES = ROOT / "results" / "exp"


def rescore(c, h, a):
    return [sorted([(x, y, a * s + (1 - a) * n) for (x, y, s), n in zip(cf, hf)], key=lambda t: -t[2]) for cf, hf in zip(c, h)]


def main():
    va, te = S.load("valid"), S.load("test"); cd = pickle.load(open(RES / "cands_ball_r18.pkl", "rb")); res = {}
    H = {(sp, r): pickle.load(open(RES / f"hires_{sp}_{r}.pkl", "rb")) for sp in ("valid", "test") for r in (0.5, 0.75, 1.0)}
    kw0 = dict(thr=0.1, lam=1.0, miss=0.2, hold=4); sc = []
    for r, a in itertools.product((0.5, 0.75, 1.0), (0.0, 0.3, 0.5, 0.7)):
        vc = S.filter_static(rescore(cd["valid"], H[("valid", r)], a), va["seq"])
        sc.append((objective(va, viterbi(vc, va["seq"], **kw0)), r, a)); print(sc[-1], flush=True)
    print("baseline", objective(va, viterbi(S.filter_static(cd["valid"], va["seq"]), va["seq"], **kw0)))
    b = max(sc); r, a = b[1:]; res["params"] = dict(r=r, a=a)
    tc = S.filter_static(rescore(cd["test"], H[("test", r)], a), te["seq"])
    res["oracle_top1_test"] = oracle(te, tc, ks=(1, 3, 10)); res["oracle_top1_test_baseline"] = oracle(te, S.filter_static(cd["test"], te["seq"]), ks=(1, 3, 10))
    res["test"] = {n: S.ball_hit_rate(te, viterbi(tc, te["seq"], lag=l, **kw0)) for n, l in (("offline", None), ("lag4", 4), ("causal", 0))}
    print(json.dumps(res, default=str)); json.dump(res, open(RES / "hires.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
