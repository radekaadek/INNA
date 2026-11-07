"""Evaluate b2_e20 (control) and b2_prior: cache top-10 candidates (final EMA weights, causal prior for b2_prior), raw top-1, greedy and Viterbi tuned on valid; test once.
-> results/exp/b2.json"""
import itertools, json, pickle, sys
import numpy as np, torch
import sn_lib as S
from exp_topk import viterbi, objective, oracle
from train_ball2 import prior_candidates
from sm_data import ROOT

RES = ROOT / "results"


def main():
    va, te = S.load("valid"), S.load("test"); res = {}
    for run, wt in [("b2_e20", "final_ema.pt"), ("b2_prior", "final_ema.pt")]:
        ck = torch.load(RES / run / wt); pr = ck["prior"]
        m = S.BallNet(pretrained=False, in_ch=4 if pr else 3).cuda(); m.load_state_dict(ck["state"]); m.eval()
        f = RES / "exp" / f"cands_{run}.pkl"
        if f.exists():
            c = pickle.load(open(f, "rb"))
        else:
            c = {sp: prior_candidates(m, d, sp, use_prior=pr) for sp, d in (("valid", va), ("test", te))}; pickle.dump(c, open(f, "wb"))
        vc = S.filter_static(c["valid"], va["seq"]); tc = S.filter_static(c["test"], te["seq"])
        raw = lambda d, cc: S.ball_hit_rate(d, np.array([[x[0][0], x[0][1], x[0][2]] if x else [0, 0, 0] for x in cc]))
        best = max(((objective(va, S.track_ball(vc, va["seq"], thr=t, hold=h)), t, h) for t, h in itertools.product([0.1, 0.2, 0.3, 0.5], [4, 8, 16])))
        grid = list(itertools.product([0.05, 0.1, 0.2], [0.3, 0.6, 1.0], [0.1, 0.2, 0.3], [4, 8, 16]))
        bv = max((objective(va, viterbi(vc, va["seq"], thr=a, lam=b, miss=mm, hold=h)), a, b, mm, h) for a, b, mm, h in grid)
        kw = dict(thr=bv[1], lam=bv[2], miss=bv[3], hold=bv[4])
        res[run] = {"oracle_valid": oracle(va, vc), "oracle_test": oracle(te, tc), "raw_valid": raw(va, vc), "raw_test": raw(te, tc),
                    "greedy": {"params": best[1:], "test": S.ball_hit_rate(te, S.track_ball(tc, te["seq"], thr=best[1], hold=best[2]))},
                    "viterbi": {"params": kw, "valid": S.ball_hit_rate(va, viterbi(vc, va["seq"], **kw)), "test": S.ball_hit_rate(te, viterbi(tc, te["seq"], **kw)),
                                "test_causal": S.ball_hit_rate(te, viterbi(tc, te["seq"], lag=0, **kw)), "test_lag4": S.ball_hit_rate(te, viterbi(tc, te["seq"], lag=4, **kw))}}
        print(run, json.dumps(res[run]), flush=True)
    json.dump(res, open(RES / "exp" / "b2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
