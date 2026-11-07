"""Evaluate a BallNet run (results/<run>/best.pt): top-10 candidates, oracle, greedy and Viterbi (tuned on valid, test once). Usage: exp_eval_net.py <run> -> results/exp/net_<run>.json"""
import itertools, json, pickle, sys
import numpy as np, torch
import sn_lib as S
from sm_data import ROOT
from exp_topk import viterbi, objective, oracle

RES = ROOT / "results"


def main(run):
    va, te = S.load("valid"), S.load("test"); p = RES / "exp" / f"cands_{run}.pkl"
    if not p.exists():
        m = S.BallNet(pretrained=False).cuda(); m.load_state_dict(torch.load(RES / run / "best.pt")["state"]); m.eval()
        pickle.dump({sp: S.ball_candidates(m, d["names"], sp, topk=10) for sp, d in (("valid", va), ("test", te))}, open(p, "wb"))
    c = pickle.load(open(p, "rb")); vc = S.filter_static(c["valid"], va["seq"]); tc = S.filter_static(c["test"], te["seq"])
    grid = list(itertools.product([0.05, 0.1, 0.2], [0.3, 0.6, 1.0], [0.1, 0.2, 0.3], [4, 8, 16]))
    sc = [(objective(va, viterbi(vc, va["seq"], thr=a, lam=b, miss=m_, hold=h)), a, b, m_, h) for a, b, m_, h in grid]
    bv = max(sc); kw = dict(thr=bv[1], lam=bv[2], miss=bv[3], hold=bv[4])
    r = {"params": kw, "oracle_test": oracle(te, tc), "valid": S.ball_hit_rate(va, viterbi(vc, va["seq"], **kw))}
    r["greedy_test"] = S.ball_hit_rate(te, S.track_ball(tc, te["seq"], thr=0.1, hold=16))
    for n, l in (("offline", None), ("lag4", 4), ("causal", 0)):
        r[n] = S.ball_hit_rate(te, viterbi(tc, te["seq"], lag=l, **kw))
    print(json.dumps(r, default=str)); json.dump(r, open(RES / "exp" / f"net_{run}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1])
