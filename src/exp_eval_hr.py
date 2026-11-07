"""Evaluate a BallHR run (results/<run>/best.pt): top-10 candidates, oracle, greedy, Viterbi (tuned on the valid split, test once).
Usage: exp_eval_hr.py <run> [valid_split=valid3] -> results/exp/net_<run>.json. With a motion head also tunes the motion weight mw of the Viterbi link cost."""
import itertools, json, pickle, sys, time
import numpy as np, torch
import sn_lib as S
from sm_data import ROOT
import ball_hr as B
from exp_topk import viterbi, objective, oracle

RES = ROOT / "results"
OLD = dict(temporal=False, motion=False, out_stride=2, scale_jit=False)  # hr_a checkpoint has no stored args


def attach(cands, extra, raw):
    """Re-attach predicted previous positions after filter_static (matched on the unchanged (cx, cy))."""
    out = []
    for cs, ex, rc in zip(cands, extra, raw):
        m = {(c[0], c[1]): e for c, e in zip(rc, ex)}
        out.append([(*c, *m[(c[0], c[1])]) if m.get((c[0], c[1])) else c for c in cs])
    return out


def main(run, vs="valid3"):
    va, te = S.load(vs), S.load("test"); p = RES / "exp" / f"cands_{run}.pkl"
    ck = torch.load(RES / run / "best.pt"); a = ck.get("args", OLD if "prior" not in ck else dict(OLD, baseline=True))
    if not p.exists() and a.get("baseline"):  # BallNet from train_ball2.py
        m = S.BallNet(pretrained=False).cuda(); m.load_state_dict(ck["state"]); m.eval()
        pickle.dump({sp: ([*S.ball_candidates(m, d["names"], sp if sp == "test" else vs, topk=10)], None) for sp, d in (("valid", va), ("test", te))}, open(p, "wb"))
    if not p.exists():
        m = B.build_model(a, pretrained=False).cuda(); m.load_state_dict(ck["state"]); m.eval()
        x = torch.zeros(1, 5 if a["temporal"] else 3, 504, 896).cuda()
        with torch.no_grad():
            for _ in range(3): m(x)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(20): m(x)
            torch.cuda.synchronize(); print("gpu ms/frame", (time.time() - t) / 20 * 1000)
        pickle.dump({sp: B.candidates(m, a, d, sp if sp == "test" else vs, topk=10) for sp, d in (("valid", va), ("test", te))}, open(p, "wb"))
    c = pickle.load(open(p, "rb"))
    vc = S.filter_static(c["valid"][0], va["seq"]); tc = S.filter_static(c["test"][0], te["seq"])
    if a["motion"]:
        vc, tc = attach(vc, c["valid"][1], c["valid"][0]), attach(tc, c["test"][1], c["test"][0])
    mws = [0.0, 0.5, 1.0] if a["motion"] else [0.0]
    grid = list(itertools.product([0.05, 0.1, 0.2], [0.3, 0.6, 1.0], [0.1, 0.2, 0.3], [4, 8, 16], mws))
    sc = [(objective(va, viterbi(vc, va["seq"], thr=t_, lam=b, miss=m_, hold=h, mw=w)), t_, b, m_, h, w) for t_, b, m_, h, w in grid]
    bv = max(sc); kw = dict(thr=bv[1], lam=bv[2], miss=bv[3], hold=bv[4], mw=bv[5])
    r = {"args": a, "params": kw, "oracle_test": oracle(te, tc), "valid": S.ball_hit_rate(va, viterbi(vc, va["seq"], **kw)), "valid_objective": bv[0]}
    r["greedy_test"] = S.ball_hit_rate(te, S.track_ball(tc, te["seq"], thr=0.1, hold=16))
    for n, l in (("offline", None), ("lag4", 4), ("causal", 0)):
        r[n] = S.ball_hit_rate(te, viterbi(tc, te["seq"], lag=l, **kw))
    if a["motion"]:
        r["offline_mw0"] = S.ball_hit_rate(te, viterbi(tc, te["seq"], **{**kw, "mw": 0.0}))
    print(json.dumps(r, default=str)); json.dump(r, open(RES / "exp" / f"net_{run}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "valid3")
