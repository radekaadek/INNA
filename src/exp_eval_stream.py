"""Evaluate a StreamBall run (results/<run>/best.pt): top-10 candidates, oracle, greedy, Viterbi (tuned on the valid split, test once).
Usage: exp_eval_stream.py <run> [valid_split=valid3] -> results/exp/net_<run>.json.
Candidates are indexed by the TARGET sample (next mode: made from the two previous samples; the first sample of a clip has none), so the usual
ball-hit metrics apply directly. In next mode also reports persistence / constant-velocity baselines built from the ball_r18 top-1 (forecast of sample k from k-1, k-2)."""
import itertools, json, pickle, sys, time
import numpy as np, torch
import sn_lib as S
from sm_data import ROOT
import streamball as B
from exp_topk import viterbi, objective, oracle

RES = ROOT / "results"


def forecast_baselines(te, tc):
    """Raw ball_r18 top-1 (score >= 0.3) at k-1 as the estimate for k (persistence) and 2*p(k-1) - p(k-2) (constant velocity)."""
    top = np.array([[c[0][0], c[0][1], c[0][2]] if c else [0, 0, 0] for c in tc]); seq = te["seq"]; N = len(seq)
    pers, cv = np.zeros((N, 3)), np.zeros((N, 3))
    for k in range(1, N):
        if seq[k - 1] == seq[k] and top[k - 1, 2] >= 0.3:
            pers[k] = top[k - 1]
            if k > 1 and seq[k - 2] == seq[k] and top[k - 2, 2] >= 0.3:
                cv[k] = [2 * top[k - 1, 0] - top[k - 2, 0], 2 * top[k - 1, 1] - top[k - 2, 1], 1.0]
    return {"persistence": S.ball_hit_rate(te, pers), "const_velocity": S.ball_hit_rate(te, cv)}


def main(run, vs="valid3"):
    va, te = S.load(vs), S.load("test"); p = RES / "exp" / f"cands_{run}.pkl"
    ck = torch.load(RES / run / "best.pt"); a = ck["args"]
    import streamball2 as B2
    m = (B2 if a.get('v2') else B).build_model(a, pretrained=False).cuda(); m.load_state_dict(ck["state"]); m.eval()
    r = {"args": a, "params_M": sum(q.numel() for q in m.parameters()) / 1e6}
    if not p.exists():
        x = torch.zeros(1, 3, 504, 896).cuda()
        with torch.no_grad():
            for _ in range(3): m(x, x)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(20): m(x, x)
            torch.cuda.synchronize(); r["gpu_ms_pair"] = (time.time() - t) / 20 * 1000; print("gpu ms/pair", r["gpu_ms_pair"])
        pickle.dump({sp: B.candidates(m, a, d, sp if sp == "test" else vs, topk=10) for sp, d in (("valid", va), ("test", te))}, open(p, "wb"))
    c = pickle.load(open(p, "rb"))
    vc = S.filter_static(c["valid"], va["seq"]); tc = S.filter_static(c["test"], te["seq"])
    grid = list(itertools.product([0.05, 0.1, 0.2], [0.3, 0.6, 1.0], [0.1, 0.2, 0.3], [4, 8, 16]))
    sc = [(objective(va, viterbi(vc, va["seq"], thr=t_, lam=b, miss=m_, hold=h)), t_, b, m_, h) for t_, b, m_, h in grid]
    bv = max(sc); kw = dict(thr=bv[1], lam=bv[2], miss=bv[3], hold=bv[4])
    r.update({"params": kw, "oracle_test": oracle(te, tc), "valid": S.ball_hit_rate(va, viterbi(vc, va["seq"], **kw)), "valid_objective": bv[0]})
    r["greedy_test"] = S.ball_hit_rate(te, S.track_ball(tc, te["seq"], thr=0.1, hold=16))
    for n, l in (("offline", None), ("lag4", 4), ("causal", 0)):
        r[n] = S.ball_hit_rate(te, viterbi(tc, te["seq"], lag=l, **kw))
    if a["target"] == "next":
        r["baselines_ball_r18"] = forecast_baselines(te, pickle.load(open(RES / "exp" / "cands_ball_r18.pkl", "rb"))["test"])
    print(json.dumps(r, default=str)); json.dump(r, open(RES / "exp" / f"net_{run}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "valid3")
