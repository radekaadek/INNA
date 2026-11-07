"""Backlog #10 (light version): learned per-candidate re-scorer fusing cached signals (heatmap score/rank, 1080p re-score at r 0.5/0.75, YOLO sports-ball and person evidence,
temporal support from neighbouring samples, position). Small MLP trained on the 6 validation clips (leave-one-clip-out predictions are used to tune Viterbi on valid), applied to test.
Offline features (use both neighbouring samples). -> results/exp/rerank.json"""
import itertools, json, pickle
import numpy as np, torch
import sn_lib as S
from sm_data import ROOT
from exp_topk import viterbi, objective, oracle, gt_xy
from exp_players import box_dist

RES = ROOT / "results" / "exp"; K = 10


def feats(split, d):
    c = pickle.load(open(RES / "cands_ball_r18.pkl", "rb"))[split]; c = S.filter_static(c, d["seq"])
    # filter_static drops candidates, so align hires scores by position: recompute from the unfiltered list
    raw = pickle.load(open(RES / "cands_ball_r18.pkl", "rb"))[split]
    h5, h75 = (pickle.load(open(RES / f"hires_{split}_{r}.pkl", "rb")) for r in (0.5, 0.75))
    Y = pickle.load(open(RES / "cands_yolo.pkl", "rb"))[split]; P = pickle.load(open(RES / "cands_person.pkl", "rb"))[split]
    keep = [[tuple(q) in {tuple(z) for z in cf} for q in rf] for cf, rf in zip(c, raw)]
    X, meta = [], []
    for i, rf in enumerate(raw):
        for k, (x, y, s) in enumerate(rf):
            if not keep[i][k]:
                continue
            sup = []
            for j in (i - 1, i + 1):
                ok = 0 <= j < len(raw) and d["seq"][j] == d["seq"][i]
                sup.append(max([sc * np.exp(-np.hypot(x - a, (y - b) * 0.5625) / 0.05) for a, b, sc in raw[j]] + [0]) if ok else 0.0)
            yn = max([sc for a, b, sc in Y[i] if np.hypot(x - a, (y - b) * 0.5625) < 0.02] + [0])
            X.append([s, np.log(s + 1e-3), k / K, h5[i][k], h75[i][k], yn, min(box_dist(x, y, P[i]), 0.2), sup[0], sup[1], x, y]); meta.append((i, x, y, s))
    return np.array(X, np.float32), meta


def labels(d, meta):
    gx, gy = gt_xy(d); vis = d["ball"][:, 0] >= 0
    return np.array([float(vis[i] and np.hypot(x - gx[i], (y - gy[i]) * 0.5625) <= 0.02) for i, x, y, s in meta], np.float32)


def fit(X, y, mu, sd, seed=0):
    torch.manual_seed(seed); net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), 1e-2, weight_decay=1e-2); xt, yt = torch.from_numpy((X - mu) / sd), torch.from_numpy(y)
    for _ in range(300):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt)[:, 0], yt); opt.zero_grad(); loss.backward(); opt.step()
    return net


def predict(net, X, mu, sd):
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy((X - mu) / sd))[:, 0]).numpy()


def to_cands(n, meta, p):
    out = [[] for _ in range(n)]
    for (i, x, y, s), q in zip(meta, p):
        out[i].append((x, y, float(q)))
    return [sorted(o, key=lambda t: -t[2]) for o in out]


def main():
    va, te = S.load("valid"), S.load("test"); Xv, mv = feats("valid", va); Xt, mt = feats("test", te); yv = labels(va, mv)
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-6; seqv = np.array([va["seq"][i] for i, *_ in mv])
    pv = np.zeros(len(yv), np.float32)
    for s in np.unique(seqv):  # leave-one-clip-out predictions on valid
        m = seqv == s; pv[m] = predict(fit(Xv[~m], yv[~m], mu, sd), Xv[m], mu, sd)
    net = fit(Xv, yv, mu, sd); pt = predict(net, Xt, mu, sd)
    vc, tc = to_cands(len(va["seq"]), mv, pv), to_cands(len(te["seq"]), mt, pt)
    res = {"oracle_valid_loco": oracle(va, vc, ks=(1, 3, 10), min_sc=0.0), "oracle_test": oracle(te, tc, ks=(1, 3, 10), min_sc=0.0), "positives_valid": int(yv.sum())}
    print(res, flush=True)
    grid = list(itertools.product([0.005, 0.02, 0.05, 0.1], [0.3, 0.6, 1.0], [0.03, 0.1, 0.2, 0.3], [4, 8]))
    sc = [(objective(va, viterbi(vc, va["seq"], thr=a, lam=b, miss=m_, hold=h)), a, b, m_, h) for a, b, m_, h in grid]
    bv = max(sc); kw = dict(thr=bv[1], lam=bv[2], miss=bv[3], hold=bv[4]); res["params"] = kw; res["valid_loco"] = S.ball_hit_rate(va, viterbi(vc, va["seq"], **kw))
    for n, l in (("offline", None), ("lag4", 4), ("causal", 0)):
        res[n] = S.ball_hit_rate(te, viterbi(tc, te["seq"], lag=l, **kw))
    res["top1_raw_test"] = S.ball_hit_rate(te, np.array([[c[0][0], c[0][1], c[0][2]] if c else [0, 0, 0] for c in tc]))
    print(json.dumps(res, default=str)); json.dump(res, open(RES / "rerank.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
