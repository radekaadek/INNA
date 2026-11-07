"""Backlog #2: re-acquisition on low confidence. In the Viterbi state, count consecutive picks with score < c; once the count reaches n the move penalty
from that track is waived (the gate opens), so the path can jump to a better candidate elsewhere. c, n tuned on valid; test once. -> results/exp/reacq.json"""
import itertools, json, pickle
import numpy as np
import sn_lib as S
from sm_data import ROOT
from exp_topk import objective, viterbi

RES = ROOT / "results"


def viterbi_r(cands, seq, thr=0.1, lam=1.0, gate=0.25, miss=0.2, hold=4, topk=5, c=0.3, n=3, lag=None):
    out = np.zeros((len(cands), 3))
    for s in np.unique(seq):
        idx = np.flatnonzero(seq == s)
        sts = [[q for q in cands[i][:topk] if q[2] >= thr] for i in idx]
        score = None; back = []; arg = []; lprev = None
        for t, st in enumerate(sts):
            m = len(st) + 1
            sc = np.full(m, -1e9); bk = np.zeros(m, int); lp = [None] * m
            for j in range(m):
                gain = st[j][2] if j < len(st) else miss
                if t == 0:
                    sc[j] = gain; lp[j] = (st[j][0], st[j][1], 0, int(st[j][2] < c)) if j < len(st) else None; continue
                for pj in range(len(lprev)):
                    pl = lprev[pj]; pen = 0.0
                    if j < len(st) and pl is not None and pl[3] < n:
                        d = np.hypot(st[j][0] - pl[0], (st[j][1] - pl[1]) * 0.5625)
                        pen = lam * min(d / (gate * (1 + pl[2] / 4)), 1.0)
                    v = score[pj] + gain - pen
                    if v > sc[j]:
                        sc[j] = v; bk[j] = pj
                        if j < len(st):
                            low = int(st[j][2] < c)
                            lp[j] = (st[j][0], st[j][1], 0, (pl[3] + 1 if (pl is not None and low) else low))
                        else:
                            lp[j] = None if pl is None else (pl[0], pl[1], pl[2] + 1, pl[3])
            back.append(bk); score = sc; lprev = lp; arg.append(int(np.argmax(sc)))
        T = len(sts)
        if lag is None:
            j = arg[-1]; path = []
            for t in range(T - 1, -1, -1):
                path.append(j); j = back[t][j]
            path = path[::-1]
        else:
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


def main():
    va, te = S.load("valid"), S.load("test")
    cd = pickle.load(open(RES / "exp" / "cands_ball_r18.pkl", "rb"))
    vc = S.filter_static(cd["valid"], va["seq"]); tc = S.filter_static(cd["test"], te["seq"])
    base = dict(thr=0.1, lam=1.0, miss=0.2, hold=4)
    sc = [(objective(va, viterbi_r(vc, va["seq"], c=c, n=n, **base)), c, n) for c, n in itertools.product([0.2, 0.3, 0.4, 0.5], [2, 4, 8])]
    for r in sorted(sc)[::-1][:5]: print("valid obj", r, flush=True)
    print("baseline valid obj", objective(va, viterbi(vc, va["seq"], **base)))
    b = max(sc); kw = dict(c=b[1], n=b[2], **base)
    res = {"params": kw, "valid": S.ball_hit_rate(va, viterbi_r(vc, va["seq"], **kw))}
    for name, lag in (("offline", None), ("lag4", 4), ("causal", 0)):
        res[name] = S.ball_hit_rate(te, viterbi_r(tc, te["seq"], lag=lag, **kw))
    print(json.dumps(res, default=str)); json.dump(res, open(RES / "exp" / "reacq.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
