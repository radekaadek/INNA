"""Viterbi variants: constant-velocity prediction in the move penalty (vk>0, backlog #3) and confidence-based dropping of the output (#7).
Tuned on validation only, test reported once. -> results/exp/vit2.json"""
import itertools, json, pickle
import numpy as np
import sn_lib as S
from sm_data import ROOT
from exp_topk import objective, viterbi

RES = ROOT / "results"


def viterbi_v(cands, seq, thr=0.1, lam=1.0, gate=0.25, miss=0.2, hold=4, topk=5, vk=0.5, vmax=0.1):
    out = np.zeros((len(cands), 3))
    for s in np.unique(seq):
        idx = np.flatnonzero(seq == s)
        sts = [[c for c in cands[i][:topk] if c[2] >= thr] for i in idx]
        score = None; back = []; arg = []; lprev = None
        for t, st in enumerate(sts):
            n = len(st) + 1
            sc = np.full(n, -1e9); bk = np.zeros(n, int); lp = [None] * n
            for j in range(n):
                gain = st[j][2] if j < len(st) else miss
                if t == 0:
                    sc[j] = gain; lp[j] = (st[j][0], st[j][1], 0, 0.0, 0.0) if j < len(st) else None; continue
                for pj in range(len(lprev)):
                    pl = lprev[pj]; pen = 0.0
                    if j < len(st) and pl is not None:
                        k = pl[2] + 1
                        px, py = pl[0] + vk * pl[3] * k, pl[1] + vk * pl[4] * k
                        d = np.hypot(st[j][0] - px, (st[j][1] - py) * 0.5625)
                        pen = lam * min(d / (gate * (1 + pl[2] / 4)), 1.0)
                    v = score[pj] + gain - pen
                    if v > sc[j]:
                        sc[j] = v; bk[j] = pj
                        if j < len(st):
                            if pl is None:
                                lp[j] = (st[j][0], st[j][1], 0, 0.0, 0.0)
                            else:
                                k = pl[2] + 1
                                vx = float(np.clip((st[j][0] - pl[0]) / k, -vmax, vmax)); vy = float(np.clip((st[j][1] - pl[1]) / k, -vmax, vmax))
                                lp[j] = (st[j][0], st[j][1], 0, vx, vy)
                        else:
                            lp[j] = None if pl is None else (pl[0], pl[1], pl[2] + 1, pl[3], pl[4])
            back.append(bk); score = sc; lprev = lp; arg.append(int(np.argmax(sc)))
        j = arg[-1]; path = []
        for t in range(len(sts) - 1, -1, -1):
            path.append(j); j = back[t][j]
        path = path[::-1]; pos = None; age = 10 ** 9
        for t, j in enumerate(path):
            i = idx[t]
            if j < len(sts[t]):
                pos = (sts[t][j][0], sts[t][j][1]); age = 0; out[i] = (*pos, sts[t][j][2])
            else:
                age += 1
                if pos is not None and age <= hold:
                    out[i] = (*pos, 0.01)
    return out


def drop(est, c):
    e = est.copy(); e[e[:, 2] < c] = 0; return e


def main():
    va, te = S.load("valid"), S.load("test")
    c = pickle.load(open(RES / "exp" / "cands_ball_r18.pkl", "rb"))
    vc = S.filter_static(c["valid"], va["seq"]); tc = S.filter_static(c["test"], te["seq"])
    base = dict(thr=0.1, lam=1.0, miss=0.2, hold=4)
    res = {"baseline": {"valid": S.ball_hit_rate(va, viterbi(vc, va["seq"], **base)), "test": S.ball_hit_rate(te, viterbi(tc, te["seq"], **base))}}
    print("baseline", res["baseline"], flush=True)
    grid = list(itertools.product([0.25, 0.5, 1.0], [0.5, 1.0], [0.2, 0.3]))  # vk, lam, gate
    sc = [(objective(va, viterbi_v(vc, va["seq"], vk=v, lam=l, gate=g, **{k: base[k] for k in ("thr", "miss", "hold")})), v, l, g) for v, l, g in grid]
    for r in sorted(sc)[::-1][:4]: print("valid obj", r, flush=True)
    b = max(sc); kw = dict(vk=b[1], lam=b[2], gate=b[3], thr=0.1, miss=0.2, hold=4)
    ev, et = viterbi_v(vc, va["seq"], **kw), viterbi_v(tc, te["seq"], **kw)
    res["velocity"] = {"params": kw, "valid": S.ball_hit_rate(va, ev), "test": S.ball_hit_rate(te, et)}
    print("velocity", res["velocity"], flush=True)
    # confidence dropping on the baseline viterbi output
    eb, etb = viterbi(vc, va["seq"], **base), viterbi(tc, te["seq"], **base)
    res["drop"] = {}
    for cth in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6):
        res["drop"][str(cth)] = {"valid": S.ball_hit_rate(va, drop(eb, cth)), "test": S.ball_hit_rate(te, drop(etb, cth)),
                                 "valid_crop": S.score_sn(va, np.where(drop(eb, cth)[:, 2] > 0, drop(eb, cth)[:, 0], 0.5), "raw")["ball_in_crop"]}
        print("drop", cth, res["drop"][str(cth)], flush=True)
    json.dump(res, open(RES / "exp" / "vit2.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
