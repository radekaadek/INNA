"""Error analysis of ball_r18 + offline Viterbi (cached candidates, CPU). Descriptive only, nothing is tuned on it. -> results/exp/errors.json"""
import json, pickle
import numpy as np
import sn_lib as S
from exp_topk import viterbi
from sm_data import ROOT

RES = ROOT / "results"


def analyse(d, cands, est, name):
    vis = d["ball"][:, 0] >= 0; b = d["ball"]
    gx, gy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    size = np.hypot(b[:, 2] - b[:, 0], (b[:, 3] - b[:, 1]) * 0.5625) * 896  # ball diagonal in px (896 wide frame)
    dist = np.hypot(est[:, 0] - gx, (est[:, 1] - gy) * 0.5625)
    hit = (est[:, 2] > 0) & (dist <= 0.02)
    in_c = np.array([any(np.hypot(cx - gx[i], (cy - gy[i]) * 0.5625) <= 0.02 and sc >= 0.05 for cx, cy, sc in cands[i]) for i in range(len(cands))])
    has = est[:, 2] > 0
    kind = np.where(hit, "hit", np.where(~has, "no_estimate", np.where(in_c, "wrong_pick(GT in top10)", "wrong_pick(GT not a candidate)")))
    out = {"n_visible": int(vis.sum()), "outcome": {k: float((kind[vis] == k).mean()) for k in np.unique(kind[vis])}}
    # speed: GT displacement from previous sample in the same clip
    sp = np.full(len(b), np.nan)
    for k in range(1, len(b)):
        if d["seq"][k] == d["seq"][k - 1] and vis[k] and vis[k - 1]:
            sp[k] = np.hypot(gx[k] - gx[k - 1], (gy[k] - gy[k - 1]) * 0.5625) * 896
    # nearest player box (centre) distance and overlap of the ball with any player box
    bx = d["boxes"]; pv = bx[..., 0] >= 0
    near = np.full(len(b), np.nan); ovl = np.zeros(len(b), bool)
    for k in np.flatnonzero(vis):
        p = bx[k][pv[k]]
        if len(p):
            pc = np.stack([(p[:, 0] + p[:, 2]) / 2, (p[:, 1] + p[:, 3]) / 2], 1)
            near[k] = np.min(np.hypot(pc[:, 0] - gx[k], (pc[:, 1] - gy[k]) * 0.5625)) * 896
            ovl[k] = bool(((gx[k] >= p[:, 0]) & (gx[k] <= p[:, 2]) & (gy[k] >= p[:, 1]) & (gy[k] <= p[:, 3])).any())
    def rate(mask, label, store):
        m = mask & vis
        if m.sum() >= 30:
            store[label] = {"n": int(m.sum()), "hit": round(float(hit[m].mean()), 3), "in_top10": round(float(in_c[m].mean()), 3)}
    for key, vals, bins in [("ball_size_px", size, [0, 8, 11, 14, 18, 100]), ("speed_px_per_sample", sp, [0, 5, 15, 30, 60, 1e9]),
                            ("nearest_player_px", near, [0, 20, 50, 100, 200, 1e9]), ("x_position", gx, [0, .2, .4, .6, .8, 1.01]),
                            ("y_position", gy, [0, .25, .4, .55, .7, 1.01])]:
        s = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            rate((vals >= lo) & (vals < hi), f"[{lo:g},{hi:g})", s)
        out[key] = s
    s = {}; rate(ovl, "ball inside a player box", s); rate(~ovl, "ball outside all player boxes", s); out["occlusion_proxy"] = s
    # score of the chosen candidate when right/wrong, and where wrong estimates land
    out["mean_conf_hit"] = float(est[hit, 2].mean()); out["mean_conf_wrong"] = float(est[has & ~hit & vis, 2].mean())
    fp = has & ~hit
    out["false_estimates_when_ball_not_visible"] = float((has & ~vis).sum() / max((~vis).sum(), 1))
    # per clip
    pc = {}
    for s_ in np.unique(d["seq"]):
        m = (d["seq"] == s_) & vis
        if m.sum():
            pc[int(s_)] = round(float(hit[m].mean()), 3)
    v = np.array(list(pc.values())); out["per_clip_hit"] = {"min": float(v.min()), "p10": float(np.percentile(v, 10)), "median": float(np.median(v)), "max": float(v.max()), "n_clips_below_50pct": int((v < 0.5).sum()), "n_clips": len(v)}
    # runs of consecutive failures
    runs = []
    for s_ in np.unique(d["seq"]):
        idx = np.flatnonzero((d["seq"] == s_)); r = 0
        for k in idx:
            if vis[k] and not hit[k]:
                r += 1
            else:
                if r: runs.append(r)
                r = 0
        if r: runs.append(r)
    runs = np.array(runs); out["failure_runs"] = {"n_runs": len(runs), "median_len": float(np.median(runs)), "frac_failures_in_runs>=8": float(runs[runs >= 8].sum() / runs.sum())}
    return out


def main():
    res = {}
    kw = json.load(open(RES / "exp" / "topk_offline.json"))["ball_r18"]["viterbi"]["params"]
    for sp in ("valid", "test"):
        d = S.load(sp); c = S.filter_static(pickle.load(open(RES / "exp" / "cands_ball_r18.pkl", "rb"))[sp], d["seq"])
        res[sp] = analyse(d, c, viterbi(c, d["seq"], **kw), sp)
    json.dump(res, open(RES / "exp" / "errors.json", "w"), indent=1)
    print(json.dumps(res["test"], indent=1))


if __name__ == "__main__":
    main()
