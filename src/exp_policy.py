"""Crop-level effect of the Viterbi tracker (ball_r18 candidates, params tuned on valid in exp_topk) -> results/exp/policy.json"""
import json, pickle
import numpy as np
import sn_lib as S
from evaluate_sn import evaluate_policies, load_cropnet, metrics_table, player_hists, SN_PLAYER_RUN
from exp_topk import viterbi
from sm_data import ROOT

RES = ROOT / "results"


def main():
    te = S.load_small("test"); C = pickle.load(open(RES / "exp" / "cands_ball_r18.pkl", "rb"))["test"]
    tc = S.filter_static(C, te["seq"]); h = player_hists(load_cropnet(SN_PLAYER_RUN), te)
    kw = json.load(open(RES / "exp" / "topk_offline.json"))["ball_r18"]["viterbi"]["params"]
    greedy = S.track_ball(tc, te["seq"], thr=0.1, hold=16)
    raw = np.array([[c[0][0], c[0][1], c[0][2]] if c and c[0][2] >= 0.1 else [0, 0, 0] for c in tc])
    out = {}
    for name, est in {"greedy": greedy, "viterbi_offline": viterbi(tc, te["seq"], **kw), "viterbi_lag4": viterbi(tc, te["seq"], lag=4, **kw), "viterbi_causal": viterbi(tc, te["seq"], lag=0, **kw)}.items():
        rows = evaluate_policies(te, est, raw, h, te["density"])
        out[name] = {p: metrics_table(te, rows[p])["ema"] for p in ("ball_only_tracked", "fused_tracked_ball")}
        out[name]["ball"] = S.ball_hit_rate(te, est)
        print(name, {p: (round(v["ball_in_crop"], 3), round(v["players_in_crop"], 3)) for p, v in out[name].items() if p != "ball"}, flush=True)
    json.dump(out, open(RES / "exp" / "policy.json", "w"), indent=1)


if __name__ == "__main__":
    main()
