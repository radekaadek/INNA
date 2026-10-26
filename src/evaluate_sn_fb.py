"""Add the FootAndBall-style joint networks (results/fb_*/best.pt) to results/sn_eval.json.  Run after evaluate_sn.py.
Each network gives ball candidates and a player map; ball source name = run name, player source = 'players_fb'."""
import json
import time

import numpy as np
import torch

import sm_lib as L
import sn_fb as FB
import sn_lib as S
from evaluate_sn import evaluate_policies, latency, metrics_table, tune_tracker
from sm_data import ROOT

RES = ROOT / "results"
RUNS = ["fb_tiny", "fb_r18", "fb_r18_v2"]


def main():
    ev = json.load(open(RES / "sn_eval.json"))
    te, va = S.load("test"), S.load("valid")
    for run in RUNS:
        if not (RES / run / "best.pt").exists() or not (RES / run / "train_log.json").exists():
            continue
        ck = torch.load(RES / run / "best.pt", map_location="cuda"); two = ck.get("two_frame", False)
        m = FB.FBNet(ck["backbone"], pretrained=False, in_ch=6 if two else 3).cuda(); m.load_state_dict(ck["state"]); m.eval()
        vc, vh = FB.fb_outputs(m, va["names"], "valid", prev=FB.prev_names(va) if two else None)
        tc, th = FB.fb_outputs(m, te["names"], "test", prev=FB.prev_names(te) if two else None)
        thr, hold = tune_tracker(va, vc)
        tc = S.filter_static(tc, te["seq"])
        est = S.track_ball(tc, te["seq"], thr=thr, hold=hold)
        est_raw = np.array([[c[0][0], c[0][1], c[0][2]] if c and c[0][2] >= thr else [0, 0, 0] for c in tc])
        info = {"tracker": {"thr": thr, "hold": hold}, "ball_raw": S.ball_hit_rate(te, est_raw), "ball_tracked": S.ball_hit_rate(te, est),
                "params_M": sum(p.numel() for p in m.parameters()) / 1e6, "two_frame": two}
        print(run, info, flush=True)
        for pol, c in evaluate_policies(te, est, est_raw, th, te["density"]).items():
            ev["systems"][f"{run}|players_fb|{pol}"] = {**info, "scores": metrics_table(te, c)}
        x = torch.rand(1, 6 if two else 3, S.H, S.W).cuda()
        with torch.no_grad():
            ev["latency"][f"{run}_gpu_ms"] = latency(lambda: m(x))
            m.cpu(); torch.set_num_threads(4); xc = x.cpu()
            ev["latency"][f"{run}_cpu_ms"] = latency(lambda: m(xc), n=15, warm=3, cuda=False)
        print(run, {k: round(v, 1) for k, v in ev["latency"].items() if k.startswith(run)}, flush=True)
    json.dump(ev, open(RES / "sn_eval.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
