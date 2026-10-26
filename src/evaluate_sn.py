"""Ball + player crop policies on SN-Tracking-2023 test (49 clips, stride 3) -> results/sn_eval.json.
Tracker / smoothing parameters are chosen on the validation clips only."""
import itertools
import json
import time

import numpy as np
import torch

import sm_lib as L
import sn_lib as S
import sm_data as D
import yolo_sn
from sm_data import BINS, density

RES = D.ROOT / "results"
SM_PLAYER_RUN, SN_PLAYER_RUN = "r18_hist", "players_sn_r18"
POSTS = {**L.POST, "ema_fast": dict(alpha=0.6, max_step=0.05)}


def load_cropnet(run):
    ck = torch.load(RES / run / "best.pt", map_location="cuda"); c = ck["cfg"]
    m = L.CropNet(c["backbone"], c["head"], c["two_frame"], pretrained=False).cuda()
    m(L.prep(np.zeros((2, 252, 448, 3), np.uint8), None, False)); m.load_state_dict(ck["state"]); return m.eval()


@torch.no_grad()
def player_hists(model, d, bs=32):
    out = []
    for i in range(0, len(d["images"]), bs):
        out.append(model(L.prep(d["images"][i:i + bs], None, False)).float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def yolo_outputs(weights, split, names, conf_ball=0.05, conf_player=0.25):
    from ultralytics import YOLO
    m = YOLO(weights); cands, hists = [], []
    paths = [str(yolo_sn.YD / split / "images" / n) for n in names]
    for i in range(0, len(paths), 32):
        for r in m.predict(paths[i:i + 32], imgsz=896, conf=min(conf_ball, conf_player), verbose=False, device=0):
            b, c, s = r.boxes.xyxyn.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()
            pl = b[(c == 0) & (s >= conf_player)]
            hists.append(density((pl[:, 0] + pl[:, 2]) / 2) if len(pl) else np.zeros(BINS, np.float32))
            bl = [((x[0] + x[2]) / 2, (x[1] + x[3]) / 2, float(sc)) for x, sc in zip(b[c == 1], s[c == 1])]
            cands.append(sorted(bl, key=lambda t: -t[2])[:5])
    return cands, np.stack(hists), m


def latency(fn, n=40, warm=8, cuda=True):
    ts = []
    for _ in range(n + warm):
        t = time.perf_counter(); fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts[warm:]) * 1000)


def tune_tracker(vd, cands):
    cands = S.filter_static(cands, vd["seq"])
    best = None
    for thr, hold in itertools.product([0.1, 0.2, 0.3, 0.5], [4, 8, 16]):
        est = S.track_ball(cands, vd["seq"], thr=thr, hold=hold)
        c = np.where(est[:, 2] > 0, est[:, 0], 0.5)
        s = S.ball_hit_rate(vd, est)["ball_recall"] + 0.5 * S.score_sn(vd, c, "raw")["ball_in_crop"]
        if best is None or s > best[0]:
            best = (s, thr, hold)
    return best[1], best[2]


def evaluate_policies(d, ball_est, ball_est_raw, hists, gt_hist):
    """Return {policy: {post: metrics}} for one (ball source, player source)."""
    rows = {}
    pc = np.array([L.center_from_hist(h) for h in hists])
    rows["players_only"] = pc
    rows["ball_only_tracked"] = np.where(ball_est[:, 2] > 0, ball_est[:, 0], 0.5)
    rows["fused_raw_ball"] = S.policy_centers(hists, ball_est_raw[:, :2], ball_est_raw[:, 2])
    rows["fused_tracked_ball"] = S.policy_centers(hists, ball_est[:, :2], ball_est[:, 2])
    return rows


def metrics_table(d, centers):
    return {p: S.score_sn(d, centers, POSTS[p]) for p in POSTS}


def main():
    te, va = S.load_small("test"), S.load_small("valid")
    res = {"n_frames": len(te["seq"]), "n_clips": int(len(te["seqs"])), "ball_visible": float((te["ball"][:, 0] >= 0).mean()), "baselines": {}, "systems": {}}
    # ---- baselines (ground truth) ----
    bx = np.where(te["ball"][:, 0] >= 0, (te["ball"][:, 0] + te["ball"][:, 2]) / 2, 0.5)
    cxp = (te["boxes"][..., 0] + te["boxes"][..., 2]) / 2; valid = te["boxes"][..., 0] >= 0
    med = np.array([np.median(cxp[i][valid[i]]) if valid[i].any() else 0.5 for i in range(len(cxp))])
    gt_est = np.concatenate([np.stack([bx, np.zeros(len(bx))], 1), (te["ball"][:, 0] >= 0)[:, None].astype(float)], 1)
    gt_hist = te["density"]
    for k, c in {"fixed_centre": np.full(len(bx), 0.5), "gt_player_median": med, "gt_ball_only": bx, "oracle_ball_players": te["center"],
                 "gt_ball_plus_pred_players_sn": S.policy_centers(player_hists(load_cropnet(SN_PLAYER_RUN), te), gt_est[:, :2], gt_est[:, 2])}.items():
        res["baselines"][k] = metrics_table(te, c)
    # ---- player modules ----
    sm_h = player_hists(load_cropnet(SM_PLAYER_RUN), te); sn_h = player_hists(load_cropnet(SN_PLAYER_RUN), te)
    sm_hv = player_hists(load_cropnet(SM_PLAYER_RUN), va); sn_hv = player_hists(load_cropnet(SN_PLAYER_RUN), va)
    # ---- ball sources ----
    srcs = {}
    bn = S.BallNet(pretrained=False).cuda(); bn.load_state_dict(torch.load(RES / "ball_r18" / "best.pt")["state"]); bn.eval()
    srcs["ball_r18"] = (S.ball_candidates(bn, va["names"], "valid"), S.ball_candidates(bn, te["names"], "test"))
    yw = str(RES / "yolo11n_sn" / "train" / "weights" / "best.pt")
    yv_c, yv_h, ym = yolo_outputs(yw, "valid", va["names"]); yt_c, yt_h, _ = yolo_outputs(yw, "test", te["names"])
    srcs["ball_yolo11n"] = (yv_c, yt_c)
    # ---- evaluate ----
    for bname, (vc, tc) in srcs.items():
        thr, hold = tune_tracker(va, vc)
        est_ns = S.track_ball(tc, te["seq"], thr=thr, hold=hold)  # without static-overlay suppression (ablation)
        tc = S.filter_static(tc, te["seq"])
        est = S.track_ball(tc, te["seq"], thr=thr, hold=hold)
        est_raw = np.array([[c[0][0], c[0][1], c[0][2]] if c and c[0][2] >= thr else [0, 0, 0] for c in tc])
        info = {"tracker": {"thr": thr, "hold": hold}, "ball_raw": S.ball_hit_rate(te, est_raw), "ball_tracked": S.ball_hit_rate(te, est),
                "ball_tracked_no_static_filter": S.ball_hit_rate(te, est_ns)}
        print(bname, info, flush=True)
        for pname, hists in {"players_sm": sm_h, "players_sn": sn_h, "players_yolo": yt_h}.items():
            if pname == "players_yolo" and bname != "ball_yolo11n":
                continue
            rows = evaluate_policies(te, est, est_raw, hists, gt_hist)
            for pol, c in rows.items():
                if pol in ("ball_only_tracked", "fused_raw_ball", "fused_tracked_ball") or pname != "players_sm":
                    res["systems"][f"{bname}|{pname}|{pol}"] = {**info, "scores": metrics_table(te, c)}
    # ---- latency ----
    x = torch.rand(1, 3, S.H, S.W).cuda(); res["latency"] = {}
    with torch.no_grad():
        res["latency"]["ball_r18_gpu_ms"] = latency(lambda: bn(x))
        bn.cpu(); torch.set_num_threads(4); xc = x.cpu(); res["latency"]["ball_r18_cpu_ms"] = latency(lambda: bn(xc), n=15, warm=3, cuda=False)
    img = np.random.randint(0, 255, (504, 896, 3), np.uint8)
    for dev, key in ((0, "gpu"), ("cpu", "cpu")):
        res["latency"][f"yolo11n_{key}_ms"] = latency(lambda: ym.predict(img, imgsz=896, verbose=False, device=dev), n=15 if dev == "cpu" else 40,
                                                       warm=3 if dev == "cpu" else 8, cuda=dev != "cpu")
    json.dump(res, open(RES / "sn_eval.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
