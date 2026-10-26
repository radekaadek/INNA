"""results/sn_eval.json -> paper/tables/sn_policies.tex, sn_ball.tex and paper/figs/sn_*.pdf"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sm_lib as L
import sn_lib as S
import sm_data as D

RES, PAPER = D.ROOT / "results", D.ROOT / "paper"
SMOOTH = "ema_fast"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
BASE = {"fixed_centre": "Fixed centre", "gt_player_median": "Median of players (GT)", "gt_ball_only": "Ball-centred (GT ball)",
        "gt_ball_plus_pred_players_sn": "GT ball + predicted players", "oracle_ball_players": "Oracle (GT ball + players)"}
BALL = {"ball_r18": "ResNet-18 heatmap", "ball_yolo11n": "YOLO11n", "fb_tiny": "FB-style tiny (0.19M)", "fb_r18": "FB-style ResNet-18",
        "fb_r18_v2": "FB-style ResNet-18 v2 (2-frame, mined neg.)"}
PLAY = {"players_sm": "SportsMOT-trained", "players_sn": "SN-trained", "players_yolo": "YOLO11n", "players_fb": "same network"}
POL = {"players_only": "players only", "ball_only_tracked": "ball only (tracked)", "fused_raw_ball": "ball + players (no tracker)",
       "fused_tracked_ball": "ball + players (tracked)"}


def name(key):
    b, p, pol = key.split("|")
    return f"{BALL[b]} / {PLAY[p]} / {POL[pol]}"


def pct(x):
    return f"{100 * x:.1f}"


def tables(ev):
    rows = []
    for k, v in ev["baselines"].items():
        rows.append((BASE[k], v["raw"], v[SMOOTH]))
    rows.append(None)
    SEL = ["ball_r18|players_sn|players_only", "ball_r18|players_sn|ball_only_tracked", "ball_r18|players_sn|fused_raw_ball",
           "ball_r18|players_sn|fused_tracked_ball", "ball_r18|players_sm|fused_tracked_ball", "ball_yolo11n|players_yolo|ball_only_tracked",
           "ball_yolo11n|players_yolo|fused_raw_ball", "ball_yolo11n|players_yolo|fused_tracked_ball"]
    SEL += [f"{r}|players_fb|{pol}" for r in ("fb_tiny", "fb_r18", "fb_r18_v2") for pol in ("ball_only_tracked", "fused_tracked_ball")]
    sysrows = [(k, ev["systems"][k]) for k in SEL if k in ev["systems"]]
    for k, v in sysrows:
        rows.append((name(k), v["scores"]["raw"], v["scores"][SMOOTH]))
    out = []
    for r in rows:
        if r is None:
            out.append("\\midrule"); continue
        n, a, b = r
        out.append(f"{n} & {pct(a['ball_in_crop'])} & {pct(b['ball_in_crop'])} & {pct(a['players_in_crop'])} & {pct(b['players_in_crop'])} & "
                   f"{pct(a['jitter'])} & {pct(b['jitter'])} \\\\")
    (PAPER / "tables" / "sn_policies.tex").write_text("\n".join(out) + "\n\\bottomrule\n")
    lat = ev["latency"]
    rows = []
    for b in BALL:
        s = next((v for k, v in ev["systems"].items() if k.startswith(b + "|")), None)
        if s is None:
            continue
        r, t = s["ball_raw"], s["ball_tracked"]
        key = {"ball_r18": "ball_r18", "ball_yolo11n": "yolo11n"}.get(b, b)
        g, c = lat[f"{key}_gpu_ms"], lat[f"{key}_cpu_ms"]
        rows.append(f"{BALL[b]} & {g:.0f} & {c:.0f} & {pct(r['ball_recall'])} & {pct(r['ball_precision'])} & {pct(t['ball_recall'])} & "
                    f"{pct(t['ball_precision'])} & {pct(t['ball_coverage'])} \\\\")
    (PAPER / "tables" / "sn_ball.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


def fig_scatter(ev):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for k, v in ev["baselines"].items():
        s = v[SMOOTH]; ax.scatter(100 * s["players_in_crop"], 100 * s["ball_in_crop"], marker="s", c="gray", s=14)
        ax.annotate(BASE[k].split(" (")[0], (100 * s["players_in_crop"], 100 * s["ball_in_crop"]), fontsize=5, xytext=(2, 2), textcoords="offset points")
    for k, v in ev["systems"].items():
        s = v["scores"][SMOOTH]; b = k.split("|")[0]
        ax.scatter(100 * s["players_in_crop"], 100 * s["ball_in_crop"], c="C0" if b == "ball_r18" else "C1", s=14)
    ax.set_xlabel("Players in crop [%]"); ax.set_ylabel("Ball in crop [%]")
    ax.scatter([], [], c="C0", s=14, label="ResNet-18 ball"); ax.scatter([], [], c="C1", s=14, label="YOLO11n ball"); ax.scatter([], [], c="gray", marker="s", s=14, label="GT baselines")
    ax.legend(fontsize=5.5, frameon=False); fig.tight_layout(); fig.savefig(PAPER / "figs" / "sn_scatter.pdf"); plt.close(fig)


def fig_traj(ev, te, est_c, est_ball):
    s = int(np.unique(te["seq"])[3]); m = te["seq"] == s; t = te["frame"][m] / 25
    gx = np.where(te["ball"][m, 0] >= 0, (te["ball"][m, 0] + te["ball"][m, 2]) / 2, np.nan)
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.plot(t, 1920 * gx, ".", ms=2, c="gray", label="GT ball x")
    ax.plot(t, 1920 * est_c[m], c="C0", lw=1.2, label="crop centre (ball + players, tracked)")
    ax.fill_between(t, 1920 * (est_c[m] - L.FRAC / 2), 1920 * (est_c[m] + L.FRAC / 2), color="C0", alpha=0.15, lw=0)
    ax.set_xlabel("time [s]"); ax.set_ylabel("x [px]"); ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(PAPER / "figs" / "sn_trajectory.pdf"); plt.close(fig)


def main():
    ev = json.load(open(RES / "sn_eval.json"))
    (PAPER / "tables").mkdir(parents=True, exist_ok=True); (PAPER / "figs").mkdir(parents=True, exist_ok=True)
    tables(ev); fig_scatter(ev)
    print("sn report written")


if __name__ == "__main__":
    main()
