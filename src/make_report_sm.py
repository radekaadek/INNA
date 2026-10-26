"""results/sm_eval.json + train logs -> paper/tables/results.tex and paper/figs/*.pdf"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sm_lib as L
import sm_data as D
import yolo_sm
from evaluate_sm import NAMES

RES = D.ROOT / "results"
PAPER = D.ROOT / "paper"
NAMES = {**NAMES, "yolo11n_coco_zeroshot": "YOLO11n (COCO, zero-shot) + rule", "yolo11n_finetuned": "YOLO11n (fine-tuned) + rule"}
BASE = {"center": "Fixed centre", "mean_players": "Mean of players (GT)", "median_players": "Median of players (GT)",
        "oracle": "Oracle (GT densest window)"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def pct(x):
    return f"{100 * x:.1f}"


def table(ev):
    rows = []
    for k, v in ev["baselines"].items():
        s, e = v["scores"]["raw"], v["scores"]["ema_lim"]
        rows.append((BASE[k], "--", "--", "--", s, e))
    rows.append(None)
    for k, v in sorted(ev["models"].items(), key=lambda kv: -kv[1]["scores"]["ema_lim"]["players_in_crop"]):
        rows.append((NAMES.get(k, k), f"{v['params_M']:.1f}", f"{v['gpu_ms']:.1f}", f"{v['cpu_ms']:.1f}",
                     v["scores"]["raw"], v["scores"]["ema_lim"]))
    lines = []
    for r in rows:
        if r is None:
            lines.append("\\midrule"); continue
        n, p, g, c, s, e = r
        lines.append(f"{n} & {p} & {g} & {c} & {pct(s['players_in_crop'])} & {pct(e['players_in_crop'])} & "
                     f"{pct(s['center_mae'])} & {pct(s['jitter'])} & {pct(e['jitter'])} \\\\")
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "tables" / "results.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")


def fig_tradeoff(ev):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for k, v in ev["models"].items():
        ax.scatter(v["cpu_ms"], 100 * v["scores"]["ema_lim"]["players_in_crop"], s=18)
        ax.annotate(NAMES.get(k, k).replace(" + rule", "").replace(", 2-frame", " 2f"), (v["cpu_ms"], 100 * v["scores"]["ema_lim"]["players_in_crop"]),
                    fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    for k, ls in (("oracle", "--"), ("center", ":")):
        ax.axhline(100 * ev["baselines"][k]["scores"]["ema_lim"]["players_in_crop"], ls=ls, c="gray", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("CPU latency per frame [ms] (4 threads)"); ax.set_ylabel("Players in crop [%]")
    fig.tight_layout(); fig.savefig(PAPER / "figs" / "tradeoff.pdf"); plt.close(fig)


def best_model(ev):
    return max((k for k in ev["models"] if k.startswith(("r18", "mnv"))), key=lambda k: ev["models"][k]["scores"]["ema_lim"]["players_in_crop"])


def fig_traj(ev, test):
    k = best_model(ev)
    c = np.array(ev["models"][k]["centers"]); seq = test["seq"]
    cs = L.smooth(c, seq, **L.POST["ema_lim"])
    s = int(np.unique(seq)[0]); m = seq == s
    t = test["frame"][m] / 25
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.plot(t, 1280 * test["center"][m], c="gray", lw=0.8, label="oracle")
    ax.plot(t, 1280 * c[m], c="C1", lw=0.8, label=f"{NAMES.get(k, k)} (raw)")
    ax.plot(t, 1280 * cs[m], c="C0", lw=1.2, label="+ EMA & pan limit")
    ax.set_xlabel("time [s]"); ax.set_ylabel("crop centre x [px]"); ax.legend(fontsize=5.5, frameon=False, ncol=1)
    fig.tight_layout(); fig.savefig(PAPER / "figs" / "trajectory.pdf"); plt.close(fig)


def fig_qual(ev, test):
    k = best_model(ev)
    c = L.smooth(np.array(ev["models"][k]["centers"]), test["seq"], **L.POST["ema_lim"])
    idx = [int(i) for i in np.linspace(len(c) * 0.08, len(c) * 0.92, 3)]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.5))
    for ax, i in zip(axs, idx):
        name = f"{test['seqs'][test['seq'][i]]}_{test['frame'][i]:06d}.jpg"
        ax.imshow(Image.open(yolo_sm.YD / "test" / "images" / name)); W = yolo_sm.YW
        for ctr, col in ((test["center"][i], "lime"), (c[i], "red")):
            ax.add_patch(plt.Rectangle(((ctr - L.FRAC / 2) * W, 0), L.FRAC * W, yolo_sm.YH, fill=False, ec=col, lw=1.4))
        ax.axis("off")
    fig.tight_layout(pad=0.2); fig.savefig(PAPER / "figs" / "qualitative.pdf"); plt.close(fig)


def fig_curves():
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for d in sorted(RES.glob("*/train_log.json")):
        lg = json.load(open(d))
        if "val_players_in_crop" not in lg["log"][0]:
            continue
        ax.plot([r["epoch"] for r in lg["log"]], [100 * r["val_players_in_crop"] for r in lg["log"]], lw=1,
                label=NAMES.get(d.parent.name, d.parent.name))
    ax.set_xlabel("epoch"); ax.set_ylabel("val players in crop [%]"); ax.legend(fontsize=5, frameon=False)
    fig.tight_layout(); fig.savefig(PAPER / "figs" / "curves.pdf"); plt.close(fig)


def main():
    (PAPER / "figs").mkdir(parents=True, exist_ok=True)
    ev = json.load(open(RES / "sm_eval.json")); test = L.load_split("test")
    table(ev); fig_tradeoff(ev); fig_traj(ev, test); fig_qual(ev, test); fig_curves()
    print("report written; best model:", best_model(ev))


if __name__ == "__main__":
    main()
