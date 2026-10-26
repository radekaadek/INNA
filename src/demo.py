"""Figures: example 9:16 crops from test predictions, and summary table (LaTeX rows)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import crop
from data import load_split

RES = Path(__file__).resolve().parent.parent / "results"


def examples(run, n_hit=4, n_miss=2):
    """One strip of 9:16 crops chosen by the model (green = ground-truth ball): hits first, then misses."""
    from PIL import ImageDraw
    r = json.load(open(RES / run / "test.json"))
    preds = {int(k): v for k, v in json.load(open(RES / run / "test_preds.json")).items()}
    items = [it for it in load_split("test") if it[2]]
    hit = lambda it: crop.ball_in_crop(it[2][0], crop.choose_center(preds.get(it[0], []), r["thr"]))
    hits = [it for it in items if hit(it)]
    misses = [it for it in items if not hit(it)]
    sel = hits[::max(1, len(hits) // n_hit)][:n_hit] + misses[:n_miss]
    tiles = []
    for iid, p, gt in sel:
        x0, x1 = crop.window(crop.choose_center(preds.get(iid, []), r["thr"]))
        im = Image.open(p).convert("RGB").crop((int(x0), 0, int(x1), 640))
        d = ImageDraw.Draw(im); g = gt[0]
        d.rectangle((g[0] - x0 - 3, g[1] - 3, g[2] - x0 + 3, g[3] + 3), outline=(0, 255, 0), width=2)
        tiles.append(im)
    W = sum(t.width for t in tiles) + 6 * (len(tiles) - 1)
    strip = Image.new("RGB", (W, 640), "white"); x = 0
    for i, t in enumerate(tiles):
        strip.paste(t, (x, 0)); x += t.width + 6
        if i == n_hit - 1 and len(tiles) > n_hit:
            strip.paste(Image.new("RGB", (6, 640), (220, 40, 40)), (x - 6, 0))
    strip.save(RES / "examples.png")


def table(runs):
    for run in runs:
        r = json.load(open(RES / run / "test.json"))
        print(f"{run} & {r['params_M']:.1f} & {r['ap50']*100:.1f} & {r['ball_in_crop']*100:.1f} & {r['ms_per_frame_cpu4']:.0f} \\\\")


if __name__ == "__main__":
    examples(sys.argv[1])
    table(sys.argv[1:])


def pareto(runs, out="pareto.png"):
    """AP50 vs CPU latency scatter (marker size ~ parameters)."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    off = {"yolo26n": (-4, 5), "yolo11n": (-2, -9), "cn_mnv4s": (4, -9), "cn_mnv4m": (4, 3), "cn_mnv3": (4, -9),
           "ssdlite": (4, 3), "centernet": (-10, 6), "cn_mnv3_tta": (4, -9), "centernet_tta": (-34, 6), "cn_mnv4s_tta": (4, 3)}
    for run in runs:
        base = run.replace("_tta", "")
        r = json.load(open(RES / base / ("test_tta.json" if run.endswith("_tta") else "test.json")))
        ax.scatter(r["ms_per_frame_cpu4"], r["ap50"], s=15 + 10 * r["params_M"], alpha=.7,
                   marker="s" if run.endswith("_tta") else "o")
        ax.annotate(run.replace("cn_", "CN-").replace("_tta", "+TTA").replace("centernet", "CN-R18"),
                    (r["ms_per_frame_cpu4"], r["ap50"]), fontsize=5.5, xytext=off.get(run, (3, 3)), textcoords="offset points")
    ax.set_xlabel("CPU latency [ms/frame, 4 threads]", fontsize=7); ax.set_ylabel("test AP$_{50}$", fontsize=7)
    ax.set_ylim(0.80, 0.97); ax.tick_params(labelsize=6); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(RES / out, dpi=250); plt.close()
