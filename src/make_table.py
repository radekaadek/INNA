"""Write paper/table_results.tex from results/*/test*.json (no hand-copied numbers)."""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent.parent / "paper" / "table_results.tex"
ROWS = [("ssdlite", "SSDlite320-MNv3"), ("frcnn", "Faster R-CNN-MNv3-FPN"), ("yolo11n", "YOLO11n"), ("yolo26n", "YOLO26n"),
        ("cn_mnv4s", "CenterNet-MNv4-S"), ("cn_mnv4m", "CenterNet-MNv4-M"), ("cn_mnv3", "CenterNet-MNv3"),
        ("centernet", "CenterNet-R18")]


def load(run, tta=False):
    f = RES / run / ("test_tta.json" if tta else "test.json")
    return json.load(open(f)) if f.exists() else None


def main():
    rows = []
    for run, label in ROWS:
        for tta in (False, True):
            if tta and run.startswith(("yolo", "ssdlite", "frcnn")):
                continue
            r = load(run, tta)
            if r:
                rows.append((label + ("+TTA" if tta else ""), r))
    best = {k: (max if k in ("ap50", "ap50_95", "ball_in_crop") else min)(r[k] for _, r in rows)
            for k in ("ap50", "ap50_95", "ball_in_crop", "ms_per_frame_cpu4", "ms_per_frame_gpu")}
    def f(r, k, fmt, scale=1):
        s = fmt.format(r[k] * scale)
        return f"\\textbf{{{s}}}" if r[k] == best[k] else s
    lines = []
    for label, r in rows:
        lines.append(f"{label} & {r['params_M']:.1f} & {f(r,'ap50','{:.3f}')} & {f(r,'ap50_95','{:.3f}')} & "
                     f"{f(r,'ball_in_crop','{:.1f}',100)} & {f(r,'ms_per_frame_cpu4','{:.0f}')} & {f(r,'ms_per_frame_gpu','{:.1f}')} \\\\")
    OUT.write_text("\n".join(lines) + "\n\\bottomrule\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
