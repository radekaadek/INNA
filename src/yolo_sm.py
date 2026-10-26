"""Player detector for the detector+rule baseline: YOLO11n fine-tuned on SportsMOT football (same frames as sm_data.py).
Usage: yolo_sm.py build | yolo_sm.py train [minutes]"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import sm_data as D

RES = D.ROOT / "results"
YD = RES / "sm_yolo"
YW, YH = 640, 360


def build():
    tr = D.football("train")
    parts = {"train": ("train", tr[:-D.VALID_SEQS]), "valid": ("train", tr[-D.VALID_SEQS:]), "test": ("val", D.football("val"))}
    for name, (split, seqs) in parts.items():
        (YD / name / "images").mkdir(parents=True, exist_ok=True); (YD / name / "labels").mkdir(parents=True, exist_ok=True)
        cached = np.load(D.OUT / f"{name}.npz")  # same frames, same order as the regressors
        for sid, f in zip(cached["seq"], cached["frame"]):
            s = seqs[sid]; stem = f"{s}_{f:06d}"
            per, _ = D.read_gt(D.SRC / split / s)
            Image.open(D.SRC / split / s / "img1" / f"{f:06d}.jpg").convert("RGB").resize((YW, YH), Image.BILINEAR).save(
                YD / name / "images" / f"{stem}.jpg", quality=92)
            lines = []
            for x0, y0, x1, y1 in per[int(f)]:
                lines.append(f"0 {(x0 + x1) / 2:.5f} {(y0 + y1) / 2:.5f} {x1 - x0:.5f} {y1 - y0:.5f}")
            (YD / name / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        print(name, len(cached["seq"]), flush=True)
    (YD / "data.yaml").write_text(f"path: {YD}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: player\n")


def train(minutes):
    from ultralytics import YOLO
    YOLO("yolo11n.pt").train(
        data=str(YD / "data.yaml"), time=minutes / 60, epochs=30, patience=8, imgsz=640, batch=8, device=0, workers=3,
        weight_decay=0.01, project=str(RES / "yolo11n_sm"), name="train", exist_ok=True, plots=False, seed=0)


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else train(float(sys.argv[2]) if len(sys.argv) > 2 else 25)
