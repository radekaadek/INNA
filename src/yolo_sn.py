"""2-class (player, ball) YOLO11n on SN-Tracking-2023 frames (896x504 JPEGs from sn_build.py).
Usage: yolo_sn.py build | yolo_sn.py train [minutes]"""
import sys
import numpy as np
from sn_build import SN, ROOT
RES = ROOT / "results"; YD = RES / "sn_yolo"


def build():
    for name in ("train", "valid", "test"):
        d = np.load(SN / f"{name}.npz")
        (YD / name).mkdir(parents=True, exist_ok=True)
        (YD / name / "images").symlink_to(SN / "frames" / name) if not (YD / name / "images").exists() else None
        (YD / name / "labels").mkdir(exist_ok=True)
        for i, n in enumerate(d["names"]):
            lines = []
            for b in d["boxes"][i]:
                if b[0] >= 0:
                    lines.append(f"0 {(b[0] + b[2]) / 2:.5f} {(b[1] + b[3]) / 2:.5f} {b[2] - b[0]:.5f} {b[3] - b[1]:.5f}")
            b = d["ball"][i]
            if b[0] >= 0:
                lines.append(f"1 {(b[0] + b[2]) / 2:.5f} {(b[1] + b[3]) / 2:.5f} {b[2] - b[0]:.5f} {b[3] - b[1]:.5f}")
            (YD / name / "labels" / n.replace(".jpg", ".txt")).write_text("\n".join(lines))
    (YD / "data.yaml").write_text(f"path: {YD}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: player\n  1: ball\n")


def train(minutes):
    from ultralytics import YOLO
    YOLO("yolo11n.pt").train(data=str(YD / "data.yaml"), time=minutes / 60, epochs=30, patience=8, imgsz=896, batch=6, device=0,
                             workers=4, weight_decay=0.01, project=str(RES / "yolo11n_sn"), name="train", exist_ok=True, plots=False, seed=0)


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else train(float(sys.argv[2]) if len(sys.argv) > 2 else 35)
