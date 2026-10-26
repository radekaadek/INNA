"""Fine-tune an Ultralytics YOLO model (COCO-pretrained) on the ball dataset. Usage: yolo_train.py yolo26n"""
import sys
from pathlib import Path

from ultralytics import YOLO

RES = Path(__file__).resolve().parent.parent / "results"
name = sys.argv[1]
YOLO(str(RES / "yolo_smoke" / f"{name}.pt")).train(
    data=str(RES / "yolo_data" / "data.yaml"), epochs=20, patience=6, imgsz=640, batch=8, device=0, workers=3,
    weight_decay=0.01, project=str(RES / name), name="train", exist_ok=True, plots=False, seed=0)
