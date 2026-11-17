"""#20: fine-tune a YOLO model (player+ball dataset built by yolo_sn.py) for a fixed time budget. Usage: yolo_sn2.py <model.pt|.yaml> <minutes> <imgsz> <name> [batch]"""
import sys
from sn_build import ROOT
RES = ROOT / "results"

if __name__ == "__main__":
    from ultralytics import YOLO
    model, minutes, imgsz, name = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    batch = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    YOLO(model).train(data=str(RES / "sn_yolo" / "data.yaml"), time=minutes / 60, epochs=30, patience=8, imgsz=imgsz, batch=batch, device=0,
                      workers=4, weight_decay=0.01, project=str(RES / "yolo_sn2"), name=name, exist_ok=True, plots=False, seed=0, amp=False)
