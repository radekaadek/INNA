"""Evaluate all runs on the held-out test split (official SportsMOT val football videos) -> results/sm_eval.json."""
import json
import time
from pathlib import Path

import numpy as np
import torch

import sm_lib as L
import sm_data as D
import yolo_sm

RES = D.ROOT / "results"
NAMES = {"r18_hist": "ResNet-18 hist", "r18_reg": "ResNet-18 reg", "r18_hist_2f": "ResNet-18 hist, 2-frame",
         "r18_hist_s1": "ResNet-18 hist (seed 1)", "r18_hist_s2": "ResNet-18 hist (seed 2)", "mnv3_hist": "MobileNetV3 hist", "mnv4s_hist": "MobileNetV4-S hist"}


def eval_cropnets(test):
    out = {}
    for d in sorted(RES.glob("*/best.pt")):
        run = d.parent.name
        if not (RES / run / "train_log.json").exists():
            continue
        ck = torch.load(d, map_location="cuda")
        c = ck["cfg"]
        m = L.CropNet(c["backbone"], c["head"], c["two_frame"], pretrained=False).cuda()
        m(L.prep(test["images"][:2], test["prev"][:2], c["two_frame"]))
        m.load_state_dict(ck["state"])
        centers = L.predict_centers(m, test)
        r = {"params_M": L.n_params(m), "gpu_ms": L.latency(m, "cuda"), "cfg": c}
        torch.set_num_threads(4)
        r["cpu_ms"] = L.latency(m.cpu(), "cpu", n=30)
        r["scores"] = {p: L.score(test, centers, p) for p in L.POST}
        r["centers"] = centers.tolist()
        out[run] = r
        print(run, r["scores"]["raw"], flush=True)
    return out


def yolo_centers(model, test, conf=0.25):
    names = [f"{test['seqs'][s]}_{f:06d}.jpg" for s, f in zip(test["seq"], test["frame"])]
    centers = []
    for i in range(0, len(names), 32):
        res = model.predict([str(yolo_sm.YD / "test" / "images" / n) for n in names[i:i + 32]], imgsz=640, conf=conf,
                            classes=[0], verbose=False, device=0)
        for r in res:
            b = r.boxes.xyxyn.cpu().numpy()
            centers.append(D.oracle_center((b[:, 0] + b[:, 2]) / 2) if len(b) else 0.5)
    return np.array(centers)


def eval_yolo(test):
    from ultralytics import YOLO
    out = {}
    cands = {"yolo11n_coco_zeroshot": "yolo11n.pt"}
    ft = RES / "yolo11n_sm" / "train" / "weights" / "best.pt"
    if ft.exists():
        cands["yolo11n_finetuned"] = str(ft)
    img = np.random.randint(0, 255, (360, 640, 3), np.uint8)
    for name, w in cands.items():
        m = YOLO(w)
        centers = yolo_centers(m, test)
        lat = {}
        for dev in (0, "cpu"):
            for _ in range(5):
                m.predict(img, imgsz=640, verbose=False, device=dev)
            ts = []
            for _ in range(30):
                t = time.perf_counter(); m.predict(img, imgsz=640, verbose=False, device=dev); ts.append(time.perf_counter() - t)
            lat["gpu_ms" if dev == 0 else "cpu_ms"] = float(np.median(ts) * 1000)
        out[name] = {"params_M": sum(p.numel() for p in m.model.parameters()) / 1e6, **lat,
                     "scores": {p: L.score(test, centers, p) for p in L.POST}, "centers": centers.tolist()}
        print(name, out[name]["scores"]["raw"], flush=True)
    return out


def main():
    test = L.load_split("test")
    res = {"n_frames": len(test["seq"]), "n_videos": int(len(test["seqs"])), "baselines": {}}
    for k, c in L.baselines(test).items():
        res["baselines"][k] = {"scores": {p: L.score(test, c, p) for p in L.POST}}
    res["models"] = eval_cropnets(test)
    res["models"].update(eval_yolo(test))
    json.dump(res, open(RES / "sm_eval.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
