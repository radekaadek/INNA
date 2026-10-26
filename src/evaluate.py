"""Evaluation: COCO AP50, CPU latency and 9:16 cropping quality (ball-in-crop)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import crop
import models
from data import BallDataset, collate, load_split

RES = Path(__file__).resolve().parent.parent / "results"


def run_inference(model, items, bs=8):
    ds = BallDataset(items)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, collate_fn=collate, num_workers=2)
    preds = {}
    dev = next(model.parameters()).device
    for imgs, tg in dl:
        out = models.predict(model, [i.to(dev) for i in imgs])
        for t, o in zip(tg, out):
            preds[t["image_id"]] = [[*b.tolist(), s.item()] for b, s in zip(o["boxes"].cpu(), o["scores"].cpu())]
    return preds


def coco_ap(items, preds):
    gt = {"images": [{"id": i, "width": 640, "height": 640} for i, _, _ in items], "categories": [{"id": 1}], "annotations": []}
    for i, _, bs in items:
        for b in bs:
            gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": i, "category_id": 1, "iscrowd": 0,
                                      "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": (b[2] - b[0]) * (b[3] - b[1])})
    dt = [{"image_id": i, "category_id": 1, "bbox": [d[0], d[1], d[2] - d[0], d[3] - d[1]], "score": d[4]}
          for i, ds in preds.items() for d in ds]
    if not dt:
        return 0.0, 0.0
    c = COCO(); c.dataset = gt; c.createIndex()
    E = COCOeval(c, c.loadRes(dt), "bbox"); E.params.maxDets = [1, 10, 100]
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        E.evaluate(); E.accumulate(); E.summarize()
    return float(E.stats[1]), float(E.stats[0])  # AP50, AP50:95


def crop_stats(items, preds, thr):
    pos = [(i, bs[0]) for i, _, bs in items if bs]
    hit = sum(crop.ball_in_crop(b, crop.choose_center(preds.get(i, []), thr)) for i, b in pos)
    return hit / len(pos)


def center_baseline(items):
    pos = [bs[0] for _, _, bs in items if bs]
    return sum(crop.ball_in_crop(b, 320) for b in pos) / len(pos)


def best_threshold(items, preds):
    thrs = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    sc = [crop_stats(items, preds, t) for t in thrs]
    return thrs[int(np.argmax(sc))]


def latency(model, dev="cpu", n=60, warm=10):
    torch.set_num_threads(4)
    model.to(dev).eval()
    x = [torch.rand(3, 640, 640, device=dev)]
    ts = []
    for k in range(n + warm):
        t = time.perf_counter(); models.predict(model, x)
        if dev != "cpu":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts[warm:]) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()
    if a.run.startswith("yolo"):
        model = models.YoloWrap(str(RES / a.run / "train" / "weights" / "best.pt"))
        ck = {"model": a.run}
    else:
        ck = torch.load(RES / a.run / "best.pt", map_location="cpu")
        model = models.build(ck["model"], pretrained=ck.get("pretrained", True))
        model.load_state_dict(ck["state"])
    model.tta = a.tta
    tag = a.run + ("_tta" if a.tta else "")
    valid, test = load_split("valid"), load_split("test")
    ms_cpu = latency(model, "cpu")
    ms_gpu = latency(model, "cuda") if torch.cuda.is_available() else None
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    thr = best_threshold(valid, run_inference(model, valid))
    preds = run_inference(model, test)
    ap50, ap = coco_ap(test, preds)
    r = {"run": tag, "model": ck["model"], "ap50": ap50, "ap50_95": ap, "thr": thr,
         "ball_in_crop": crop_stats(test, preds, thr), "center_baseline": center_baseline(test),
         "ms_per_frame_cpu4": ms_cpu, "ms_per_frame_gpu": ms_gpu, "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
         "size_MB": sum(p.numel() for p in model.parameters()) * 4 / 1e6}
    json.dump(r, open(RES / a.run / f"test{tag[len(a.run):]}.json", "w"), indent=1)
    json.dump({str(k): v for k, v in preds.items()}, open(RES / a.run / f"test_preds{tag[len(a.run):]}.json", "w"))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
