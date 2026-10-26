"""Convert the extra 1280x720 COCO set (Roboflow 'football-player-detection', ball class only) into the 640x640
stretched COCO layout used by data.load_split.  Per frame: one full-frame view, plus 16:9 tiles (640x360 px of the
original, stretched to 640x640 - the same geometry as the tiled video inference) around the ball / random negatives."""
import json
import random
import sys
from pathlib import Path

from PIL import Image

RES = Path(__file__).resolve().parent.parent / "results" / "extra"
SRC = RES / "naimur" / "x"
TW, TH = 640, 360


def convert(split, tiles_per_pos=2, seed=0):
    rng = random.Random(seed)
    ann = next(SRC.joinpath(split).rglob("_annotations.coco.json"))
    d = json.load(open(ann))
    ball_id = next(c["id"] for c in d["categories"] if c["name"].lower() == "football")
    boxes = {}
    for a in d["annotations"]:
        if a["category_id"] == ball_id:
            x, y, w, h = a["bbox"]
            boxes.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
    out = RES / "ds" / split
    out.mkdir(parents=True, exist_ok=True)
    images, annots = [], []

    def add(img, bxs, name):
        iid = len(images) + 1
        img.save(out / name, quality=95)
        images.append({"id": iid, "file_name": name, "width": 640, "height": 640})
        for b in bxs:
            annots.append({"id": len(annots) + 1, "image_id": iid, "category_id": 1,
                           "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": (b[2] - b[0]) * (b[3] - b[1]), "iscrowd": 0})

    for im in d["images"]:
        img = Image.open(ann.parent / im["file_name"]).convert("RGB")
        W, H = img.size
        bx = boxes.get(im["id"], [])
        stem = Path(im["file_name"]).stem[:40]
        sx, sy = 640 / W, 640 / H
        add(img.resize((640, 640), Image.BILINEAR), [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in bx], f"{stem}_full.jpg")
        for k in range(tiles_per_pos if bx else 1):
            if bx:  # tile containing the ball centre
                b = rng.choice(bx)
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                x0 = int(min(max(cx - rng.uniform(0.1, 0.9) * TW, 0), W - TW))
                y0 = int(min(max(cy - rng.uniform(0.1, 0.9) * TH, 0), H - TH))
            else:
                x0, y0 = rng.randint(0, W - TW), rng.randint(0, H - TH)
            tb = []
            for b in bx:
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if x0 <= cx < x0 + TW and y0 <= cy < y0 + TH:
                    tb.append([(b[0] - x0) * 640 / TW, (b[1] - y0) * 640 / TH, (b[2] - x0) * 640 / TW, (b[3] - y0) * 640 / TH])
            add(img.crop((x0, y0, x0 + TW, y0 + TH)).resize((640, 640), Image.BILINEAR), tb, f"{stem}_t{k}.jpg")
    json.dump({"images": images, "annotations": annots, "categories": [{"id": 1, "name": "ball"}]},
              open(out / "_annotations.coco.json", "w"))
    print(split, len(images), "views,", len(annots), "ball boxes,", len({a["image_id"] for a in annots}), "positive views")


if __name__ == "__main__":
    for s in sys.argv[1:] or ["train", "valid", "test"]:
        convert(s)
