"""Build a 640x640 COCO ball dataset from SN-Tracking-2023 (needs sn_scan.py first).  Frames are read straight out of the
remote zips (range requests).  Same geometry as extra_data.py: one full-frame view plus 16:9 tiles (640x360 px of the
1080p frame, stretched to 640x640) around the ball / random negatives.  Splits are by clip: valid = last 6 train clips."""
import io
import json
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfFileSystem
from PIL import Image

RES = Path(__file__).resolve().parent.parent / "results" / "sn_tracking"
REPO = "datasets/SoccerNet/SN-Tracking-2023/"
TW, TH = 640, 360
N_VALID = 6


def build(split, src, clips, stride, tiles_per_pos, seed=0):
    rng = random.Random(seed)
    meta = json.load(open(RES / f"ball_{src}.json"))
    out = RES / "ds" / split
    out.mkdir(parents=True, exist_ok=True)
    jobs = []  # (clip, frame, box)
    for c in clips:
        frames = sorted(meta[c]["boxes"], key=int)
        for fr in frames[rng.randrange(stride)::stride]:
            jobs.append((c, int(fr), meta[c]["boxes"][fr]))
    jobs = [(j, [(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)) for _ in range(tiles_per_pos)]) for j in jobs]
    chunks = [jobs[i::8] for i in range(8)]

    def work(chunk):
        """Fetch the frames of one chunk and write the views to disk; returns [(file_name, [box])]."""
        res = []
        fs = HfFileSystem()
        with fs.open(REPO + f"{src}.zip", "rb") as f:
            z = zipfile.ZipFile(f)
            for (c, fr, (x, y, w, h)), offs in chunk:
                img = Image.open(io.BytesIO(z.read(f"{src}/{c}/img1/{fr:06d}.jpg"))).convert("RGB")
                W, H = img.size
                b = [x, y, x + w, y + h]
                stem = f"{c}_{fr:06d}"
                sx, sy = 640 / W, 640 / H
                views = [(img.resize((640, 640), Image.BILINEAR), [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], f"{stem}_full.jpg")]
                for k, (ux, uy) in enumerate(offs):
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    x0 = int(min(max(cx - ux * TW, 0), W - TW))
                    y0 = int(min(max(cy - uy * TH, 0), H - TH))
                    views.append((img.crop((x0, y0, x0 + TW, y0 + TH)).resize((640, 640), Image.BILINEAR),
                                  [(b[0] - x0) * 640 / TW, (b[1] - y0) * 640 / TH, (b[2] - x0) * 640 / TW, (b[3] - y0) * 640 / TH], f"{stem}_t{k}.jpg"))
                for v, vb, name in views:
                    v.save(out / name, quality=95)
                    res.append((name, vb))
        return res

    with ThreadPoolExecutor(8) as ex:
        done = sorted(x for part in ex.map(work, chunks) for x in part)
    images, annots = [], []
    for name, b in done:
        images.append({"id": len(images) + 1, "file_name": name, "width": 640, "height": 640})
        annots.append({"id": len(annots) + 1, "image_id": len(images), "category_id": 1,
                       "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": (b[2] - b[0]) * (b[3] - b[1]), "iscrowd": 0})
    json.dump({"images": images, "annotations": annots, "categories": [{"id": 1, "name": "ball"}]}, open(out / "_annotations.coco.json", "w"))
    print(split, len(clips), "clips,", len(jobs), "frames,", len(images), "views,", len(annots), "ball boxes", flush=True)


def build2(split, src, clips, stride, seed=1):
    """Tile-heavy train set: per frame a native 640x360 tile, a 2x zoomed 320x180 tile around the ball, a ball-free tile
    (hard negative) and, for every 2nd frame, the full frame."""
    rng = random.Random(seed)
    meta = json.load(open(RES / f"ball_{src}.json"))
    out = RES / "ds2" / split
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for c in clips:
        frames = sorted(meta[c]["boxes"], key=int)
        for i, fr in enumerate(frames[rng.randrange(stride)::stride]):
            jobs.append(((c, int(fr), meta[c]["boxes"][fr]), i % 2 == 0,
                         [rng.uniform(0.1, 0.9) for _ in range(6)], [rng.random() for _ in range(2)]))
    chunks = [jobs[i::8] for i in range(8)]

    def tile(img, b, tw, th, ux, uy):
        W, H = img.size
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x0 = int(min(max(cx - ux * tw, 0), W - tw))
        y0 = int(min(max(cy - uy * th, 0), H - th))
        bx = [(b[0] - x0) * 640 / tw, (b[1] - y0) * 640 / th, (b[2] - x0) * 640 / tw, (b[3] - y0) * 640 / th]
        return img.crop((x0, y0, x0 + tw, y0 + th)).resize((640, 640), Image.BILINEAR), [bx]

    def work(chunk):
        res = []
        with HfFileSystem().open(REPO + f"{src}.zip", "rb") as f:
            z = zipfile.ZipFile(f)
            for (c, fr, (x, y, w, h)), full, u, nr in chunk:
                img = Image.open(io.BytesIO(z.read(f"{src}/{c}/img1/{fr:06d}.jpg"))).convert("RGB")
                W, H = img.size
                b = [x, y, x + w, y + h]
                stem = f"{c}_{fr:06d}"
                views = [tile(img, b, TW, TH, u[0], u[1]) + (f"{stem}_t0.jpg",),
                         tile(img, b, TW // 2, TH // 2, u[2], u[3]) + (f"{stem}_z0.jpg",)]
                if full:
                    views.append((img.resize((640, 640), Image.BILINEAR), [[b[0] * 640 / W, b[1] * 640 / H, b[2] * 640 / W, b[3] * 640 / H]], f"{stem}_full.jpg"))
                for _ in range(10):  # random tile without the ball (nor its neighbourhood)
                    x0, y0 = int(nr[0] * (W - TW)), int(nr[1] * (H - TH))
                    if not (x0 - 40 < b[2] and b[0] < x0 + TW + 40 and y0 - 40 < b[3] and b[1] < y0 + TH + 40):
                        views.append((img.crop((x0, y0, x0 + TW, y0 + TH)).resize((640, 640), Image.BILINEAR), [], f"{stem}_n0.jpg"))
                        break
                    nr = [rng_local.random(), rng_local.random()]
                for v, vb, name in views:
                    v.save(out / name, quality=95)
                    res.append((name, vb))
        return res

    rng_local = random.Random(seed + 1)
    with ThreadPoolExecutor(8) as ex:
        done = sorted(x for part in ex.map(work, chunks) for x in part)
    images, annots = [], []
    for name, bxs in done:
        images.append({"id": len(images) + 1, "file_name": name, "width": 640, "height": 640})
        for b in bxs:
            annots.append({"id": len(annots) + 1, "image_id": len(images), "category_id": 1,
                           "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": (b[2] - b[0]) * (b[3] - b[1]), "iscrowd": 0})
    json.dump({"images": images, "annotations": annots, "categories": [{"id": 1, "name": "ball"}]}, open(out / "_annotations.coco.json", "w"))
    print(split, len(clips), "clips,", len(jobs), "frames,", len(images), "views,", len(annots), "ball boxes", flush=True)


def build_testz(seed=2):
    """Zoomed SN test views (same clips/frames as `test`): a 2x tile (320x180 px of the frame) around the ball plus a ball-free
    640x360 tile per frame, for scoring zoomed inference."""
    rng = random.Random(seed)
    meta = json.load(open(RES / "ball_test.json"))
    out = RES / "dsz" / "test"
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for c in sorted(meta):
        frames = sorted(meta[c]["boxes"], key=int)
        for fr in frames[rng.randrange(25)::25]:
            jobs.append(((c, int(fr), meta[c]["boxes"][fr]), [rng.uniform(0.1, 0.9) for _ in range(2)], [rng.random() for _ in range(2)]))
    chunks = [jobs[i::8] for i in range(8)]

    def work(chunk):
        res = []
        with HfFileSystem().open(REPO + "test.zip", "rb") as f:
            z = zipfile.ZipFile(f)
            for (c, fr, (x, y, w, h)), u, nr in chunk:
                img = Image.open(io.BytesIO(z.read(f"test/{c}/img1/{fr:06d}.jpg"))).convert("RGB")
                W, H = img.size
                b = [x, y, x + w, y + h]
                stem = f"{c}_{fr:06d}"
                tw, th = TW // 2, TH // 2
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                x0, y0 = int(min(max(cx - u[0] * tw, 0), W - tw)), int(min(max(cy - u[1] * th, 0), H - th))
                bx = [(b[0] - x0) * 640 / tw, (b[1] - y0) * 640 / th, (b[2] - x0) * 640 / tw, (b[3] - y0) * 640 / th]
                img.crop((x0, y0, x0 + tw, y0 + th)).resize((640, 640), Image.BILINEAR).save(out / f"{stem}_z0.jpg", quality=95)
                res.append((f"{stem}_z0.jpg", [bx]))
                for _ in range(10):
                    x0, y0 = int(nr[0] * (W - TW)), int(nr[1] * (H - TH))
                    if not (x0 - 40 < b[2] and b[0] < x0 + TW + 40 and y0 - 40 < b[3] and b[1] < y0 + TH + 40):
                        img.crop((x0, y0, x0 + TW, y0 + TH)).resize((640, 640), Image.BILINEAR).save(out / f"{stem}_n0.jpg", quality=95)
                        res.append((f"{stem}_n0.jpg", []))
                        break
                    nr = [rng.random(), rng.random()]
        return res

    with ThreadPoolExecutor(8) as ex:
        done = sorted(x for part in ex.map(work, chunks) for x in part)
    images, annots = [], []
    for name, bxs in done:
        images.append({"id": len(images) + 1, "file_name": name, "width": 640, "height": 640})
        for b in bxs:
            annots.append({"id": len(annots) + 1, "image_id": len(images), "category_id": 1,
                           "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": (b[2] - b[0]) * (b[3] - b[1]), "iscrowd": 0})
    json.dump({"images": images, "annotations": annots, "categories": [{"id": 1, "name": "ball"}]}, open(out / "_annotations.coco.json", "w"))
    print("testz", len(jobs), "frames,", len(images), "views,", len(annots), "ball boxes", flush=True)


if __name__ == "__main__":
    tr = sorted(json.load(open(RES / "ball_train.json")))
    te = sorted(json.load(open(RES / "ball_test.json")))
    which = sys.argv[1:] or ["train", "valid", "test"]
    if "train" in which:
        build("train", "train", tr[:-N_VALID], 15, 2)
    if "valid" in which:
        build("valid", "train", tr[-N_VALID:], 25, 1)
    if "test" in which:
        build("test", "test", te, 25, 1)
    if "train2" in which:
        build2("train", "train", tr[:-N_VALID], 5)
    if "testz" in which:
        build_testz()
