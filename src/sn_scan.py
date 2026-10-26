"""Scan SoccerNet SN-Tracking-2023 zips remotely (HTTP range reads, no full download) and store per-clip ball boxes.
Writes results/sn_tracking/ball_<split>.json: {clip: {"size": [W, H], "n_frames": N, "boxes": {frame: [x, y, w, h]}}}."""
import configparser
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfFileSystem

OUT = Path(__file__).resolve().parent.parent / "results" / "sn_tracking"
REPO = "datasets/SoccerNet/SN-Tracking-2023/"


def scan_clip(split, clip, fs):
    with fs.open(REPO + f"{split}.zip", "rb") as f:
        z = zipfile.ZipFile(f)
        gi = configparser.ConfigParser()
        gi.read_string(z.read(f"{split}/{clip}/gameinfo.ini").decode())
        ball_ids = [k.split("_")[1] for k, v in gi["Sequence"].items() if k.startswith("trackletid_") and v.strip().startswith("ball")]
        si = configparser.ConfigParser()
        si.read_string(z.read(f"{split}/{clip}/seqinfo.ini").decode())
        W, H, n = int(si["Sequence"]["imWidth"]), int(si["Sequence"]["imHeight"]), int(si["Sequence"]["seqLength"])
        boxes = {}
        for line in z.read(f"{split}/{clip}/gt/gt.txt").decode().splitlines():
            p = line.split(",")
            if p[1] in ball_ids:
                boxes[int(p[0])] = [float(v) for v in p[2:6]]
    return clip, {"size": [W, H], "n_frames": n, "boxes": boxes}


def scan(split):
    fs = HfFileSystem()
    with fs.open(REPO + f"{split}.zip", "rb") as f:
        names = zipfile.ZipFile(f).namelist()
    clips = sorted({n.split("/")[1] for n in names if n.endswith("gt/gt.txt")})
    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(lambda c: scan_clip(split, c, HfFileSystem()), clips))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"ball_{split}.json", "w"))
    nb = [len(v["boxes"]) for v in res.values()]
    sizes = sorted({tuple(v["size"]) for v in res.values()})
    ws = sorted(b[2] for v in res.values() for b in v["boxes"].values())
    print(split, len(res), "clips,", sum(nb), "ball boxes /", sum(v["n_frames"] for v in res.values()), "frames; clips without ball:",
          sum(n == 0 for n in nb), "sizes", sizes, "ball width px median", ws[len(ws) // 2] if ws else None, flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["train", "test"]:
        scan(s)
