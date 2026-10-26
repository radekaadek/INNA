"""AP50 of a run on the held-out extra-domain test views (extra_data.py), full-frame views and tile views separately."""
import json
import sys

import torch

import models
from data import EXTRA, SN, SNZ, load_split
from evaluate import RES, coco_ap, run_inference

root = SN if sys.argv[1] == "--sn" else SNZ if sys.argv[1] == "--snz" else EXTRA
for run in sys.argv[2:] if root is not EXTRA else sys.argv[1:]:
    ck = torch.load(RES / run / "best.pt", map_location="cpu")
    m = models.build(ck["model"], pretrained=False)
    m.load_state_dict(ck["state"])
    m.to("cuda").eval()
    items = load_split("test", root)
    r = {}
    for name, sel in (("all", lambda p: True), ("full", lambda p: p.stem.endswith("_full")), ("tiles", lambda p: "_t" in p.stem[-3:]),
                      ("zoom", lambda p: p.stem.endswith("_z0")), ("neg_tiles", lambda p: p.stem.endswith("_n0"))):
        sub = [it for it in items if sel(it[1])]
        if not sub or (name == "neg_tiles"):  # empty subset / negatives only (AP undefined): skip
            continue
        r[name] = (len(sub), round(coco_ap(sub, run_inference(m, sub))[0], 3))
    print(run, r, flush=True)
    json.dump(r, open(RES / run / ("test_sn.json" if root is SN else "test_snz.json" if root is SNZ else "test_extra.json"), "w"))
