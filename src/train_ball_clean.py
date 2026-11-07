"""Backlog #14: as train_ball.py but without label-suspect training frames. Train the ResNet-18 ball heatmap network on random crops of SN-Tracking-2023 frames."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sm_lib as L
import sn_lib as S

RES = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ball_r18_clean")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--per-epoch", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-min", type=float, default=45)
    ap.add_argument("--patience", type=int, default=4)
    a = ap.parse_args()
    out = RES / a.run; out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    torch.manual_seed(0); torch.backends.cudnn.benchmark = True
    tr, va = S.load("train"), S.load("valid")
    keep = ~np.load(RES / "exp" / "train_suspect.npy")  # backlog #14: drop label-suspect frames
    print("train frames kept", int(keep.sum()), "of", len(keep), flush=True)
    tr = {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(keep) else v) for k, v in tr.items()}
    vsub = np.arange(0, len(va["names"]), 2)  # half of the validation frames keeps epochs short
    vd = {k: (v[vsub] if k in ("ball", "seq", "names", "boxes", "center") else v) for k, v in va.items()}
    model = S.BallNet().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    steps = a.epochs * (a.per_epoch // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    log, best, bad, t_start = [], -1, 0, time.time()
    for ep in range(a.epochs):
        t0 = time.time(); model.train(); tot, n = 0.0, 0
        dl = torch.utils.data.DataLoader(S.BallCrops(tr, "train", a.per_epoch, seed=ep), batch_size=a.bs, num_workers=6, drop_last=True)
        for x, hm in dl:
            x, hm = x.cuda(non_blocking=True), hm.cuda(non_blocking=True)
            loss = S.focal_loss(model(x), hm)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10); opt.step(); sched.step()
            tot += loss.item(); n += 1
        cands = S.ball_candidates(model, vd["names"], "valid")
        top1 = np.array([[c[0][0], c[0][1], c[0][2]] if c else [0, 0, 0] for c in cands])
        m = S.ball_hit_rate(vd, top1, 0.02)  # top-1 peak, no threshold (scores > 0)
        rec = {"epoch": ep + 1, "train_loss": tot / n, "val_top1_hit": m["ball_recall"], "sec": time.time() - t0}
        log.append(rec); print(rec, flush=True)
        if m["ball_recall"] > best:
            best, bad = m["ball_recall"], 0
            torch.save({"state": model.state_dict()}, out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break
        if (time.time() - t_start) / 60 > a.max_min:
            print("time cap"); break
    json.dump({"args": vars(a), "log": log, "best": best}, open(out / "train_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
