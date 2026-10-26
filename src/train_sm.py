"""Train a CropNet (small CNN -> crop centre / player-density histogram) on cached SportsMOT football frames."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sm_lib as L

RES = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "mobilenetv3", "mobilenetv4s"])
    ap.add_argument("--head", default="hist", choices=["hist", "reg"])
    ap.add_argument("--two-frame", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--max-min", type=float, default=0, help="wall-clock cap in minutes (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="sm", choices=["sm", "sn"], help="sm: SportsMOT football, sn: SN-Tracking-2023 (448x252 views)")
    a = ap.parse_args()
    out = RES / a.run; out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - refusing to fall back to CPU")
    dev = "cuda"
    if a.data == "sn":
        import sn_lib
        tr, va = sn_lib.load_small("train"), sn_lib.load_small("valid")
    else:
        tr, va = L.load_split("train"), L.load_split("valid")
    model = L.CropNet(a.backbone, a.head, a.two_frame).to(dev)
    model(L.prep(tr["images"][:2], tr["prev"][:2], a.two_frame))  # materialise the lazy layers
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    n = len(tr["images"]); steps = a.epochs * (n // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    torch.backends.cudnn.benchmark = True
    log, best, bad, t_start = [], -1, 0, time.time()
    for ep in range(a.epochs):
        t0 = time.time(); model.train(); perm = rng.permutation(n); tot = 0.0
        for i in range(0, n - a.bs + 1, a.bs):
            idx = np.sort(perm[i:i + a.bs])
            x = L.prep(tr["images"][idx], tr["prev"][idx], a.two_frame)
            dens = torch.from_numpy(tr["density"][idx]).to(dev)
            ctr = torch.from_numpy(tr["center"][idx]).to(dev)
            flip = torch.rand(len(idx), device=dev) < 0.5  # horizontal flip: mirror image, density and centre
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
            dens = torch.where(flip.view(-1, 1), dens.flip(1), dens)
            ctr = torch.where(flip, 1 - ctr, ctr)
            x[:, :3] = x[:, :3] * torch.empty(len(idx), 1, 1, 1, device=dev).uniform_(0.8, 1.2) \
                + torch.empty(len(idx), 1, 1, 1, device=dev).uniform_(-0.2, 0.2)  # contrast / brightness jitter
            y = model(x)  # fp32: the GTX 1650 has no tensor cores and AMP is ~4x slower
            if a.head == "hist":
                tgt = F.avg_pool1d(dens[:, None], 3, 1, 1)[:, 0] + 1e-3
                tgt = tgt / tgt.sum(1, keepdim=True)
                loss = F.kl_div(y.float().log_softmax(-1), tgt, reduction="batchmean")
            else:
                loss = F.l1_loss(y.float(), ctr)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        m = L.score(va, L.predict_centers(model, va), "raw")
        rec = {"epoch": ep + 1, "train_loss": tot / (n // a.bs), **{f"val_{k}": v for k, v in m.items()}, "sec": time.time() - t0}
        log.append(rec); print(rec, flush=True)
        if m["players_in_crop"] > best:
            best, bad = m["players_in_crop"], 0
            torch.save({"cfg": vars(a), "state": model.state_dict()}, out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break
        if a.max_min and (time.time() - t_start) / 60 > a.max_min:
            print("time cap reached"); break
    json.dump({"args": vars(a), "log": log, "best_val_players_in_crop": best}, open(out / "train_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
