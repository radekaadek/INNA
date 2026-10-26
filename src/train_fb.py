"""Train a FootAndBall-style joint ball+player network on SN-Tracking-2023 (Adam, lr 1e-3, step decay x0.1 at 75% of the epochs)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sn_fb as FB
import sn_lib as S

RES = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--backbone", default="tiny", choices=["tiny", "r18"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--per-epoch", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-min", type=float, default=30)
    ap.add_argument("--two-frame", action="store_true", help="input = [frame, frame - previous sampled frame]")
    ap.add_argument("--hardneg", default=None, help="npz from mine_negatives.py: crops centred on mined false positives")
    ap.add_argument("--p-hn", type=float, default=0.25)
    ap.add_argument("--bb-lr-mult", type=float, default=1.0, help="backbone lr = lr * this (new layers keep lr)")
    a = ap.parse_args()
    out = RES / a.run; out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    torch.manual_seed(0); torch.backends.cudnn.benchmark = True
    tr, va = S.load("train"), S.load("valid")
    vsub = np.arange(0, len(va["names"]), 2)
    vd = {k: (v[vsub] if k in ("ball", "seq", "names", "boxes", "center") else v) for k, v in va.items()}
    vprev = [FB.prev_names(va)[i] for i in vsub] if a.two_frame else None
    hn = dict(np.load(a.hardneg)) if a.hardneg else None
    model = FB.FBNet(a.backbone, in_ch=6 if a.two_frame else 3).cuda()
    print("params M", sum(p.numel() for p in model.parameters()) / 1e6, flush=True)
    opt = torch.optim.Adam([{"params": list(model.backbone_parameters()), "lr": a.lr * a.bb_lr_mult}, {"params": model.head_parameters(), "lr": a.lr}])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(a.epochs * 0.75)], 0.1)
    log, best, t_start, start = [], -1, time.time(), 0
    last = out / "last.pt"
    if last.exists():  # resume after an interruption
        ck = torch.load(last, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); log, best, start = ck["log"], ck["best"], ck["epoch"]
        print("resumed from epoch", start, flush=True)
    for ep in range(start, a.epochs):
        t0 = time.time(); model.train(); tot = np.zeros(3); n = 0
        dl = torch.utils.data.DataLoader(FB.FBCrops(tr, "train", a.per_epoch, seed=ep + 1, two_frame=a.two_frame, hardneg=hn, p_hn=a.p_hn), batch_size=a.bs, num_workers=8, drop_last=True)
        for x, bl, pl in dl:
            x, bl, pl = x.cuda(non_blocking=True), bl.cuda(non_blocking=True), pl.cuda(non_blocking=True)
            loss, lb, lp = FB.fb_loss(model(x), bl, pl)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += (loss.item(), lb, lp); n += 1
        sched.step()
        cands, _ = FB.fb_outputs(model, vd["names"], "valid", prev=vprev)
        top1 = np.array([[c[0][0], c[0][1], c[0][2]] for c in cands])
        hit = S.ball_hit_rate(vd, top1, 0.02)["ball_recall"]
        rec = {"epoch": ep + 1, "loss": tot[0] / n, "ball_loss": tot[1] / n, "player_loss": tot[2] / n, "val_top1_hit": hit, "sec": time.time() - t0}
        log.append(rec); print(rec, flush=True)
        if hit > best:
            best = hit; torch.save({"backbone": a.backbone, "two_frame": a.two_frame, "state": model.state_dict()}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "log": log, "best": best, "epoch": ep + 1}, last)
        if (time.time() - t_start) / 60 > a.max_min:
            print("time cap"); break
    json.dump({"args": vars(a), "log": log, "best": best,
               "params_M": sum(p.numel() for p in model.parameters()) / 1e6}, open(out / "train_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
