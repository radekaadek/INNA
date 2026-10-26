"""Fine-tuning of ball detectors on CPU with generalisation options."""
import argparse
import json
import random
import time
from pathlib import Path

import torch

import models
from data import EXTRA, SN, SN2, BallDataset, collate, epoch_subset, load_split
from evaluate import coco_ap, run_inference

RES = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--run", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--strong-aug", action="store_true")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay (0=off)")
    ap.add_argument("--cpu", action="store_true", help="allow CPU training (default: require CUDA)")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-pretrain", action="store_true")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-train-imgs", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", action="store_true", help="add the extra broadcast dataset (extra_data.py) to train/valid")
    ap.add_argument("--sn", action="store_true", help="add SoccerNet SN-Tracking-2023 views (sn_data.py) to train/valid")
    ap.add_argument("--sn2", action="store_true", help="like --sn but train on the tile-heavy sn_data.py train2 set (valid stays ds/valid)")
    ap.add_argument("--epoch-cap", type=int, default=0, help="max images per epoch (keeps the training time fixed with a bigger pool)")
    ap.add_argument("--init", default=None, help="start from results/<init>/best.pt (fine-tune an existing run)")
    a = ap.parse_args()
    run = a.run or a.model
    out = RES / run; out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); random.seed(a.seed)
    rng = random.Random(a.seed)

    train, valid = load_split("train"), load_split("valid")
    if a.extra:
        off = lambda items: [(i + 1_000_000, p, b) for i, p, b in items]  # keep COCO image ids unique
        train += off(load_split("train", EXTRA)); valid += off(load_split("valid", EXTRA))
    if a.sn:
        off = lambda items: [(i + 2_000_000, p, b) for i, p, b in items]
        train += off(load_split("train", SN)); valid += off(load_split("valid", SN))
    if a.sn2:
        train = train * 3  # Roboflow x3 so the bigger SoccerNet pool does not crowd it out
        train += [(i + 2_000_000, p, b) for i, p, b in load_split("train", SN2)]
        valid += [(i + 2_000_000, p, b) for i, p, b in load_split("valid", SN)]
    if a.max_train_imgs:
        train = train[: a.max_train_imgs]; valid = valid[:64]
    if not a.cpu and not torch.cuda.is_available():
        raise SystemExit("CUDA not available (GPU hung?) - refusing to fall back to CPU; use --cpu to force")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, flush=True)
    model = models.build(a.model, pretrained=not a.no_pretrain).to(dev)
    if a.init:
        model.load_state_dict(torch.load(RES / a.init / "best.pt", map_location=dev)["state"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    steps_total = a.epochs * (min(len(epoch_subset(train, a.neg_ratio, random.Random(0))), a.epoch_cap or 10 ** 9) // a.bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps_total, pct_start=0.1)

    ema = None
    if a.ema:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True)
    log, best, bad, start = [], -1, 0, 0
    last = out / "last.pt"
    if last.exists():  # resume after a crash (the GPU occasionally hangs)
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        log, best, bad, start = ck["log"], ck["best"], ck["bad"], ck["epoch"]
        rng.setstate(ck["rng"]); print(f"resumed from epoch {start}", flush=True)
    for ep in range(start, a.epochs):
        t0 = time.time(); model.train()
        sub = epoch_subset(train, a.neg_ratio, rng)
        if a.epoch_cap:
            sub = sub[: a.epoch_cap]
        dl = torch.utils.data.DataLoader(BallDataset(sub, train=True, augment=not a.no_aug, strong=a.strong_aug), batch_size=a.bs,
                                         shuffle=True, collate_fn=collate, num_workers=3, drop_last=True)
        tot, n = 0.0, 0
        for imgs, tg in dl:
            imgs = [i.to(dev) for i in imgs]
            tg = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in x.items()} for x in tg]
            losses = models.compute_loss(model, imgs, tg)
            loss = sum(losses.values())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 10.0); opt.step()
            if sched.last_epoch < steps_total - 1:
                sched.step()
            if ema is not None:
                ema.update_parameters(model)
            tot += loss.item(); n += 1
        ev = ema.module if ema is not None else model
        ap50, ap = coco_ap(valid, run_inference(ev, valid))
        rec = {"epoch": ep + 1, "train_loss": tot / n, "val_ap50": ap50, "val_ap": ap, "sec": time.time() - t0}
        log.append(rec); print(rec, flush=True)
        if ap50 > best:
            best, bad = ap50, 0
            torch.save({"model": a.model, "pretrained": not a.no_pretrain, "state": ev.state_dict()}, out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None, "log": log, "best": best, "bad": bad,
                    "epoch": ep + 1, "rng": rng.getstate()}, last)
    json.dump({"args": vars(a), "log": log, "best_val_ap50": best}, open(out / "train_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
