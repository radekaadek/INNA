"""Evaluate player-based 9:16 cropping on TeamTrack soccer panoramas (zero-shot, models trained on broadcast footage) and render videos.
Usage: tt_eval.py eval | tt_eval.py render <players_method> [clips...]   (needs results/tt/<clip>.pkl from tt_extract.py)"""
import json
import pickle
import subprocess
import sys

import numpy as np
from PIL import Image

import sn_lib as S
from sm_data import ROOT
from tt_extract import DATASET, GRID, IH, IW, OUT, TT, read_frames

CLIPS = ["F_1680_1710", "F_1710_1740", "F_1740_1770"] if DATASET == "teamtrack" else ["F_20200220_1_0270_0300", "F_20200220_1_0300_0330", "F_20200220_1_0330_0360"]
WW = 9 / 16 * IH                 # 9:16 window at full height: 562.5 px of 6500
LO, HI = WW / 2, IW - WW / 2
POSTS = {"raw": dict(alpha=1.0), "ema_lim": dict(alpha=0.5, step=150.0)}  # step: max pan per sample (0.12 s) in px (not tuned)


def gt_centres(clip, n):
    g = np.loadtxt(TT / clip / "gt.txt", delimiter=",")
    per = {}
    for f, _, x, y, w, h, *_ in g:
        per.setdefault(int(f), []).append(x + w / 2)
    return [np.array(per.get(3 * k + 1, [])) for k in range(n)]  # sample k = original frame 3k (0-based) = MOT frame 3k+1


def gt_ball(clip, n):
    """(n, 2) ball centre per sample (nan when not annotated); only SoccerTrack has ball labels."""
    p = TT / clip / "ball.txt"
    if not p.exists():
        return None
    d = {}
    for line in p.read_text().split():
        f, x, y, *_ = line.split(","); d[int(f)] = (float(x), float(y))
    return np.array([d.get(3 * k + 1, (np.nan, np.nan)) for k in range(n)])


def best_window(cx, ball_x=None, margin=0.0):
    """Window centre (px) covering most `cx` (player centres); with `ball_x` only windows containing the ball (minus margin)."""
    cs = np.arange(LO, HI + 1, 10.0)
    if ball_x is not None:
        ok = np.abs(cs - ball_x) <= WW / 2 - margin
        cs = cs[ok] if ok.any() else np.array([min(max(ball_x, LO), HI)])
    if len(cx) == 0:
        return float(cs[np.argmin(np.abs(cs - (ball_x if ball_x is not None else IW / 2)))])
    cnt = np.array([(np.abs(cx - c) <= WW / 2).sum() for c in cs]); best = np.flatnonzero(cnt == cnt.max())
    ref = ball_x if ball_x is not None else np.median(cx)
    return float(cs[best[np.argmin(np.abs(cs[best] - ref))]])


def hist_window(h):
    k = int(round(WW / (IW / GRID))); cs = np.convolve(h, np.ones(k), mode="valid")
    return float(np.clip((np.argmax(cs) + k / 2) * IW / GRID, LO, HI))


def smooth(c, alpha, step=None):
    c = np.clip(np.asarray(c, float), LO, HI); out = c.copy()
    for i in range(1, len(c)):
        v = alpha * c[i] + (1 - alpha) * out[i - 1]
        if step is not None:
            v = out[i - 1] + np.clip(v - out[i - 1], -step, step)
        out[i] = v
    return out


def metrics(gtc, centres, gball=None):
    inside = np.mean([(np.abs(g - c) <= WW / 2).mean() for g, c in zip(gtc, centres)])
    orc = np.array([best_window(g) for g in gtc])
    jit = np.abs(np.diff(centres)).mean()
    m = {"players_in_crop": float(inside), "centre_mae_px": float(np.abs(np.asarray(centres) - orc).mean()), "jitter_px": float(jit)}
    if gball is not None:
        vis = ~np.isnan(gball[:, 0])
        m["ball_in_crop"] = float((np.abs(gball[vis, 0] - np.asarray(centres)[vis]) <= WW / 2).mean())
    return m


def static_filter(cands, cell=24.0, frac=0.4):
    cnt = {}
    for cs in cands:
        for c in {(int(x // cell), int(y // cell)) for x, y, s in cs if s >= 0.3}:
            cnt[c] = cnt.get(c, 0) + 1
    mask = {c for c, n in cnt.items() if n > frac * len(cands)}
    return [[(x, y, s) for x, y, s in cs if not any((int(x // cell) + dx, int(y // cell) + dy) in mask for dx in (-1, 0, 1) for dy in (-1, 0, 1))] for cs in cands]


def track(cands, thr=0.3, hold=8, gate=0.25):
    """S.track_ball on pseudo-normalised coordinates (u = x/1000, v = y/1000/0.5625, so 1 unit = 1000 px in both axes)."""
    pc = [[(x / 1000, y / 1000 / 0.5625, s) for x, y, s in cs] for cs in static_filter(cands)]
    e = S.track_ball(pc, np.zeros(len(pc), int), thr=thr, gate=gate, hold=hold)
    return np.stack([e[:, 0] * 1000, e[:, 1] * 1000 * 0.5625, e[:, 2]], 1)


def centres_for(rec, players="yolo_sn", ball="ball_r18"):
    """Per-sample window centres of every method for one clip (px)."""
    out = {}
    cx = lambda key: [((b[:, 0] + b[:, 2]) / 2) if len(b) else np.zeros(0) for b in rec[key]]
    for k in ("yolo_sn", "yolo_sm", "yolo_coco"):
        c, prev = [], IW / 2
        for x in cx(k):
            prev = best_window(x) if len(x) else prev; c.append(prev)
        out[k] = np.array(c)
    out["hist_r18"] = np.array([hist_window(h) for h in rec["hist_r18"]])
    out["hist_fb"] = np.array([hist_window(h) for h in rec["hist_fb"]])
    px = cx(players)
    for tag, src in (("", ball), ("fb_", "ball_fb")):
        e_ = track(rec[src])
        out[tag + "ball_only"] = np.where(e_[:, 2] > 0, e_[:, 0], IW / 2)
        out[tag + "ball_plus_players"] = np.array([best_window(p, e[0] if e[2] > 0 else None, 0.1 * WW) for p, e in zip(px, e_)])
        if tag == "":
            est = e_
    return out, est


def evaluate():
    res = {}
    allm = {}
    for clip in CLIPS:
        rec = pickle.load(open(OUT / f"{clip}.pkl", "rb")); n = len(rec["yolo_sn"]); gtc = gt_centres(clip, n)
        cs, est = centres_for(rec)
        gb = gt_ball(clip, n)
        if gb is not None:
            bx = np.where(np.isnan(gb[:, 0]), IW / 2, gb[:, 0])
            cs["gt_ball_only"] = bx; cs["gt_ball_players"] = np.array([best_window(g, b, 0.1 * WW) for g, b in zip(gtc, bx)])
            stats = {}
            for src in ("ball_r18", "ball_fb"):
                tr = track(rec[src]); vis = ~np.isnan(gb[:, 0])
                top1 = np.array([[c[0][0], c[0][1]] if c else [np.nan, np.nan] for c in rec[src]])
                d_top = np.hypot(*(top1 - gb).T); d_tr = np.hypot(tr[:, 0] - gb[:, 0], tr[:, 1] - gb[:, 1])
                stats[src] = {"top1_recall_50px": float((d_top[vis] <= 50).mean()), "tracked_recall_50px": float(((d_tr <= 50) & (tr[:, 2] > 0))[vis].mean()),
                              "tracked_precision_50px": float(((d_tr <= 50) & (tr[:, 2] > 0)).sum() / max((tr[:, 2] > 0).sum(), 1)),
                              "recall_any_of_top5_50px": float(np.mean([any(np.hypot(x - gb[k, 0], y - gb[k, 1]) <= 50 for x, y, _ in rec[src][k]) for k in np.flatnonzero(vis)]))}
            res_stats = stats
        cs["fixed_centre"] = np.full(n, IW / 2); cs["gt_median"] = np.array([np.median(g) for g in gtc]); cs["gt_oracle"] = np.array([best_window(g) for g in gtc])
        for k, c in cs.items():
            for p, kw in POSTS.items():
                c2 = smooth(c, kw["alpha"], kw.get("step"))
                allm.setdefault((k, p), []).append((metrics(gtc, c2, gb), n))
        res[clip] = {"ball_estimate_coverage": float((est[:, 2] > 0).mean()), **({"ball_detection": res_stats} if gb is not None else {})}
    res["mean"] = {f"{k}|{p}": {m: float(np.average([v[0][m] for v in vs], weights=[v[1] for v in vs])) for m in vs[0][0]} for (k, p), vs in allm.items()}
    json.dump(res, open(ROOT / "results" / ("tt_eval.json" if DATASET == "teamtrack" else "st_eval.json"), "w"), indent=1)
    print(f"{'method':22s} {'raw in-crop/jit':>18s} {'smoothed in-crop/jit':>22s}   (window = {WW / IW * 100:.1f}% of the width)")
    names = ["fixed_centre", "gt_median", "gt_oracle", "gt_ball_only", "gt_ball_players", "yolo_sn", "yolo_sm", "yolo_coco", "hist_r18", "hist_fb",
             "ball_only", "ball_plus_players", "fb_ball_only", "fb_ball_plus_players"]
    for k in names:
        if f"{k}|raw" not in res["mean"]:
            continue
        a, b = res["mean"][f"{k}|raw"], res["mean"][f"{k}|ema_lim"]
        bb = f"  ball-in-crop raw {100 * a['ball_in_crop']:5.1f}% smooth {100 * b['ball_in_crop']:5.1f}%" if "ball_in_crop" in a else ""
        print(f"{k:22s} {100 * a['players_in_crop']:7.1f}% /{a['jitter_px']:6.1f}px {100 * b['players_in_crop']:12.1f}% /{b['jitter_px']:6.1f}px   mae(raw) {a['centre_mae_px']:.0f}px{bb}")
    for c in CLIPS:
        if "ball_detection" in res[c]:
            print(c, json.dumps({s: {k: round(v, 3) for k, v in d.items()} for s, d in res[c]["ball_detection"].items()}))
    print("ball estimate coverage per clip:", {c: round(res[c]["ball_estimate_coverage"], 2) for c in CLIPS})


def render(method, clips):
    OW, OHH, PW, PH = 540, 960, 1624, 250
    outs = {}
    for tag in ("players", "ball"):
        outs[tag] = (subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OHH}", "-r", "25", "-i", "-", "-c:v", "libx264",
                                       "-crf", "20", "-pix_fmt", "yuv420p", str(OUT / f"tt_{tag}_vertical.mp4")], stdin=subprocess.PIPE),
                     subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{PW}x{PH}", "-r", "25", "-i", "-", "-c:v", "libx264",
                                       "-crf", "20", "-pix_fmt", "yuv420p", str(OUT / f"tt_{tag}_preview.mp4")], stdin=subprocess.PIPE))
    for clip in clips:
        rec = pickle.load(open(OUT / f"{clip}.pkl", "rb")); n = len(rec["yolo_sn"])
        cs, est = centres_for(rec, players=method)
        full = {}
        for tag, key in (("players", method), ("ball", "ball_plus_players")):
            c = smooth(cs[key], **{"alpha": POSTS["ema_lim"]["alpha"], "step": POSTS["ema_lim"]["step"]})
            full[tag] = np.interp(np.arange(750) / 3, np.arange(n), c)
        bx = np.interp(np.arange(750) / 3, np.arange(n), est[:, 0]); by = np.interp(np.arange(750) / 3, np.arange(n), est[:, 1])
        bok = np.interp(np.arange(750) / 3, np.arange(n), est[:, 2]) > 0.05
        for t, fr in enumerate(read_frames(clip, stride=1)):
            small = np.asarray(Image.fromarray(fr).resize((PW, PH), Image.BILINEAR))
            for tag in ("players", "ball"):
                x0 = int(np.clip(full[tag][t] - WW / 2, 0, IW - WW))
                outs[tag][0].stdin.write(np.asarray(Image.fromarray(fr[:, x0:x0 + int(WW)]).resize((OW, OHH), Image.LANCZOS)).tobytes())
                p = small.copy(); a = int(x0 / IW * PW); b = int((x0 + WW) / IW * PW)
                p[:, a:b][[0, -1], :] = (255, 0, 0); p[:, [a, min(b, PW - 1)]] = (255, 0, 0)
                if tag == "ball" and bok[t]:
                    u, v = int(bx[t] / IW * PW), int(by[t] / IH * PH); p[max(v - 5, 0):v + 5, max(u - 5, 0):u + 5][[0, -1], :] = (255, 255, 0); p[max(v - 5, 0):v + 5, max(u - 5, 0):u + 5][:, [0, -1]] = (255, 255, 0)
                outs[tag][1].stdin.write(p.tobytes())
        print(clip, "rendered", flush=True)
    for pv in outs.values():
        for q in pv:
            q.stdin.close(); q.wait()


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate()
    else:
        render(sys.argv[2], sys.argv[3:] or CLIPS)
