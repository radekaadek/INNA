"""#19b/#20: ball-class candidates from a fine-tuned YOLO (class 1), oracle top-K, ByteTrack-style (exp_mot), Viterbi alone and merged with ball_r18
(tuned on valid, test once). Usage: exp_yolo_eval.py <name> [weights.pt] [imgsz] -> results/exp/yolo2_<name>.json"""
import itertools, json, pickle, sys
import numpy as np
import sn_lib as S
from sm_data import ROOT
from exp_topk import oracle, viterbi, objective
from exp_mot import bytetrack
from exp_yolo_cands import merge

RES = ROOT / "results"; SN = ROOT / "data" / "sn"


def detect(w, split, d, imgsz):
    from ultralytics import YOLO
    m = YOLO(w); out = []; paths = [str(SN / "frames" / split / n) for n in d["names"]]
    for i in range(0, len(paths), 16):
        for r in m.predict(paths[i:i + 16], imgsz=imgsz, conf=0.01, classes=[1], verbose=False, device=0):
            h, ww = r.orig_shape; b = r.boxes
            out.append(sorted([(float((x1 + x2) / 2 / ww), float((y1 + y2) / 2 / h), float(c)) for (x1, y1, x2, y2), c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy())], key=lambda t: -t[2])[:10])
    return out


def tune_viterbi(va, vc):
    grid = list(itertools.product([0.02, 0.05, 0.1, 0.2], [0.3, 0.6, 1.0], [0.1, 0.2, 0.3], [4, 8, 16]))
    return max(((objective(va, viterbi(vc, va["seq"], thr=a, lam=b, miss=m_, hold=h)), dict(thr=a, lam=b, miss=m_, hold=h)) for a, b, m_, h in grid), key=lambda t: t[0])


def report(va, te, vc, tc, kw):
    r = {"params": kw, "valid": S.ball_hit_rate(va, viterbi(vc, va["seq"], **kw))}
    for n, l in (("offline", None), ("lag4", 4), ("causal", 0)):
        r[n] = S.ball_hit_rate(te, viterbi(tc, te["seq"], lag=l, **kw))
    return r


def main(name, w=None, imgsz=896):
    w = w or str(RES / "yolo_sn2" / name / "weights" / "best.pt")
    va, te = S.load("valid"), S.load("test"); p = RES / "exp" / f"cands_{name}.pkl"
    if not p.exists():
        pickle.dump({"valid": detect(w, "valid", va, imgsz), "test": detect(w, "test", te, imgsz)}, open(p, "wb"))
    y = pickle.load(open(p, "rb")); res = {"weights": w, "imgsz": imgsz}
    res["oracle_valid"] = oracle(va, y["valid"], min_sc=0.01); res["oracle_test"] = oracle(te, y["test"], min_sc=0.01); print(res, flush=True)
    vc = S.filter_static(y["valid"], va["seq"]); tc = S.filter_static(y["test"], te["seq"])
    b = max((objective(va, bytetrack(vc, va["seq"], hi=h, lo=l, gate=g, hold=o, kf=False)), h, l, g, o) for h, l, g, o in itertools.product([0.15, 0.3, 0.5], [0.02, 0.05], [0.1, 0.2, 0.3], [4, 8, 16]))
    kw = dict(hi=b[1], lo=b[2], gate=b[3], hold=b[4]); res["bytetrack"] = {"params": kw, "test": S.ball_hit_rate(te, bytetrack(tc, te["seq"], **kw))}
    res["greedy_test"] = S.ball_hit_rate(te, S.track_ball(tc, te["seq"], thr=0.05, hold=16))
    res["viterbi_alone"] = report(va, te, vc, tc, tune_viterbi(va, vc)[1]); print(res["viterbi_alone"], flush=True)
    c = pickle.load(open(RES / "exp" / "cands_ball_r18.pkl", "rb")); best = None
    for wt in (0.5, 1.0):
        mv = S.filter_static(merge(c["valid"], y["valid"], wt), va["seq"]); o, kw = tune_viterbi(va, mv)
        if best is None or o > best[0]: best = (o, wt, kw)
    mt = S.filter_static(merge(c["test"], y["test"], best[1]), te["seq"])
    res["merged_oracle_test"] = oracle(te, mt); res["merged"] = dict(weight=best[1], **report(va, te, S.filter_static(merge(c["valid"], y["valid"], best[1]), va["seq"]), mt, best[2]))
    print(json.dumps(res, default=str, indent=1)); json.dump(res, open(RES / "exp" / f"yolo2_{name}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, int(sys.argv[3]) if len(sys.argv) > 3 else 896)
