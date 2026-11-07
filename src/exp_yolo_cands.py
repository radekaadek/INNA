"""Backlog #15: COCO-pretrained YOLO11n 'sports ball' detections as extra candidates next to ball_r18 heatmap peaks.
Step 1 (GPU): cache yolo candidates results/exp/cands_yolo.pkl. Step 2 (CPU): oracle + Viterbi with merged candidates, tuned on valid, test once. -> results/exp/yolo_cands.json"""
import itertools, json, pickle
import numpy as np
import sn_lib as S
from sm_data import ROOT
from exp_topk import oracle, viterbi, objective

RES = ROOT / "results"; SN = ROOT / "data" / "sn"


def detect(split, d):
    from ultralytics import YOLO
    m = YOLO(str(ROOT / "src" / "yolo11n.pt")); out = []
    paths = [str(SN / "frames" / split / n) for n in d["names"]]
    for i in range(0, len(paths), 32):
        for r in m.predict(paths[i:i + 32], imgsz=896, conf=0.02, classes=[32], verbose=False, device=0):
            h, w = r.orig_shape; b = r.boxes
            out.append(sorted([(float((x1 + x2) / 2 / w), float((y1 + y2) / 2 / h), float(c)) for (x1, y1, x2, y2), c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy())], key=lambda t: -t[2])[:10])
    return out


def merge(hm, yo, w, dup=0.01):
    res = []
    for h, y in zip(hm, yo):
        c = list(h)
        for x, yy, s in y:
            s = s * w
            near = [k for k, (cx, cy, cs) in enumerate(c) if np.hypot(cx - x, (cy - yy) * 0.5625) < dup]
            if near:  # agreeing detectors: boost the heatmap score
                k = near[0]; c[k] = (c[k][0], c[k][1], min(1.0, c[k][2] + 0.5 * s))
            else:
                c.append((x, yy, s))
        res.append(sorted(c, key=lambda t: -t[2]))
    return res


def main():
    va, te = S.load("valid"), S.load("test"); p = RES / "exp" / "cands_yolo.pkl"
    if not p.exists():
        pickle.dump({"valid": detect("valid", va), "test": detect("test", te)}, open(p, "wb"))
    y = pickle.load(open(p, "rb")); c = pickle.load(open(RES / "exp" / "cands_ball_r18.pkl", "rb")); res = {}
    res["oracle_yolo_valid"] = oracle(va, y["valid"], min_sc=0.02); res["oracle_yolo_test"] = oracle(te, y["test"], min_sc=0.02)
    print(res, flush=True)
    sc = []
    for w in (0.5, 1.0, 2.0):
        vc = S.filter_static(merge(c["valid"], y["valid"], w), va["seq"])
        for k in (5, 8):
            for thr in (0.1, 0.2):
                kw = dict(thr=thr, lam=1.0, miss=0.2, hold=4, topk=k)
                sc.append((objective(va, viterbi(vc, va["seq"], **kw)), w, k, thr))
        print(w, sorted(sc)[-1], flush=True)
    b = max(sc); w, k, thr = b[1:]; kw = dict(thr=thr, lam=1.0, miss=0.2, hold=4, topk=k)
    vc = S.filter_static(merge(c["valid"], y["valid"], w), va["seq"]); tc = S.filter_static(merge(c["test"], y["test"], w), te["seq"])
    res["merged"] = {"params": dict(w=w, **kw), "oracle_test_info": oracle(te, tc, ks=(1, 3, 5, 10)), "valid": S.ball_hit_rate(va, viterbi(vc, va["seq"], **kw)), "test": S.ball_hit_rate(te, viterbi(tc, te["seq"], **kw)),
                     "test_causal": S.ball_hit_rate(te, viterbi(tc, te["seq"], lag=0, **kw))}
    print(json.dumps(res["merged"], default=str)); json.dump(res, open(RES / "exp" / "yolo_cands.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
