"""Backlog #5: player-aware candidate scoring. YOLO11n 'person' boxes (COCO, no training) on the 896x504 frames; candidates far from every person get their
score reduced: s' = s - beta * min(1, max(0, dist - R) / R), dist = distance (frame-width units, y*0.5625) from the candidate to the nearest person box.
beta, R tuned on valid, Viterbi as before, test once. -> results/exp/players.json, cache cands_person.pkl"""
import itertools, json, pickle
import numpy as np
import sn_lib as S
from sm_data import ROOT
from exp_topk import viterbi, objective, oracle

RES = ROOT / "results" / "exp"; SN = ROOT / "data" / "sn"


def detect(split, d):
    from ultralytics import YOLO
    m = YOLO(str(ROOT / "src" / "yolo11n.pt")); out = []
    paths = [str(SN / "frames" / split / n) for n in d["names"]]
    for i in range(0, len(paths), 32):
        for r in m.predict(paths[i:i + 32], imgsz=896, conf=0.15, classes=[0], verbose=False, device=0):
            h, w = r.orig_shape
            out.append(np.array([[x1 / w, y1 / h, x2 / w, y2 / h] for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy()]).reshape(-1, 4))
    return out


def box_dist(x, y, b):
    if len(b) == 0:
        return 1.0
    dx = np.maximum(np.maximum(b[:, 0] - x, x - b[:, 2]), 0); dy = np.maximum(np.maximum(b[:, 1] - y, y - b[:, 3]), 0) * 0.5625
    return float(np.hypot(dx, dy).min())


def rescore(c, boxes, beta, R):
    return [sorted([(x, y, s - beta * min(1.0, max(0.0, box_dist(x, y, b) - R) / R)) for x, y, s in cf], key=lambda t: -t[2]) for cf, b in zip(c, boxes)]


def main():
    va, te = S.load("valid"), S.load("test"); p = RES / "cands_person.pkl"
    if not p.exists():
        pickle.dump({"valid": detect("valid", va), "test": detect("test", te)}, open(p, "wb"))
    P = pickle.load(open(p, "rb")); cd = pickle.load(open(RES / "cands_ball_r18.pkl", "rb")); kw0 = dict(thr=0.1, lam=1.0, miss=0.2, hold=4); sc = []
    print("baseline valid obj", objective(va, viterbi(S.filter_static(cd["valid"], va["seq"]), va["seq"], **kw0)), flush=True)
    for beta, R in itertools.product((0.1, 0.2, 0.4), (0.03, 0.06, 0.12)):
        vc = S.filter_static(rescore(cd["valid"], P["valid"], beta, R), va["seq"])
        sc.append((objective(va, viterbi(vc, va["seq"], **kw0)), beta, R)); print(sc[-1], flush=True)
    b = max(sc); beta, R = b[1:]; res = {"params": dict(beta=beta, R=R)}
    tc = S.filter_static(rescore(cd["test"], P["test"], beta, R), te["seq"])
    res["oracle_test"] = oracle(te, tc, ks=(1, 3, 10))
    res["test"] = {n: S.ball_hit_rate(te, viterbi(tc, te["seq"], lag=l, **kw0)) for n, l in (("offline", None), ("lag4", 4), ("causal", 0))}
    print(json.dumps(res, default=str)); json.dump(res, open(RES / "players.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
