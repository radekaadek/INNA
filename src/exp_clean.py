"""Backlog #14: clean training labels. Flag labelled-ball training frames where neither the ball_r18 top-3 candidates nor a YOLO11n 'sports ball' detection lies within
0.03 (frame width) of the label ('suspect'); save results/exp/train_suspect.npy (bool per train frame). Report counts."""
import numpy as np, torch
import sn_lib as S
from sm_data import ROOT
from exp_topk import gt_xy

SN = ROOT / "data" / "sn"; RES = ROOT / "results"


def main():
    d = S.load("train")
    m = S.BallNet(pretrained=False).cuda(); m.load_state_dict(torch.load(RES / "ball_r18" / "best.pt")["state"]); m.eval()
    c = S.ball_candidates(m, d["names"], "train", topk=3)
    from ultralytics import YOLO
    y = YOLO(str(ROOT / "src" / "yolo11n.pt")); paths = [str(SN / "frames" / "train" / n) for n in d["names"]]; yc = []
    for i in range(0, len(paths), 32):
        for r in y.predict(paths[i:i + 32], imgsz=896, conf=0.05, classes=[32], verbose=False, device=0):
            h, w = r.orig_shape
            yc.append([((x1 + x2) / 2 / w, (y1 + y2) / 2 / h) for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy()])
    gx, gy = gt_xy(d); vis = d["ball"][:, 0] >= 0
    near = lambda pts, i: any(np.hypot(px - gx[i], (py - gy[i]) * 0.5625) <= 0.03 for px, py in pts)
    agree_h = np.array([near([(a, b) for a, b, s in c[i]], i) if vis[i] else True for i in range(len(vis))])
    agree_y = np.array([near(yc[i], i) if vis[i] else True for i in range(len(vis))])
    sus = vis & ~agree_h & ~agree_y
    print("train frames", len(vis), "ball-labelled", int(vis.sum()), "heatmap-agree", int((agree_h & vis).sum()), "yolo-agree", int((agree_y & vis).sum()), "suspect", int(sus.sum()))
    np.save(RES / "exp" / "train_suspect.npy", sus)


if __name__ == "__main__":
    main()
