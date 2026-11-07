"""StreamBall v2: same DFP / TAL / box head as streamball.py, but the detection score is trained like the proven ball_r18 recipe:
CenterNet penalty-reduced focal loss on a Gaussian heatmap (sigma 1.5 cells) at stride 4 only (v1 used hard 3x3 positives with sigmoid focal on two
levels and a cls*obj product: top-1 oracle 63% vs top-10 87%, i.e. poor ranking). Regression (IoU loss with TAL) is trained at the GT cell only.
Candidates = heatmap peaks (score = sigmoid(obj); the cls logit is pinned to +20). Usage identical to streamball.py; checkpoints get args['v2']=True."""
import sys

import torch
import torch.nn.functional as F

import sn_lib as S
import streamball as B


class StreamBall2(B.StreamBall):
    def forward(self, x_prev, x_cur):
        outs = super().forward(x_prev, x_cur)[:1]  # stride-4 level only
        cls, obj, reg = outs[0]
        return [(torch.full_like(cls, 20.0), obj, reg)]


def build_model(a, pretrained=True):
    return StreamBall2(pretrained, dfp=not a["no_dfp"])


def stream_loss2(outs, gt, ref, use_tal=True, reg_w=2.0):
    cls, obj, reg = outs[0]; B_, _, h, w = obj.shape; dev = obj.device
    hm = torch.zeros_like(obj); pos = []
    iou_ref = B.iou_xyxy(gt[:, 1:], ref[:, 1:])
    tal = B.tal_weights(iou_ref, ref[:, 0] > 0) if use_tal else torch.ones(B_, device=dev)
    for b in range(B_):
        if gt[b, 0] <= 0:
            continue
        cx, cy = ((gt[b, 1] + gt[b, 3]) / 2 / 4).item(), ((gt[b, 2] + gt[b, 4]) / 2 / 4).item()
        hm[b, 0] = S.gaussian_map(h, w, cx, cy).to(dev)
        if 0 <= int(cx) < w and 0 <= int(cy) < h:
            pos.append((b, int(cy) * w + int(cx)))
    l_hm = S.focal_loss(obj, hm)
    l_reg = torch.zeros((), device=dev)
    if pos:
        bi = torch.tensor([p[0] for p in pos], device=dev); ci = torch.tensor([p[1] for p in pos], device=dev)
        pb = B.decode_boxes(reg, 4)[bi, ci]
        l_i = 1 - B.iou_xyxy(pb, gt[bi, 1:]) ** 2
        wt = tal[bi]; wt = wt * l_i.detach().sum() / (wt * l_i.detach()).sum().clamp(min=1e-6)
        l_reg = (wt * l_i).sum() / len(pos)
    return l_hm + reg_w * l_reg, {"hm": float(l_hm), "reg": float(l_reg)}


if __name__ == "__main__":
    B.build_model, B.stream_loss = build_model, stream_loss2
    run = sys.argv[sys.argv.index("--run") + 1]
    B.main()
    for f in ("best.pt", "final.pt"):
        p = B.RES / run / f
        if p.exists():
            ck = torch.load(p, map_location="cpu"); ck["args"]["v2"] = True; torch.save(ck, p)
