"""Model factory. All detectors output boxes in 640x640 pixel coordinates."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator, DefaultBoxGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models import MobileNet_V3_Large_Weights, ResNet18_Weights, mobilenet_v3_large, resnet18

IMG = 640


def build_ssdlite(pretrained=True):
    m = ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone="DEFAULT" if pretrained else None, num_classes=2,
        score_thresh=0.01, detections_per_img=20)
    # small objects: 640 input and anchors starting at ~13 px
    m.transform = GeneralizedRCNNTransform(IMG, IMG, [0.5] * 3, [0.5] * 3, size_divisible=1, fixed_size=(IMG, IMG))
    m.anchor_generator = DefaultBoxGenerator([[2, 3]] * 6, scales=[0.02, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0])
    return m


def build_frcnn(pretrained=True):
    m = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None,
        weights_backbone=None if pretrained else "DEFAULT",
        min_size=IMG, max_size=IMG, box_detections_per_img=20, box_score_thresh=0.01)
    cin = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(cin, 2)
    # anchors matched to ~8x14 px balls (15 anchors per location, same as the pretrained head)
    m.rpn.anchor_generator = AnchorGenerator(((8, 12, 16, 24, 32),) * 3, ((0.5, 1.0, 2.0),) * 3)
    return m


class CenterNetR18(nn.Module):
    """CenterNet-style: ResNet-18 + light upsampling decoder (stride 4) -> heatmap + wh + offset."""

    def __init__(self, pretrained=True, backbone="resnet18", stride2=False):
        super().__init__()
        self.backbone, self.tta = backbone, False
        self.stride2 = stride2 and backbone == "mobilenet"  # extra stride-2 stage: 1/2-resolution heatmap for tiny balls
        self.stride = 2 if self.stride2 else 4
        if backbone == "resnet18":
            r = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
            self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
            chans = (64, 128, 256, 512)
        elif backbone.startswith("mobilenetv4"):
            import timm
            name = {"mobilenetv4_small": "mobilenetv4_conv_small.e2400_r224_in1k",
                    "mobilenetv4_medium": "mobilenetv4_conv_medium.e500_r256_in1k"}[backbone]
            self.tm = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
            chans = tuple(self.tm.feature_info.channels())
        else:  # mobilenet_v3_large taps at strides 4/8/16/32
            f = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None).features
            if self.stride2:
                self.mb = nn.ModuleList([f[:2], f[2:4], f[4:7], f[7:13], f[13:16]])
                chans = (16, 24, 40, 112, 160)
            else:
                self.mb = nn.ModuleList([f[:4], f[4:7], f[7:13], f[13:16]])
                chans = (24, 40, 112, 160)
        def cbr(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
        hc = 32 if self.stride2 else 64  # head width (narrower at 1/2 resolution to limit the cost)
        self.lat = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in (chans[1:] if self.stride2 else chans)])
        self.smooth = cbr(64, hc)
        if self.stride2:
            self.lat0 = nn.Conv2d(chans[0], hc, 1)
            self.smooth0 = cbr(hc, hc)
        def head(o, b=0.0):
            h = nn.Sequential(cbr(hc, hc), nn.Conv2d(hc, o, 1))
            return h
        self.hm, self.wh, self.off = head(1), head(2), head(2)
        nn.init.constant_(self.hm[-1].bias, -4.6)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward_maps(self, x):
        x = (x - self.mean) / self.std
        if self.backbone == "resnet18":
            c1 = self.l1(self.stem(x)); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        elif self.backbone.startswith("mobilenetv4"):
            c1, c2, c3, c4 = self.tm(x)
        else:
            if self.stride2:
                c0 = self.mb[0](x); c1 = self.mb[1](c0); c2 = self.mb[2](c1); c3 = self.mb[3](c2); c4 = self.mb[4](c3)
            else:
                c1 = self.mb[0](x); c2 = self.mb[1](c1); c3 = self.mb[2](c2); c4 = self.mb[3](c3)
        p = self.lat[3](c4)
        for lat, c in ((self.lat[2], c3), (self.lat[1], c2), (self.lat[0], c1)):
            p = F.interpolate(p, size=c.shape[-2:], mode="nearest") + lat(c)
        p = self.smooth(p)
        if self.stride2:
            p = self.smooth0(F.interpolate(p, size=c0.shape[-2:], mode="nearest") + self.lat0(c0))
        return self.hm(p), self.wh(p), self.off(p)

    def targets(self, tg, hw, device):
        H, W = hw
        hm = torch.zeros(len(tg), 1, H, W, device=device)
        wh = torch.zeros(len(tg), 2, H, W, device=device)
        off = torch.zeros(len(tg), 2, H, W, device=device)
        mask = torch.zeros(len(tg), 1, H, W, device=device)
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        for i, t in enumerate(tg):
            for b in t["boxes"]:
                s = self.stride
                w, h = (b[2] - b[0]) / s, (b[3] - b[1]) / s
                cx, cy = (b[0] + b[2]) / (2 * s), (b[1] + b[3]) / (2 * s)
                ix, iy = int(cx.clamp(0, W - 1)), int(cy.clamp(0, H - 1))
                sigma = max(0.8, float((w * h).sqrt()) / 3)
                g = torch.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma ** 2))
                hm[i, 0] = torch.maximum(hm[i, 0], g)
                hm[i, 0, iy, ix] = 1
                wh[i, :, iy, ix] = torch.stack([w, h]); off[i, :, iy, ix] = torch.stack([cx - ix, cy - iy]); mask[i, 0, iy, ix] = 1
        return hm, wh, off, mask

    def loss(self, images, targets):
        x = torch.stack(images)
        hm_p, wh_p, off_p = self.forward_maps(x)
        hm_t, wh_t, off_t, mask = self.targets(targets, hm_p.shape[-2:], x.device)
        p = hm_p.sigmoid().clamp(1e-4, 1 - 1e-4)
        pos = hm_t.eq(1).float()
        npos = pos.sum().clamp(min=1)
        fl = -(pos * (1 - p) ** 2 * p.log() + (1 - pos) * (1 - hm_t) ** 4 * p ** 2 * (1 - p).log()).sum() / npos
        l1 = (F.l1_loss(wh_p, wh_t, reduction="none") * mask).sum() / npos
        l2 = (F.l1_loss(off_p, off_t, reduction="none") * mask).sum() / npos
        return {"hm": fl, "wh": 0.1 * l1, "off": l2}

    @torch.no_grad()
    def predict(self, images, k=20):
        x = torch.stack(images)
        hm, wh, off = self.forward_maps(x)
        hm = hm.sigmoid()
        if self.tta:  # horizontal-flip test-time augmentation: average heatmap and box size
            hf, wf, _ = self.forward_maps(x.flip(-1))
            hm = (hm + hf.sigmoid().flip(-1)) / 2
            wh = (wh + wf.flip(-1)) / 2
        keep = (F.max_pool2d(hm, 3, 1, 1) == hm).float() * hm
        out = []
        B, _, H, W = hm.shape
        for i in range(B):
            sc, idx = keep[i].flatten().topk(k)
            ys, xs = idx // W, idx % W
            s = self.stride
            cx = (xs + off[i, 0, ys, xs]) * s; cy = (ys + off[i, 1, ys, xs]) * s
            w = wh[i, 0, ys, xs].clamp(min=1) * s; h = wh[i, 1, ys, xs].clamp(min=1) * s
            out.append({"boxes": torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1),
                        "scores": sc, "labels": torch.ones(k, dtype=torch.int64)})
        return out


MODELS = {"ssdlite": build_ssdlite, "frcnn": build_frcnn, "centernet": CenterNetR18,
          "cn_mnv3": lambda pretrained=True: CenterNetR18(pretrained, backbone="mobilenet"),
          "cn_mnv3_s2": lambda pretrained=True: CenterNetR18(pretrained, backbone="mobilenet", stride2=True),
          "cn_mnv4s": lambda pretrained=True: CenterNetR18(pretrained, backbone="mobilenetv4_small"),
          "cn_mnv4m": lambda pretrained=True: CenterNetR18(pretrained, backbone="mobilenetv4_medium")}


class YoloWrap(nn.Module):
    """Adapter: Ultralytics YOLO weights -> the same predict() interface as the other detectors."""

    def __init__(self, path):
        super().__init__()
        from ultralytics import YOLO
        self.net = YOLO(path).model.float().eval()
        self.tta = False

    @torch.no_grad()
    def predict(self, images):
        from ultralytics.utils.nms import non_max_suppression
        out = self.net(torch.stack(images))
        out = out[0] if isinstance(out, (tuple, list)) else out
        if out.shape[-1] == 6:  # end-to-end head (YOLO26): (B, N, [x1,y1,x2,y2,conf,cls]), no NMS needed
            res = [o[o[:, 4].topk(min(20, len(o)))[1]] for o in out]
        else:
            res = non_max_suppression(out, 0.001, 0.6, max_det=20)
        return [{"boxes": r[:, :4], "scores": r[:, 4], "labels": torch.ones(len(r), dtype=torch.int64)} for r in res]


def build(name, pretrained=True):
    return MODELS[name](pretrained=pretrained)


def compute_loss(model, images, targets):
    if isinstance(model, CenterNetR18):
        return model.loss(images, targets)
    return model(list(images), [{k: v for k, v in t.items() if k != "image_id"} for t in targets])


@torch.no_grad()
def predict(model, images):
    model.eval()
    if isinstance(model, (CenterNetR18, YoloWrap)):
        return model.predict(images)
    return model(list(images))
