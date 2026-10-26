"""Dataset COCO (Roboflow) -> torchvision detection format, with augmentations."""
import json
import random
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent / "imgs"
IMG = 640


EXTRA = ROOT.parent / "results" / "extra" / "ds"  # built by extra_data.py
SN2 = ROOT.parent / "results" / "sn_tracking" / "ds2"  # tile-heavy train set (sn_data.py train2)
SNZ = ROOT.parent / "results" / "sn_tracking" / "dsz"  # zoomed SN test views (sn_data.py testz)
SN = ROOT.parent / "results" / "sn_tracking" / "ds"  # built by sn_data.py (SoccerNet SN-Tracking-2023)


def load_split(split, root=ROOT):
    ROOT = Path(root)
    d = json.load(open(ROOT / split / "_annotations.coco.json"))
    boxes = {}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        boxes.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
    items = [(im["id"], ROOT / split / im["file_name"], boxes.get(im["id"], [])) for im in d["images"]]
    return items


class BallDataset(Dataset):
    """Single-class ball detection. Label 1 = ball, empty targets = negatives."""

    def __init__(self, items, train=False, augment=True, strong=False):
        self.items, self.train, self.augment, self.strong = items, train, augment and train, strong

    def __len__(self):
        return len(self.items)

    def _augment(self, img, boxes):
        w, h = img.size
        if random.random() < 0.5:  # horizontal flip
            img = TF.hflip(img)
            boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]
        # photometric jitter: brightness / contrast / saturation / hue
        img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
        img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
        img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
        img = TF.adjust_hue(img, random.uniform(-0.03, 0.03))
        # random scale-crop (zoom in 1.0-1.5x), keep ball if present
        if random.random() < 0.5:
            s = random.uniform(1.0, 1.5)
            cw, ch = int(w / s), int(h / s)
            x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
            if boxes:  # keep the ball center inside the crop
                b = boxes[0]
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                lo, hi = max(0, int(cx - cw + 8)), min(w - cw, int(cx - 8))
                x0 = random.randint(lo, hi) if lo <= hi else min(max(int(cx - cw / 2), 0), w - cw)
                lo, hi = max(0, int(cy - ch + 8)), min(h - ch, int(cy - 8))
                y0 = random.randint(lo, hi) if lo <= hi else min(max(int(cy - ch / 2), 0), h - ch)
            x0, y0 = max(0, min(x0, w - cw)), max(0, min(y0, h - ch))
            img = TF.resized_crop(img, y0, x0, ch, cw, [h, w])
            nb = []
            for b in boxes:
                nb.append([(b[0] - x0) * s, (b[1] - y0) * s, (b[2] - x0) * s, (b[3] - y0) * s])
            boxes = [[max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])] for b in nb]
            boxes = [b for b in boxes if b[2] - b[0] > 2 and b[3] - b[1] > 2]
        return img, boxes

    def _strong(self, img, boxes):
        """Extra augmentation: zoom-out (wide shots), blur / noise (compression, motion), random grayscale."""
        w, h = img.size
        if random.random() < 0.4:  # zoom-out: shrink frame, paste on a grey canvas
            s = random.uniform(0.6, 0.9)
            nw, nh = int(w * s), int(h * s)
            ox, oy = random.randint(0, w - nw), random.randint(0, h - nh)
            canvas = Image.new("RGB", (w, h), (114, 114, 114))
            canvas.paste(img.resize((nw, nh)), (ox, oy))
            img = canvas
            boxes = [[b[0] * s + ox, b[1] * s + oy, b[2] * s + ox, b[3] * s + oy] for b in boxes]
        if random.random() < 0.3:
            img = TF.gaussian_blur(img, 3, random.uniform(0.3, 1.2))
        if random.random() < 0.1:
            img = TF.rgb_to_grayscale(img, 3)
        t = TF.to_tensor(img)
        if random.random() < 0.3:
            t = (t + torch.randn_like(t) * random.uniform(0.01, 0.04)).clamp(0, 1)
        return TF.to_pil_image(t), boxes

    def __getitem__(self, i):
        iid, path, boxes = self.items[i]
        img = Image.open(path).convert("RGB")
        boxes = [list(b) for b in boxes]
        if self.augment:
            img, boxes = self._augment(img, boxes)
            if self.strong:
                img, boxes = self._strong(img, boxes)
        t = TF.to_tensor(img)
        b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        return t, {"boxes": b, "labels": torch.ones(len(b), dtype=torch.int64), "image_id": iid}


def collate(batch):
    return tuple(zip(*batch))


def epoch_subset(items, neg_ratio, rng):
    """All positives + neg_ratio * #pos random negatives (resampled each epoch)."""
    pos = [it for it in items if it[2]]
    neg = [it for it in items if not it[2]]
    k = min(len(neg), int(neg_ratio * len(pos)))
    sub = pos + rng.sample(neg, k)
    rng.shuffle(sub)
    return sub
