"""9:16 crop window from the original 16:9 frame (images are stretched to 640x640)."""
CROP_W_FRAC = (9 / 16) / (16 / 9)  # window width as fraction of frame width (~0.316)


def window(cx, W=640, frac=CROP_W_FRAC):
    cw = W * frac
    x0 = min(max(cx - cw / 2, 0), W - cw)
    return x0, x0 + cw


def choose_center(dets, thr, W=640, prev=None, alpha=1.0):
    """Center of best detection with score>=thr; else previous center (or frame center).
    alpha<1 gives exponential smoothing for video streams."""
    best = max(dets, key=lambda d: d[4], default=None)
    if best is not None and best[4] >= thr:
        c = (best[0] + best[2]) / 2
    else:
        c = prev if prev is not None else W / 2
    return c if prev is None else alpha * c + (1 - alpha) * prev


def ball_in_crop(gt_box, cx, W=640):
    x0, x1 = window(cx, W)
    g = (gt_box[0] + gt_box[2]) / 2
    return x0 <= g <= x1
