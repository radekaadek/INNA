"""Convert the COCO splits in imgs/ to the YOLO layout (copies images, writes normalised labels)."""
import shutil
from pathlib import Path

from data import ROOT, load_split

OUT = Path(__file__).resolve().parent.parent / "results" / "yolo_data"


def main():
    for split in ("train", "valid", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for _, p, boxes in load_split(split):
            shutil.copy(p, OUT / "images" / split / p.name)
            lines = [f"0 {(b[0]+b[2])/1280:.6f} {(b[1]+b[3])/1280:.6f} {(b[2]-b[0])/640:.6f} {(b[3]-b[1])/640:.6f}" for b in boxes]
            (OUT / "labels" / split / (p.stem + ".txt")).write_text("\n".join(lines))
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/valid\nnames:\n  0: ball\n")


if __name__ == "__main__":
    main()
