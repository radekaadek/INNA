"""Stride-3 rebuild with a larger validation set (backlog #8, #9): train3 = first 45 official-train clips, valid3 = last 12 (the old 6 valid + 6 more),
both at stride 3 (= test gap, 0.12 s). Does not touch train/valid/test. Output data/sn/{train3,valid3}.npz + frames/{train3,valid3}."""
import sn_build as B

B.STRIDE.update({"train3": 3, "valid3": 3})
tr = B.clips_of(B.SN / "train.zip", "train")
B.build("train3", "train", tr[:-12], B.SN / "train.zip")
B.build("valid3", "train", tr[-12:], B.SN / "train.zip")
