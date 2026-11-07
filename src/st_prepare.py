"""SoccerTrack wide-view annotations (SportsLabKit csv) -> MOT-style gt.txt (players) + ball.txt (frame, cx, cy, w, h), per clip folder."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "soccertrack"
for clip in sys.argv[1:]:
    rows = list(csv.reader(open(ROOT / clip / "annot.csv")))
    team, pid, att, data = rows[0], rows[1], rows[2], rows[4:]  # rows[3] is the 'frame' label row
    groups = {}
    for i in range(1, len(team)):
        groups.setdefault((team[i], pid[i]), {})[att[i]] = i
    f = lambda r, i: float(r[i]) if r[i] not in ("", "nan") else None
    gt, ball = [], []
    for r in data:
        fr = int(float(r[0]))
        for k, (key, g) in enumerate(sorted(groups.items())):
            vals = [f(r, g[a]) for a in ("bb_left", "bb_top", "bb_width", "bb_height")]
            if None in vals:
                continue
            if key[0] == "BALL":
                ball.append(f"{fr},{vals[0] + vals[2] / 2:.1f},{vals[1] + vals[3] / 2:.1f},{vals[2]:.1f},{vals[3]:.1f}")
            else:
                gt.append(f"{fr},{k},{vals[0]:.1f},{vals[1]:.1f},{vals[2]:.1f},{vals[3]:.1f},1,-1,-1,-1")
    (ROOT / clip / "gt.txt").write_text("\n".join(gt) + "\n"); (ROOT / clip / "ball.txt").write_text("\n".join(ball) + "\n")
    print(clip, len(gt), "player boxes (", len(gt) / len(data), "per frame ),", len(ball), "ball boxes of", len(data), "frames")
