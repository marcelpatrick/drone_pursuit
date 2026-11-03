# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\check_capture.py
# ── NEW file — the whole content, nothing to insert into.
# ── Environment: drone_vision
# ── Checks the arena frames made by the new capture_arena_frames.py:
# ──   1. RESTART BUG: does every picture agree with its readings?
# ──      - no two rows with identical readings (the old bug repeated the chase's start position)
# ──      - in view (v1): the drone's real pixels (green box) sit where the readings say (red box)
# ──      - in view (v1): the picture really contains the drone
# ──   2. PICTURE QUALITY: average sharpness of the new pictures, and of an old folder if given.
# ──      Sharper = higher number. Smeared, doubled grid lines lower it.
# ── Usage:
# ──   python check_capture.py                                   (restart check + new sharpness)
# ──   python check_capture.py --old data\arena_frames_old       (also compares with the old pictures)
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(r"C:\projects\drone_pursuit\drone_pursuit")
parser = argparse.ArgumentParser()
parser.add_argument("--new", type=str, default=r"data\arena_frames")
parser.add_argument("--old", type=str, default=None)      # optional: the earlier capture, for the sharpness comparison
args = parser.parse_args()
NEW = BASE / args.new
W, H = 640, 480

with open(NEW / "readings.csv", newline="") as f:
    rows = list(csv.DictReader(f))
if "gradeable" not in rows[0]:
    raise SystemExit("STOP: readings.csv has no 'gradeable' column, so these frames came from the OLD capture script")

# ── 1a. identical readings ────────────────────────────────────────────────────
print("1. RESTART BUG")
keys = Counter((r["dist_m"], r["bx"], r["by"]) for r in rows)
repeats = [k for k, n in keys.items() if n > 1]
print(f"   identical readings repeated: {len(repeats)}   (want 0)")
for k in repeats:
    names = [r["name"] for r in rows if (r["dist_m"], r["bx"], r["by"]) == k]
    print(f"      {', '.join(names)}")

# ── 1b/1c. picture vs readings ────────────────────────────────────────────────
far_apart, missing = [], []
for r in rows:
    if r["visible"] != "1":
        continue
    if int(r["drone_px"]) == 0:                            # readings say "in view", picture has no drone
        missing.append(r["name"])
        continue
    xc = W / 2 + float(r["bx"]) * W / 2                    # red box centre: where the readings put the drone
    yc = H / 2 + float(r["by"]) * H / 2
    gx, gy = float(r["box_x"]), float(r["box_y"])          # green box centre: where its pixels really are
    gap = max(abs(gx - xc), abs(gy - yc))
    if gap > max(20.0, float(r["box_w"]), float(r["box_h"])):
        far_apart.append((r["name"], gap))
n_v1 = sum(r["visible"] == "1" for r in rows)
print(f"   in-view frames: {n_v1}")
print(f"   red and green boxes far apart: {len(far_apart)}   (want 0)")
for name, gap in far_apart:
    print(f"      {name}: {gap:.0f} px apart")
print(f"   readings say in view, but no drone in the picture: {len(missing)}   (want 0)")
for name in missing:
    print(f"      {name}")
print(f"   gradeable (at least 10 x 4 px): {sum(r['gradeable'] == '1' for r in rows)} of {n_v1}")


# ── 2. sharpness ──────────────────────────────────────────────────────────────
def sharpness(folder):
    """Average detail per picture: how strongly brightness changes between neighbouring pixels."""
    scores = []
    for png in sorted(folder.glob("frame_*.png")):
        g = np.asarray(Image.open(png).convert("L"), dtype=np.float32)
        lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        scores.append(lap.var())
    return (float(np.mean(scores)), len(scores)) if scores else (0.0, 0)


print("\n2. PICTURE QUALITY (sharpness, higher = sharper)")
s_new, n_new = sharpness(NEW)
print(f"   new: {s_new:8.1f}   ({n_new} pictures)")
if args.old:
    s_old, n_old = sharpness(BASE / args.old)
    print(f"   old: {s_old:8.1f}   ({n_old} pictures)")
    if s_old:
        print(f"   change: {100 * (s_new - s_old) / s_old:+.0f}%   (want clearly positive)")