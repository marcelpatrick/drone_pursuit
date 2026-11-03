# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\score_arena_frames.py
# ── The whole content (replaces the earlier version; diagnose_arena_misses.py is no longer needed).
# ── Environment: drone_vision
# ── Reads: data\arena_frames\*.png and readings.csv (from the new capture_arena_frames.py), best.pt (5.1)
# ── What it does:
# ──   - grades only frames where the attacker is in view AND covers at least 10 x 4 px
# ──     ("gradeable", the same limit convert_to_yolo.py uses for training labels)
# ──   - counts a frame as FOUND only if a box sits on the attacker's real pixels
# ──   - checks out-of-view frames (v0) for false alarms
# ──   - shows what lower thresholds would give, and the real confidence of every miss
import csv
from pathlib import Path
from ultralytics import YOLO

FRAMES = Path(r"C:\projects\drone_pursuit\drone_pursuit\data\arena_frames")
MODEL = r"C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v4\weights\best.pt"   # ← your model
CONF = 0.4                                    # the threshold vision_bridge.py uses
THRESHOLDS = [0.4, 0.3, 0.25, 0.2]
BANDS = [(0, 2), (2, 5), (5, 99)]             # distance ranges in metres
LOW = 0.05                                    # keep almost every box, to read real confidences

model = YOLO(MODEL)
graded, empty, too_small = [], [], 0
with open(FRAMES / "readings.csv", newline="") as f:
    for r in csv.DictReader(f):
        res = model.predict(str(FRAMES / r["name"]), conf=LOW, verbose=False)[0]
        boxes = res.boxes.xywh.cpu().numpy()          # centre x, centre y, width, height (pixels)
        confs = res.boxes.conf.cpu().numpy()
        if r["visible"] != "1":                       # attacker out of view: any box is a false alarm
            empty.append(float(confs.max()) if len(confs) else 0.0)
            continue
        if r["gradeable"] != "1":                     # in view, but too few pixels for any detector
            too_small += 1
            continue
        cx, cy = float(r["box_x"]), float(r["box_y"])
        tol = max(15.0, float(r["box_w"]), float(r["box_h"]))
        on, off = 0.0, 0.0                            # best confidence ON the attacker / ELSEWHERE
        for (x, y, _, _), c in zip(boxes, confs):
            if abs(x - cx) <= tol and abs(y - cy) <= tol:
                on = max(on, float(c))
            else:
                off = max(off, float(c))
        graded.append(dict(name=r["name"], dist=float(r["dist_m"]), w=int(r["box_w"]), h=int(r["box_h"]),
                           bx=float(r["bx"]), by=float(r["by"]), on=on, off=off))

print(f"MISSED AT {CONF} (best confidence on the attacker / elsewhere)")
for g in graded:
    if g["on"] < CONF:
        note = "   <- box in the WRONG place" if g["off"] >= CONF else ""
        print(f"  {g['name']}: on {g['on']:.2f}  elsewhere {g['off']:.2f}   drone {g['w']}x{g['h']} px   "
              f"x {g['bx']:+.2f} y {g['by']:+.2f}{note}")

print(f"\nin view but too small to grade (under 10 x 4 px): {too_small}")
print(f"out-of-view frames: {len(empty)}, false alarms at {CONF}: {sum(c >= CONF for c in empty)}")

print(f"\nAT {CONF}:   distance   graded   found   rate")
for lo, hi in BANDS:
    band = [g for g in graded if lo <= g["dist"] < hi]
    if band:
        hit = sum(g["on"] >= CONF for g in band)
        print(f"            {lo:>2}–{hi:<2} m   {len(band):>6}   {hit:>5}   {hit / len(band):4.0%}")

print("\nthreshold   0–2 m found     2–5 m found     false alarms (v0)")
for t in THRESHOLDS:
    cells = []
    for lo, hi in BANDS[:2]:
        band = [g for g in graded if lo <= g["dist"] < hi]
        hit = sum(g["on"] >= t for g in band)
        cells.append(f"{hit:>3}/{len(band):<3} {hit / max(len(band), 1):4.0%}")
    print(f"   {t:.2f}     " + "    ".join(cells) + f"     {sum(c >= t for c in empty):>3} of {len(empty)}")