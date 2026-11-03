# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\score_arena_frames.py
# ── The whole content (replaces the 5.2 Step 3 version).
# ── Reads: data\arena_frames\frame_XXX_dDD.DDm_vN.png and readings.csv (5.2 Step 2), best.pt (5.1)
# ── Scores the detector only on frames where the attacker is in view (v1), checks
# ──   out-of-view frames (v0) for false alarms, and prints where each visible attacker
# ──   was (off-centre?) and how big (pixels), to explain misses.
import csv
from pathlib import Path
from ultralytics import YOLO

FRAMES = Path(r"C:\projects\drone_pursuit\drone_pursuit\data\arena_frames")
MODEL = r"C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v3\weights\best.pt"
CONF = 0.4                                   # same threshold vision_bridge.py uses
BANDS = [(0, 2), (2, 5), (5, 99)]            # distance ranges in metres
W = 640                                      # camera image width in pixels

model = YOLO(MODEL)
rows, false_alarms, out_of_view = [], 0, 0
with open(FRAMES / "readings.csv", newline="") as f:
    for r in csv.DictReader(f):
        result = model.predict(str(FRAMES / r["name"]), conf=CONF, verbose=False)[0]
        found = len(result.boxes) > 0
        conf = float(result.boxes.conf.max()) if found else 0.0
        if r["visible"] != "1":                              # attacker out of view: nothing to find
            out_of_view += 1
            false_alarms += found
            continue
        dist, bx, by = float(r["dist_m"]), float(r["bx"]), float(r["by"])
        width_px = float(r["asz"]) * W                       # attacker width in the picture
        rows.append((dist, found))
        print(f"{r['name']}: {'FOUND ' if found else 'missed'} conf {conf:.2f}   "
              f"off-centre x {bx:+.2f} y {by:+.2f}   ~{width_px:.0f} px wide")

print(f"\nout-of-view frames: {out_of_view}, false alarms in them: {false_alarms}")
print("\ndistance   visible   found   rate")
for lo, hi in BANDS:
    band = [found for d, found in rows if lo <= d < hi]
    if band:
        print(f"{lo:>2}–{hi:<2} m   {len(band):>7}   {sum(band):>5}   {sum(band) / len(band):5.0%}")