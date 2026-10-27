# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\score_arena_frames.py
# ── NEW file — the whole content.
# ── Reads: data\arena_frames\frame_XXX_dDD.DDm.png (Step 2) and best.pt (5.1)

import re
from pathlib import Path
from ultralytics import YOLO

FRAMES = Path(r"C:\projects\drone_pursuit\drone_pursuit\data\arena_frames")
MODEL = r"C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\weights\best.pt"
CONF = 0.4                                   # same confidence threshold the converter uses in Chapter 6
BANDS = [(0, 2), (2, 5), (5, 99)]            # distance ranges in metres

model = YOLO(MODEL)
rows = []
for img in sorted(FRAMES.glob("frame_*.png")):
    dist = float(re.search(r"_d([\d.]+)m", img.stem).group(1))   # distance from the file name
    result = model.predict(str(img), conf=CONF, verbose=False)[0]
    found = len(result.boxes) > 0
    conf = float(result.boxes.conf.max()) if found else 0.0
    rows.append((dist, found))
    print(f"{img.name}: {'FOUND ' if found else 'missed'}  confidence {conf:.2f}")

print("\ndistance     pictures   found   rate")
for lo, hi in BANDS:
    band = [found for d, found in rows if lo <= d < hi]
    if band:
        print(f"{lo:>2}–{hi:<2} m     {len(band):>6}   {sum(band):>5}   {sum(band) / len(band):5.0%}")