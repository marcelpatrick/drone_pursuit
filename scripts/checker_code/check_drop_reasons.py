# ── Counts, per batch, why frames are kept or dropped by convert_to_yolo.py. Changes nothing.
import json
from collections import Counter
from pathlib import Path
import numpy as np

RAW = Path(r"C:\projects\drone_pursuit\drone_pursuit\data\raw")
counts = Counter()

for rgb in sorted(RAW.glob("rgb_*.png")):
    idx = rgb.stem.split("_")[-1]
    i = int(idx)
    batch = "s1" if i < 2500 else "s2" if i < 5000 else "birds"
    npy = RAW / f"bounding_box_2d_tight_{idx}.npy"
    lbl = RAW / f"bounding_box_2d_tight_labels_{idx}.json"
    if not npy.exists() or not lbl.exists():
        counts[batch, "1 missing files"] += 1
        continue
    boxes = np.load(npy)
    labels = json.loads(lbl.read_text())
    drone_ids = {int(k) for k, v in labels.items()
                 if (v.get("class") if isinstance(v, dict) else v) == "drone"}
    drones = [b for b in boxes if int(b["semanticId"]) in drone_ids]
    if not drones:
        counts[batch, "2 no drone box"] += 1
        continue
    b = drones[0]
    w, h = b["x_max"] - b["x_min"], b["y_max"] - b["y_min"]
    long_side, short_side = max(w, h), min(w, h)
    if short_side >= 10:
        counts[batch, "3 kept: both sides >= 10 px"] += 1
    elif long_side >= 10 and short_side >= 4:
        counts[batch, "4 THIN: long >= 10, short 4-9 px"] += 1
    else:
        counts[batch, "5 speck: too small to learn"] += 1

for batch in ("s1", "s2", "birds"):
    print(f"\n{batch}")
    for (bt, reason), n in sorted(counts.items()):
        if bt == batch:
            print(f"  {reason:38s} {n:5d}")