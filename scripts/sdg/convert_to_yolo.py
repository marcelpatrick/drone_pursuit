# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\convert_to_yolo.py
# ── The whole content (replaces the 4.3 version).
# ── Reads : C:\projects\drone_pursuit\drone_pursuit\data\raw
# ── Writes: C:\projects\drone_pursuit\drone_pursuit\data\yolo
# ── Changes since 4.3:
# ──   1. thin, side-on drones are kept (10 px long and at least 4 px thick)
# ──   2. frames inside BG_RANGES (the bird batch) that contain NO drone at all become
# ──      background images: an empty label file that teaches "nothing to find here"

import json, random, shutil
from pathlib import Path
import numpy as np

RAW = Path(r"C:\projects\drone_pursuit\drone_pursuit\data\raw")
OUT = Path(r"C:\projects\drone_pursuit\drone_pursuit\data\yolo")
IMG_W, IMG_H = 640, 480
MIN_BOX_PX = 10          # a drone's long side must be at least this many pixels
MIN_THIN_PX = 4          # ...and its short side at least this many (keeps side-on drones)
BG_RANGES = [(5000, 6499)]   # frame numbers of the bird batch (from merge_raw_batches.py);
                             # only these may become background images


def class_name(entry):
    """A labels-file entry is usually {"class": "drone"}; return the class name either way."""
    return entry.get("class") if isinstance(entry, dict) else entry


def in_bg_range(number):
    return any(lo <= number <= hi for lo, hi in BG_RANGES)


# start clean: images and labels from an earlier conversion would mix with this one
# (drone.yaml, written in Step 2, lives in OUT and is kept)
for sub in ("images", "labels"):
    shutil.rmtree(OUT / sub, ignore_errors=True)

kept, background, dropped = 0, 0, 0
samples = []
for rgb in sorted(RAW.glob("rgb_*.png")):
    idx = rgb.stem.split("_")[-1]
    npy = RAW / f"bounding_box_2d_tight_{idx}.npy"
    lbl = RAW / f"bounding_box_2d_tight_labels_{idx}.json"
    if not npy.exists() or not lbl.exists() or rgb.stat().st_size == 0:   # incomplete or corrupt frame
        dropped += 1; continue
    boxes = np.load(npy)
    id2label = {int(k): class_name(v) for k, v in json.loads(lbl.read_text()).items()}
    has_drone = False
    lines = []
    for b in boxes:
        if id2label.get(int(b["semanticId"])) != "drone":    # birds and distractor boxes: ignore
            continue
        has_drone = True
        w, h = b["x_max"] - b["x_min"], b["y_max"] - b["y_min"]
        if max(w, h) < MIN_BOX_PX or min(w, h) < MIN_THIN_PX:   # a speck: too small to learn
            continue
        xc, yc = (b["x_min"] + w / 2) / IMG_W, (b["y_min"] + h / 2) / IMG_H
        lines.append(f"0 {xc:.6f} {yc:.6f} {w/IMG_W:.6f} {h/IMG_H:.6f}")
    if lines:                                                # a drone big enough to learn from
        samples.append((rgb, lines)); kept += 1
    elif in_bg_range(int(idx)) and not has_drone:           # bird photo with no drone at all
        samples.append((rgb, [])); background += 1           # → empty label file = background image
    else:                                                    # no drone box, or only a speck
        dropped += 1

random.seed(42); random.shuffle(samples)
n_val = max(1, int(0.15 * len(samples)))
for split, chunk in [("val", samples[:n_val]), ("train", samples[n_val:])]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    for rgb, lines in chunk:
        shutil.copy(rgb, OUT / "images" / split / rgb.name)
        (OUT / "labels" / split / f"{rgb.stem}.txt").write_text("\n".join(lines))
total = kept + background + dropped
print(f"kept {kept} with a drone, {background} background, dropped {dropped} "
      f"({dropped / total:.0%} of {total}), val {n_val}")