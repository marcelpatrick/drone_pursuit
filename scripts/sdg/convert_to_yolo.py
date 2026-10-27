# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\convert_to_yolo.py
# ── This is a NEW file — the whole content, nothing to insert into.
# ── Reads : C:\projects\drone_pursuit\drone_pursuit\data\raw
# ── Writes: C:\projects\drone_pursuit\drone_pursuit\data\yolo

import json, random, shutil
from pathlib import Path
import numpy as np

RAW = Path(r"C:\projects\drone_pursuit\drone_pursuit\data\raw")
OUT = Path(r"C:\projects\drone_pursuit\drone_pursuit\data\yolo")
IMG_W, IMG_H = 640, 480
MIN_BOX_PX = 10          # boxes smaller than this are unlearnable specks — drop them


def class_name(entry):
    """A labels-file entry is usually {"class": "drone"}; return the class name either way."""
    return entry.get("class") if isinstance(entry, dict) else entry


# start clean: images and labels from an earlier conversion would mix with this one
# (drone.yaml, written in Step 2, lives in OUT and is kept)
for sub in ("images", "labels"):
    shutil.rmtree(OUT / sub, ignore_errors=True)

kept, dropped = 0, 0
samples = []
for rgb in sorted(RAW.glob("rgb_*.png")):
    idx = rgb.stem.split("_")[-1]
    npy = RAW / f"bounding_box_2d_tight_{idx}.npy"
    lbl = RAW / f"bounding_box_2d_tight_labels_{idx}.json"
    if not npy.exists() or not lbl.exists() or rgb.stat().st_size == 0:   # incomplete or corrupt frame
        dropped += 1; continue
    boxes = np.load(npy)
    id2label = {int(k): class_name(v) for k, v in json.loads(lbl.read_text()).items()}
    lines = []
    for b in boxes:
        if id2label.get(int(b["semanticId"])) != "drone":    # distractor boxes: ignore
            continue
        w, h = b["x_max"] - b["x_min"], b["y_max"] - b["y_min"]
        if w < MIN_BOX_PX or h < MIN_BOX_PX:
            continue
        xc, yc = (b["x_min"] + w / 2) / IMG_W, (b["y_min"] + h / 2) / IMG_H
        lines.append(f"0 {xc:.6f} {yc:.6f} {w/IMG_W:.6f} {h/IMG_H:.6f}")
    if lines:                                                # keep only frames with a drone
        samples.append((rgb, lines)); kept += 1
    else:
        dropped += 1

random.seed(42); random.shuffle(samples)
n_val = max(1, int(0.15 * len(samples)))
for split, chunk in [("val", samples[:n_val]), ("train", samples[n_val:])]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    for rgb, lines in chunk:
        shutil.copy(rgb, OUT / "images" / split / rgb.name)
        (OUT / "labels" / split / f"{rgb.stem}.txt").write_text("\n".join(lines))
print(f"kept {kept}, dropped {dropped}, val {n_val}")