# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\preview_boxes.py
# ── Draws each frame's saved boxes onto a COPY of its photo (drone red,
# ──   distractors yellow), plus an enlarged copy of the area around the drone.
# ──   The originals in data\raw are never changed.

import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

RAW = r"C:\projects\drone_pursuit\drone_pursuit\data\raw"
OUT = r"C:\projects\drone_pursuit\drone_pursuit\data\preview"
os.makedirs(OUT, exist_ok=True)

for png in sorted(glob.glob(os.path.join(RAW, "rgb_*.png"))):
    idx = os.path.basename(png)[4:8]                                  # "0000" from rgb_0000.png
    boxes = np.load(os.path.join(RAW, f"bounding_box_2d_tight_{idx}.npy"))
    with open(os.path.join(RAW, f"bounding_box_2d_tight_labels_{idx}.json")) as f:
        labels = json.load(f)
    drone_ids = {int(k) for k, v in labels.items() if "drone" in str(v)}
    drone_boxes = [b for b in boxes if int(b["semanticId"]) in drone_ids]

    img = Image.open(png).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in boxes:
        colour = (255, 0, 0) if int(b["semanticId"]) in drone_ids else (255, 220, 0)
        draw.rectangle([b["x_min"], b["y_min"], b["x_max"], b["y_max"]], outline=colour, width=2)
    img.save(os.path.join(OUT, f"preview_{idx}.png"))

    # enlarged copy of the drone, so a 10-pixel drone can be checked by eye
    if drone_boxes:
        b = drone_boxes[0]
        Z = 8                                                     # enlargement factor
        cx, cy = (b["x_min"] + b["x_max"]) / 2, (b["y_min"] + b["y_max"]) / 2
        half = max(b["x_max"] - b["x_min"], b["y_max"] - b["y_min"], 16)   # window at least 32x32 px
        x0, y0 = int(cx - half), int(cy - half)
        clean = Image.open(png).convert("RGB")                    # the photo WITHOUT boxes drawn on it
        crop = clean.crop((x0, y0, x0 + 2 * int(half), y0 + 2 * int(half)))
        crop = crop.resize((crop.width * Z, crop.height * Z), Image.NEAREST)
        zd = ImageDraw.Draw(crop)
        zd.rectangle([(b["x_min"] - x0) * Z - 3, (b["y_min"] - y0) * Z - 3,       # 3 px outside the drone,
                      (b["x_max"] - x0 + 1) * Z + 2, (b["y_max"] - y0 + 1) * Z + 2],  # so it never covers it
                     outline=(255, 0, 0), width=1)
        crop.save(os.path.join(OUT, f"zoom_{idx}.png"))

    print(f"{idx}: drone={len(drone_boxes)}  distractors={len(boxes) - len(drone_boxes)}")

print(f"done — open {OUT}")