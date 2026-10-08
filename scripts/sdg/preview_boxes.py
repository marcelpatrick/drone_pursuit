# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\preview_boxes.py
# ── Draws each frame's saved box onto a COPY of its photo, for checking by eye.
# ──   The originals in data\raw are never changed.

import glob
import os

import numpy as np
from PIL import Image, ImageDraw

RAW = r"C:\projects\drone_pursuit\drone_pursuit\data\raw"
OUT = r"C:\projects\drone_pursuit\drone_pursuit\data\preview"
os.makedirs(OUT, exist_ok=True)

for png in sorted(glob.glob(os.path.join(RAW, "rgb_*.png"))):
    idx = os.path.basename(png)[4:8]                                  # "0000" from rgb_0000.png
    boxes = np.load(os.path.join(RAW, f"bounding_box_2d_tight_{idx}.npy"))

    img = Image.open(png).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle([b["x_min"], b["y_min"], b["x_max"], b["y_max"]], outline=(255, 0, 0), width=2)

    img.save(os.path.join(OUT, f"preview_{idx}.png"))
    print(f"{idx}: {len(boxes)} box(es)  {[(int(b['x_min']), int(b['y_min']), int(b['x_max']), int(b['y_max'])) for b in boxes]}")

print(f"done — open {OUT}")