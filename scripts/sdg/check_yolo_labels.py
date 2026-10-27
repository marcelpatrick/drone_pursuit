# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\check_yolo_labels.py
# ── Draws the YOLO labels of 10 random training images back onto COPIES of them.
# ──   Writes to data\yolo_check — outside data\yolo, so nothing here is ever trained on.

import random
from pathlib import Path
from PIL import Image, ImageDraw

YOLO = Path(r"C:\projects\drone_pursuit\drone_pursuit\data\yolo")
OUT = YOLO.parent / "yolo_check"
OUT.mkdir(exist_ok=True)

images = sorted((YOLO / "images" / "train").glob("*.png"))
random.seed(0)
for img_path in random.sample(images, min(10, len(images))):
    label = YOLO / "labels" / "train" / f"{img_path.stem}.txt"
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    for line in label.read_text().splitlines():
        cls, xc, yc, w, h = line.split()
        xc, yc, w, h = float(xc) * W, float(yc) * H, float(w) * W, float(h) * H   # back to pixels
        draw.rectangle([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], outline=(0, 255, 0), width=2)
        print(f"{img_path.name}: class {cls}, box {w:.0f} x {h:.0f} px")
    img.save(OUT / f"check_{img_path.name}")

print(f"done — open {OUT}")