# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\make_grey_floors.py
# ── NEW file — the whole content.
# ── Writes: C:\projects\drone_pursuit\drone_pursuit\data\bg\floor_grey\*.png
# ── Makes 14 plain, low-contrast grey ground textures (plain grey, concrete, asphalt) for the
# ──   ground batch. They are generic on purpose: NOT the arena's grid floor, so the arena in
# ──   5.2 stays a background the detector has never seen (an honest test).
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path(r"C:\projects\drone_pursuit\drone_pursuit\data\bg\floor_grey")
SIZE = 1024
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def blotches(cells, amplitude):
    """Soft, large patches of lighter and darker grey, like stains on concrete."""
    small = Image.fromarray(rng.uniform(-1, 1, (cells, cells)).astype(np.float32), mode="F")
    return np.asarray(small.resize((SIZE, SIZE), Image.BICUBIC)) * amplitude


def save(name, grey):
    img = np.clip(grey, 0, 255).astype(np.uint8)
    Image.fromarray(np.stack([img] * 3, axis=-1)).save(OUT / f"{name}.png")
    print(f"saved {name}.png  (average brightness {img.mean():.0f} of 255)")


# 1. plain grey: almost flat, from dark to light, with a little grain
for level in (70, 100, 130, 160, 190):
    save(f"plain_{level}", level + rng.normal(0, 3, (SIZE, SIZE)))

# 2. concrete: mid grey, soft stains plus fine grain
for i in range(5):
    base = rng.uniform(95, 175)
    save(f"concrete_{i}", base + blotches(12, 15) + blotches(48, 6) + rng.normal(0, 6, (SIZE, SIZE)))

# 3. asphalt: dark grey with light speckles (the stones in the tar)
for i in range(4):
    base = rng.uniform(50, 95)
    speckle = (rng.random((SIZE, SIZE)) < 0.04) * rng.uniform(20, 60, (SIZE, SIZE))
    save(f"asphalt_{i}", base + blotches(16, 8) + rng.normal(0, 10, (SIZE, SIZE)) + speckle)

print(f"done — 14 textures in {OUT}")