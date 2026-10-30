# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\check_bridge.py
# ── NEW file — the whole content, nothing to insert into.
# ── Reads  : data\arena_frames\frame_*.png and data\arena_frames\readings.csv (5.2 Step 2)
# ── Imports: vision_bridge.py, in this SAME folder (scripts\demo\)
# ── NOTE   : no Isaac Lab here — it only reads saved files

import csv
from pathlib import Path
import numpy as np
from PIL import Image
from vision_bridge import DroneDetector, CameraReadingBridge

ROOT = Path(r"C:\projects\drone_pursuit\drone_pursuit")
FRAMES = ROOT / "data" / "arena_frames"
BEAR_TOL = 0.05        # bearings must agree within 0.05 ("a few hundredths")
SIZE_TOL = 0.20        # size must agree within 20 %

detector = DroneDetector(str(ROOT / "models" / "drone_detector.onnx"))
with open(FRAMES / "readings.csv", newline="") as f:
    rows = list(csv.DictReader(f))


def load(name):
    rgb = np.array(Image.open(FRAMES / name).convert("RGB"))
    assert rgb.shape == (480, 640, 3), f"{name} is {rgb.shape}, the bridge expects (480, 640, 3)"
    return rgb


# ── PART 1 (Step 2): attacker in view — the same moment computed two ways ─────
print("STEP 2 — projected (training) vs bridge (from image)\n")
results = []                                         # (projected, bridge, ok)
for r in rows:
    if r["visible"] != "1":
        continue                                     # out-of-view frames are for Part 2
    box = detector.detect(load(r["name"]))
    if box is None:
        print(f"{r['name']}: detector found nothing, skipped (a 5.2 issue, not a bridge issue)")
        continue
    b = CameraReadingBridge().update(box)[:3]        # fresh bridge: only bx, by, size matter here
    p = np.array([float(r["bx"]), float(r["by"]), float(r["asz"])])
    ok = abs(b[0] - p[0]) <= BEAR_TOL and abs(b[1] - p[1]) <= BEAR_TOL and abs(b[2] / p[2] - 1) <= SIZE_TOL
    results.append((p, b, ok))
    print(f"{r['name']}\n   projected: bx={p[0]:+.3f} by={p[1]:+.3f} size={p[2]:.3f}"
          f"\n   bridge   : bx={b[0]:+.3f} by={b[1]:+.3f} size={b[2]:.3f}   {'OK' if ok else 'MISMATCH'}")

n_ok = sum(ok for _, _, ok in results)
print(f"\nmatching frames: {n_ok} of {len(results)} compared   (need 15 or more)")


def scale(i):                                        # how much bigger the bridge reading is, typically
    v = [b[i] / p[i] for p, b, _ in results if abs(p[i]) > 0.1]
    return float(np.median(v)) if v else float("nan")


print(f"typical ratio bridge/projected:  horizontal {scale(0):+.2f}   vertical {scale(1):+.2f}   "
      f"size {np.median([b[2] / p[2] for p, b, _ in results]) if results else float('nan'):.2f}")
print("   all three near +1.00 → same camera on both sides")
print("   a negative ratio     → that axis is flipped")
print("   all three off by the same factor → field of view differs (focal length / aperture)")
print("   only size off        → --fixed_span differs from the rendered drone's width")

# ── PART 2 (Step 3): attacker out of view — the bridge must hold its last values ─
print("\nSTEP 3 — out-of-view behaviour\n")
bridge = CameraReadingBridge()
last = bridge.update(np.array([400.0, 200.0, 32.0, 24.0]))   # a last sighting: right of and above centre
out_frames = [r for r in rows if r["visible"] == "0"
              and (abs(float(r["bx"])) >= 1 or abs(float(r["by"])) >= 1)]   # truly outside the picture
if out_frames:
    inputs = [(r["name"], detector.detect(load(r["name"]))) for r in out_frames]
else:
    print("no out-of-view frames were captured — testing a simulated detector miss instead")
    inputs = [("simulated miss", None)]

all_ok = True
for name, box in inputs:
    out = bridge.update(box)
    frozen = np.allclose(out[:3], last[:3])
    zero_d = np.allclose(out[3:6], 0.0)
    ok = out[6] == 0.0 and frozen and zero_d
    all_ok &= ok
    note = "" if box is None else "   ← detector saw a drone that is not there"
    print(f"{name}: visible={out[6]:.0f}  bearings held={frozen}  rates zero={zero_d}  "
          f"{'OK' if ok else 'FAIL'}{note}")
print(f"\nout-of-view check: {'PASS' if all_ok else 'FAIL'}")