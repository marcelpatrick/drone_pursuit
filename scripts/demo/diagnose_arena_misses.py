# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\diagnose_arena_misses.py
# ── NEW file — the whole content, nothing to insert into. Does NOT replace score_arena_frames.py.
# ── Environment: drone_vision
# ── Reads: data\arena_frames\*.png and readings.csv (5.2 Step 2), drone_v3 best.pt (5.1)
# ── Answers three questions score_arena_frames.py cannot:
# ──   1. For each miss: did the detector see nothing, or see the drone but with a confidence under 0.4?
# ──   2. Is each box ON the attacker (where readings.csv says it should be), or somewhere else?
# ──   3. What would the found rate and the false alarms be at a lower threshold (0.3, 0.25, 0.2)?
import csv
from pathlib import Path
from ultralytics import YOLO

FRAMES = Path(r"C:\projects\drone_pursuit\drone_pursuit\data\arena_frames")
MODEL = r"C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v3\weights\best.pt"
W, H = 640, 480                       # camera image size in pixels
LOW = 0.05                            # keep almost every box, so we can read its real confidence
THRESHOLDS = [0.4, 0.3, 0.25, 0.2]    # 0.4 = the threshold vision_bridge.py uses today
BANDS = [(0, 2), (2, 5), (5, 99)]     # distance ranges in metres

model = YOLO(MODEL)
visible, empty = [], []               # visible: one row per v1 frame; empty: best confidence per v0 frame

with open(FRAMES / "readings.csv", newline="") as f:
    for r in csv.DictReader(f):
        res = model.predict(str(FRAMES / r["name"]), conf=LOW, verbose=False)[0]
        boxes = res.boxes.xywh.cpu().numpy()          # box centre x, centre y, width, height (pixels)
        confs = res.boxes.conf.cpu().numpy()
        if r["visible"] != "1":                       # attacker out of view: any box is a false alarm
            empty.append(float(confs.max()) if len(confs) else 0.0)
            continue
        bx, by = float(r["bx"]), float(r["by"])
        xc, yc = W / 2 + bx * W / 2, H / 2 + by * H / 2   # where the attacker should be (same maths as the red box)
        width_px = float(r["asz"]) * W
        tol = max(30.0, 1.5 * width_px)               # how far a box centre may be from that spot and still count
        on, off = 0.0, 0.0                            # best confidence ON the attacker / ELSEWHERE in the picture
        for (x, y, _, _), c in zip(boxes, confs):
            if abs(x - xc) <= tol and abs(y - yc) <= tol:
                on = max(on, float(c))
            else:
                off = max(off, float(c))
        visible.append(dict(name=r["name"], dist=float(r["dist_m"]), px=width_px,
                            bx=bx, by=by, on=on, off=off))

# ── 1. every frame missed at 0.4, with what the detector actually saw ──────────
print("MISSED AT 0.4 (best confidence ON the attacker / ELSEWHERE in the picture)")
for v in visible:
    if v["on"] < 0.4:
        note = "   <- box in the WRONG place" if v["off"] >= 0.4 else ""
        print(f"  {v['name']}: on {v['on']:.2f}  elsewhere {v['off']:.2f}   "
              f"~{v['px']:.0f} px wide   x {v['bx']:+.2f} y {v['by']:+.2f}{note}")

print("\nFOUND AT 0.4 BUT THE BOX IS NOT ON THE ATTACKER")
wrong_only = [v for v in visible if v["on"] < 0.4 <= v["off"]]
for v in wrong_only:
    print(f"  {v['name']}")
if not wrong_only:
    print("  none")

# ── 2. found rate and false alarms at each threshold ──────────────────────────
print("\nthreshold   " + "   ".join(f"{lo}–{hi} m found" for lo, hi in BANDS[:2])
      + "   wrong-place boxes   false alarms (v0)")
for t in THRESHOLDS:
    cells = []
    for lo, hi in BANDS[:2]:
        band = [v for v in visible if lo <= v["dist"] < hi]
        hit = sum(v["on"] >= t for v in band)
        cells.append(f"{hit:>3}/{len(band):<3} {hit / max(len(band), 1):4.0%}")
    wrong = sum(v["off"] >= t for v in visible)
    alarms = sum(c >= t for c in empty)
    print(f"   {t:.2f}     " + "      ".join(cells) + f"       {wrong:>5}              {alarms:>3} of {len(empty)}")

# ── 3. found rate by size in the picture, at 0.4 ──────────────────────────────
print("\nwidth in picture   visible   found at 0.4")
for lo, hi in [(0, 16), (16, 20), (20, 25), (25, 40), (40, 999)]:
    band = [v for v in visible if lo <= v["px"] < hi]
    if band:
        hit = sum(v["on"] >= 0.4 for v in band)
        print(f"  {lo:>3}–{hi:<3} px      {len(band):>5}     {hit:>3}  ({hit / len(band):.0%})")