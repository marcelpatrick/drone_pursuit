# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\merge_raw_batches.py
# ── NEW file — the whole content.
# ── Moves one or more raw batch folders into data\raw, renumbering every frame so it
# ──   continues after the highest number already in data\raw (or from 0000 if it is empty).
# ── All 4 files of a frame (rgb, .npy, labels .json, prim_paths .json) get the same new number.
# ── Usage:  python merge_raw_batches.py raw_s1 raw_s2 raw_birds     (first merge)
# ──         python merge_raw_batches.py raw_ground                  (adds a later batch)
import csv, re, shutil, sys
from pathlib import Path

DATA = Path(r"C:\projects\drone_pursuit\drone_pursuit\data")
DST = DATA / "raw"
NAME = re.compile(r"^(.*_)(\d{4})(\.\w+)$")      # "rgb_" + "0042" + ".png"
SOURCES = sys.argv[1:]
if not SOURCES:
    raise SystemExit("STOP: name the folders to merge, e.g.  python merge_raw_batches.py raw_ground")

# 1. start numbering after whatever data\raw already holds
DST.mkdir(exist_ok=True)
existing = [int(m.group(2)) for f in DST.iterdir() if (m := NAME.match(f.name))]
offset = max(existing) + 1 if existing else 0
print(f"data\\raw already holds frames up to {offset - 1:04d}" if existing else "data\\raw is empty")

# 2. plan every move first, touch nothing yet
plan = []                                         # (old file, new file, source folder, old number)
for src in SOURCES:
    folder = DATA / src
    if not folder.is_dir():
        raise SystemExit(f"STOP: {folder} not found")
    matches = [(f, m) for f in sorted(folder.iterdir()) if (m := NAME.match(f.name))]
    if not matches:
        raise SystemExit(f"STOP: no frame files found in {folder}")
    top = max(int(m.group(2)) for _, m in matches)
    for f, m in matches:
        new = DST / f"{m.group(1)}{int(m.group(2)) + offset:04d}{m.group(3)}"
        plan.append((f, new, src, m.group(2)))
    photos = sum(1 for f, _ in matches if f.suffix == ".png")
    print(f"{src:12s} {photos:5d} photos -> numbers {offset:04d} to {top + offset:04d}")
    offset += top + 1                             # next batch starts right after this one

# 3. safety checks
if offset > 10000:
    raise SystemExit("STOP: more than 10,000 frames would need 5-digit numbers")
new_names = [new.name for _, new, _, _ in plan]
if len(set(new_names)) != len(new_names) or any(new.exists() for _, new, _, _ in plan):
    raise SystemExit("STOP: a new name would collide with an existing file")

# 4. move, and add to the record of where every file came from
log_path = DATA / "raw_merge_log.csv"
new_log = not log_path.exists()
with open(log_path, "a", newline="") as fh:
    log = csv.writer(fh)
    if new_log:
        log.writerow(["new_name", "from_folder", "old_number"])
    for f, new, src, old in plan:
        shutil.move(str(f), str(new))
        log.writerow([new.name, src, old])
print(f"moved {len(plan)} files into {DST}")