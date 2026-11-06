# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\run_robustness.py
# ── NEW file — the whole content, nothing to insert into.
# ── Runs   : vision_pursuit_demo.py (6.2), in this SAME folder, once per condition
# ── Writes : C:\projects\drone_pursuit\drone_pursuit\data\robustness\<date-time>\
#            one log + one results file per condition, and summary.txt
# ── NOTE   : no Isaac Lab here. Each condition is a separate run of the demo script,
#            because the attacker's size is fixed when the scene is built.
# ── NOTE   : every run opens the Isaac Sim window, on purpose. On Windows, a headless run
#            sometimes starts with a camera that never shows the attacker, and every chase of
#            that run then fails. With the window open this does not happen.
#
# Conditions, in order:
#   1. baseline      normal size, normal light, BOTH rounds (the 6.2 test)
#   2. small         attacker x0.6 (about 8 cm),  camera round only
#   3. large         attacker x2.5 (about 33 cm), camera round only
#   4. light         normal size, random lighting, camera round only
#
#   --checkpoint ...   your Ch.3 best_agent.pt (required)
#   --episodes 50      chases per round (default 50)
#   --only small,large run only some conditions (baseline is always included)

import argparse, json, os, subprocess, sys
from datetime import datetime
from pathlib import Path

# Windows sends a script's output through a pipe in an old code page that has no "─", "≥" or "→",
# and the demo prints those characters. Force UTF-8 for this script and for every demo it starts.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CHILD_ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

HERE = Path(__file__).resolve().parent
DEMO = HERE / "vision_pursuit_demo.py"
ROOT = Path(r"C:\projects\drone_pursuit\drone_pursuit")

# ── Pass marks from 6.2 Step 2 and 6.3 Step 3 (suggested values for this project) ─
BASELINE_CALC_MIN = 0.80    # 6.2: calculated round captures ≥ 80%
BASELINE_CAM_MIN = 0.60     # 6.2: camera round captures ≥ 60%
BASELINE_MAX_GAP = 0.25     # 6.2: camera at most 25 points below calculated
COND_CAM_MIN = 0.80         # 6.3: every condition captures ≥ 80%
COND_CRASH_MAX = 0.06       # 6.3: crashes ≤ 3 of 50 chases (6%)
CONFIRM_MAX_DIFF = 0.15     # 6.3: size conditions' "confirmed" share within 15 points of the baseline

CONDITIONS = [
    ("baseline", "normal size, normal light", ["--mode", "both"]),
    ("small",    "attacker x0.6 (about 8 cm)", ["--mode", "camera", "--attacker_scale", "0.6"]),
    ("large",    "attacker x2.5 (about 33 cm)", ["--mode", "camera", "--attacker_scale", "2.5"]),
    ("light",    "normal size, random lighting", ["--mode", "camera", "--light_random"]),
]

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--only", type=str, default="", help="e.g. small,light (baseline always runs)")
args = parser.parse_args()

wanted = {"baseline"} | {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None
conditions = [c for c in CONDITIONS if wanted is None or c[0] in wanted]

out_dir = ROOT / "data" / "robustness" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Results folder: {out_dir}\n")



def run(key, label, extra):
    """Run the demo once, show its output live, save it to a log, return its results (or None)."""
    results_file = out_dir / f"{key}.json"
    cmd = [sys.executable, str(DEMO), "--checkpoint", args.checkpoint, "--enable_cameras",
           "--episodes", str(args.episodes), "--results_json", str(results_file)] + extra
    print("=" * 90)
    print(f"CONDITION '{key}': {label}")
    print("  " + " ".join(cmd))
    print("=" * 90, flush=True)
    with open(out_dir / f"{key}.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=CHILD_ENV,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            log.write(line)
            # show only the lines that matter, so the console stays readable
            if line.startswith(("chase", "──", "[6.3]", "RESULT", "Traceback", "  [")) \
                    or ("Error" in line and "Failed to clone in Fabric" not in line):
                print(line, end="", flush=True)
        proc.wait()
    if proc.returncode != 0 or not results_file.exists():
        print(f"\n!! Condition '{key}' did not finish (exit code {proc.returncode}). "
              f"See {out_dir / (key + '.log')}\n")
        return None
    return json.loads(results_file.read_text(encoding="utf-8"))


results = {key: run(key, label, extra) for key, label, extra in conditions}

# ── SUMMARY ───────────────────────────────────────────────────────────────────
lines = []
def say(s=""):
    print(s)
    lines.append(s)

def pct(x):
    return f"{100 * x:.0f}%"

say("\n" + "=" * 90)
say(f"6.3 ROBUSTNESS SUMMARY   ({args.episodes} chases per round, checkpoint {Path(args.checkpoint).parent.parent.name})")
say("=" * 90)
say(f"{'condition':<12}{'capture':>9}{'crashes':>9}{'time-outs':>11}{'camera saw':>12}"
    f"{'confirmed':>11}{'nearly':>9}{'jitter':>8}")
cam = {}
for key, label, _ in conditions:
    r = results[key]
    if r is None:
        say(f"{key:<12}{'did not finish':>40}")
        continue
    c = r["rounds"]["camera"]
    cam[key] = c
    n = max(c["captures"], 1)
    say(f"{key:<12}{pct(c['rate']):>9}{c['crashes']:>9}{c['timeouts']:>11}{pct(c['seen_share']):>12}"
        f"{pct(c['confirmed'] / n):>11}{pct(c['near'] / n):>9}{c['jitter_mean']:>8.3f}")
if results.get("baseline") and "calculated" in results["baseline"]["rounds"]:
    calc = results["baseline"]["rounds"]["calculated"]
    say(f"{'(calculated)':<12}{pct(calc['rate']):>9}{calc['crashes']:>9}{calc['timeouts']:>11}"
        f"{'':>12}{'':>11}{'':>9}{calc['jitter_mean']:>8.3f}")
say("confirmed / nearly = share of captures the camera alone confirmed (fully / within 1.25x the threshold)")

say("\nCHECKS")
checks = []
def check(ok, text):
    checks.append(ok)
    say(f"  [{'PASS' if ok else 'FAIL'}] {text}")

b = results.get("baseline")
if b is None:
    check(False, "baseline (6.2) finished")
else:
    calc, bc = b["rounds"]["calculated"], b["rounds"]["camera"]
    check(calc["rate"] >= BASELINE_CALC_MIN, f"baseline: calculated capture ≥ {pct(BASELINE_CALC_MIN)}   ({pct(calc['rate'])})")
    check(bc["rate"] >= BASELINE_CAM_MIN, f"baseline: camera capture ≥ {pct(BASELINE_CAM_MIN)}       ({pct(bc['rate'])})")
    gap = calc["rate"] - bc["rate"]
    check(gap <= BASELINE_MAX_GAP, f"baseline: gap ≤ {100 * BASELINE_MAX_GAP:.0f} points               ({100 * gap:.0f} points)")
base_conf = cam["baseline"]["confirmed"] / max(cam["baseline"]["captures"], 1) if "baseline" in cam else None

for key, label, _ in conditions[1:]:
    if results[key] is None:
        check(False, f"{key}: finished")
        continue
    c = cam[key]
    # Sanity check first: if the camera NEVER sees the attacker, the attacker is probably not
    # in the picture at all (a scene problem), and the other numbers say nothing about the detector.
    check(c["seen_share"] >= 0.05, f"{key}: attacker visible to the camera at all  ({pct(c['seen_share'])} of steps)"
          + ("" if c["seen_share"] >= 0.05 else "   ← look with --show_view: is the attacker drawn?"))
    check(c["rate"] >= COND_CAM_MIN, f"{key}: camera capture ≥ {pct(COND_CAM_MIN)}           ({pct(c['rate'])})")
    check(c["crashes"] / max(c["episodes"], 1) <= COND_CRASH_MAX,
          f"{key}: crashes ≤ {COND_CRASH_MAX * args.episodes:.0f} of {args.episodes}             ({c['crashes']})")
    if key in ("small", "large") and base_conf is not None:
        conf = c["confirmed"] / max(c["captures"], 1)
        check(abs(conf - base_conf) <= CONFIRM_MAX_DIFF,
              f"{key}: confirmed within {100 * CONFIRM_MAX_DIFF:.0f} points of baseline  "
              f"({pct(conf)} vs {pct(base_conf)})")

say("\nIF A CHECK FAILS (details in 6.3 Step 3):")
say("  baseline calculated   → the checkpoint is the problem, not the camera")
say("  small                 → detector misses small, far drones: more small-drone images (Ch.4), retrain detector (Ch.5)")
say("  large                 → box cut off up close: rely on the larger stop threshold in Chapter 7")
say("  light                 → add brightness/colour randomization to the synthetic images (Ch.4), retrain detector")
say("  confirmed differs     → time-to-contact shifts with size in rendered boxes: re-check it in 7.4")

say(f"\nRESULT: {'PASS' if checks and all(checks) else 'FAIL'}")

say("\nCopy these lines into project_notes.txt:")
for key, label, _ in conditions:
    if key in cam:
        c = cam[key]
        say(f"6.3 [{key}: {label}] ({args.episodes} chases): camera capture {pct(c['rate'])} | crashes {c['crashes']} | "
            f"camera saw {pct(c['seen_share'])} | confirmed {c['confirmed']}/{c['captures']} | "
            f"nearly {c['near']}/{c['captures']} | jitter {c['jitter_mean']:.3f}")

(out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nSummary saved: {out_dir / 'summary.txt'}")