# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_pursuit_demo.py
# ── NEW file — the whole content, nothing to insert into.
# ── Imports  : vision_bridge.py, sitting in this SAME folder (scripts\demo\)
# ── Loads    : C:\projects\drone_pursuit\drone_pursuit\models\drone_detector.onnx
# ── Loads    : your Ch.3 checkpoint, passed on the command line
#
# By default it runs TWO rounds of chases and stops by itself:
#   round 1 "calculated": splice OFF — the policy gets the simulator's calculated readings (Chapter 3 baseline)
#   round 2 "camera"    : splice ON  — the policy gets the readings measured from the camera picture
# Then it prints a summary table and PASS/FAIL checks.
#
#   --episodes                    chases per round 
#   --mode both|calculated|camera    which rounds to run (default both)
#   --print_steps                    also print one line per control step

import argparse, torch
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)   # your Ch.3 best_agent.pt
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--mode", choices=["both", "calculated", "camera"], default="both")
parser.add_argument("--print_steps", action="store_true")   # one line per control step (--verbose is taken by AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
import drone_pursuit.tasks  # noqa: F401  (registers your task)
from vision_bridge import DroneDetector, CameraReadingBridge

# ── Pass marks (edit if you want stricter or looser checks) ───────────────────
BASELINE_MIN = 0.80     # calculated round must capture ≥ 80% (same bar as Checkpoint 3.3)
CAMERA_MIN = 0.60       # camera round must capture ≥ 60%
MAX_GAP = 0.25          # camera may be at most 25 percentage points below calculated
SKIP_FRAMES = 2         # pictures ignored after each reset: they can still show the PREVIOUS chase
NEAR_FACTOR = 1.25      # camera "nearly confirms" a capture if its lowest time-to-contact got within 1.25x the threshold

TASK = "Template-Drone-Pursuit-Direct-v0"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
env_cfg.scene.num_envs = 1
env = gym.make(TASK, cfg=env_cfg, render_mode=None)
raw_env = env.unwrapped                      # to reach the camera and the ground truth
env = SkrlVecEnvWrapper(env)

# rebuild the skrl agent exactly as train.py does, then load your checkpoint
agent_cfg = load_cfg_from_registry(TASK, "skrl_cfg_entry_point")
runner = Runner(env, agent_cfg)
runner.agent.load(args.checkpoint)
runner.agent.set_running_mode("eval")

detector = DroneDetector(r"C:\projects\drone_pursuit\drone_pursuit\models\drone_detector.onnx")
TTC = raw_env.cfg.capture_ttc_s           # camera capture threshold in seconds, measured in Chapter 3.3 Step 4
DT = raw_env.step_dt                      # seconds between control steps
RADIUS = raw_env.cfg.capture_radius       # simulator capture distance in metres, set in Chapter 2.1


def run_round(splice):
    """Run args.episodes chases. splice=True → camera readings replace slots 6..12."""
    name = "camera" if splice else "calculated"
    print(f"\n── Round '{name}': splice {'ON' if splice else 'OFF'}, {args.episodes} chases "
          f"(camera time-to-contact threshold {TTC:.2f} s, capture radius {RADIUS:.2f} m) ──")
    res = {"captures": 0, "crashes": 0, "timeouts": 0, "confirmed": 0, "near": 0,
           "steps": 0, "seen": 0, "jitter": []}
    obs, _ = env.reset()
    raw_env.episode_length_buf.zero_()                      # full time for every chase (see note below)
    bridge = CameraReadingBridge()
    ep = {"steps": 0, "seen": 0, "min_ttc": float("inf"), "last_dist": 0.0,
          "cam_says": False, "jitter": []}
    prev_action = None
    done_eps = 0

    while app.is_running() and done_eps < args.episodes:
        # 1) SEE — this step's rendered frame, (1, H, W, 3) uint8 → numpy HWC
        # The first pictures after a reset are rendered before the scene moved, so they still
        # show the end of the previous chase. Treat them as "nothing seen".
        if ep["steps"] < SKIP_FRAMES:
            readings = bridge.update(None)
        else:
            rgb = raw_env._camera.data.output["rgb"][0].cpu().numpy()
            readings = bridge.update(detector.detect(rgb))     # 7 numbers

        # 2) SPLICE — the camera replaces exactly the slots it owns
        if splice:
            obs[:, 6:13] = torch.tensor(readings, device=obs.device)

        # record this moment BEFORE stepping: if the chase ends, the scene resets at once
        ep["steps"] += 1
        ep["last_dist"] = raw_env._dist[0].item()
        if readings[6] == 1.0:                              # "camera saw it" = detector found a box
            ep["seen"] += 1
        ttc = bridge.time_to_contact(DT)
        ep["min_ttc"] = min(ep["min_ttc"], ttc)
        if bridge.captured_ttc(TTC, DT):
            ep["cam_says"] = True
        if args.print_steps:
            print(f"   true dist {ep['last_dist']:5.2f} m | seen {readings[6]:.0f} | size {readings[2]:.3f} "
                  f"| time-to-contact {ttc:5.2f} s | camera says captured: {bridge.captured_ttc(TTC, DT)}")

        # 3) ACT
        with torch.no_grad():
            actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
        a = actions[0].detach().cpu().numpy()
        if prev_action is not None:                         # how much the command changed since last step
            ep["jitter"].append(float(np.abs(a - prev_action).mean()))
        prev_action = a
        obs, reward, terminated, truncated, info = env.step(actions)

        # 4) END OF A CHASE
        if bool(terminated.any()) or bool(truncated.any()):
            done_eps += 1
            if hasattr(raw_env, "_captured"):
                captured = bool(raw_env._captured[0])        # set by _get_dones in Chapter 2
            else:
                captured = ep["last_dist"] < RADIUS * 1.2
            if captured:
                result = "CAPTURE "
                res["captures"] += 1
                res["confirmed"] += ep["cam_says"]
                res["near"] += ep["min_ttc"] <= NEAR_FACTOR * TTC
            elif bool(terminated.any()):
                result = "CRASH   "
                res["crashes"] += 1
            else:
                result = "TIME OUT"
                res["timeouts"] += 1
            jit = float(np.mean(ep["jitter"])) if ep["jitter"] else 0.0
            print(f"chase {done_eps:2d}: {result} | last true dist {ep['last_dist']:.2f} m "
                  f"| camera saw it {ep['seen']:3d}/{ep['steps']:3d} steps "
                  f"| min time-to-contact {ep['min_ttc']:.2f} s / threshold {TTC:.2f} s "
                  f"| camera says captured: {ep['cam_says']} | jitter {jit:.3f}")
            res["steps"] += ep["steps"]
            res["seen"] += ep["seen"]
            res["jitter"].append(jit)
            bridge = CameraReadingBridge()                   # fresh memory for the next chase
            # With one environment, every reset is a reset of ALL environments, and the quadcopter
            # template then gives each chase a RANDOM amount of time left (to spread resets out
            # in training). Give every chase its full time instead.
            raw_env.episode_length_buf.zero_()
            ep = {"steps": 0, "seen": 0, "min_ttc": float("inf"), "last_dist": 0.0,
                  "cam_says": False, "jitter": []}
            prev_action = None

    n = max(done_eps, 1)
    res["episodes"] = done_eps
    res["rate"] = res["captures"] / n
    res["seen_share"] = res["seen"] / max(res["steps"], 1)
    res["jitter_mean"] = float(np.mean(res["jitter"])) if res["jitter"] else 0.0
    return res


rounds = {}
if args.mode in ("both", "calculated"):
    rounds["calculated"] = run_round(splice=False)
if args.mode in ("both", "camera"):
    rounds["camera"] = run_round(splice=True)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
cols = list(rounds)
def row(label, fn):
    print(f"{label:<34}" + "".join(f"{fn(rounds[c]):>14}" for c in cols))

print("\n================ 6.2 SUMMARY ================")
print(f"{'':<34}" + "".join(f"{c:>14}" for c in cols))
row("chases run", lambda r: f"{r['episodes']}")
row("capture rate", lambda r: f"{100 * r['rate']:.0f}%")
row("crashes / time-outs", lambda r: f"{r['crashes']} / {r['timeouts']}")
row("camera saw attacker (of steps)", lambda r: f"{100 * r['seen_share']:.0f}%")
row("action jitter (avg change/step)", lambda r: f"{r['jitter_mean']:.3f}")
row("captures camera confirmed", lambda r: f"{r['confirmed']}/{r['captures']}")
row(f"captures camera nearly confirmed", lambda r: f"{r['near']}/{r['captures']}")

print("\nCHECKS")
checks = []
if "calculated" in rounds:
    r = rounds["calculated"]["rate"]
    checks.append((r >= BASELINE_MIN, f"calculated capture rate ≥ {100*BASELINE_MIN:.0f}%   ({100*r:.0f}%)"
                   "   ← if this fails, the problem is the checkpoint, not the camera"))
if "camera" in rounds:
    r = rounds["camera"]["rate"]
    checks.append((r >= CAMERA_MIN, f"camera capture rate ≥ {100*CAMERA_MIN:.0f}%       ({100*r:.0f}%)"))
if len(rounds) == 2:
    gap = rounds["calculated"]["rate"] - rounds["camera"]["rate"]
    checks.append((gap <= MAX_GAP, f"gap ≤ {100*MAX_GAP:.0f} points               ({100*gap:.0f} points)"))
for ok, text in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")

if "camera" in rounds:
    c = rounds["camera"]
    print(f"  [INFO] camera nearly confirmed {c['near']} of {c['captures']} captures "
          f"(lowest time-to-contact within {NEAR_FACTOR:.2f}x of {TTC:.2f} s).")
    print("         Most of them → the simulator just ends the chase one step before the camera's threshold. Harmless.")
    print("         Few of them  → the detector loses the attacker up close. Look at 'camera saw it' near the end.")
if len(rounds) == 2:
    j0, j1 = rounds["calculated"]["jitter_mean"], rounds["camera"]["jitter_mean"]
    print(f"  [INFO] action jitter camera/calculated = {j1 / max(j0, 1e-9):.2f}x "
          "(above 1 = the defender corrects more often with camera readings)")

all_ok = all(ok for ok, _ in checks)
print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
if len(rounds) == 2:
    print("\nCopy this line into project_notes.txt:")
    print(f"6.2 capture rate ({args.episodes} chases): calculated {100*rounds['calculated']['rate']:.0f}% | "
          f"camera {100*rounds['camera']['rate']:.0f}% | gap "
          f"{100*(rounds['calculated']['rate'] - rounds['camera']['rate']):.0f} points")

env.close()
app.close()