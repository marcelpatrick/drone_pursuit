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
#   --episodes 20                    chases per round (default 20)
#   --mode both|calculated|camera    which rounds to run (default both)
#   --print_steps                    also print one line per control step
#   --attacker_scale 2.5             render the attacker this many times its normal size (6.3)
#   --light_random                   new random light brightness and tint at every chase (6.3)
#   --show_view                      live window: the defender's camera, the detector's box, distances (6.3)
#   --record view.mp4                save that same view as a video file (6.3)
#   --results_json out.json          also save the results as a file, for run_robustness.py (6.3)

import argparse, torch
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)   # your Ch.3 best_agent.pt
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--mode", choices=["both", "calculated", "camera"], default="both")
parser.add_argument("--attacker_scale", type=float, default=1.0)   # 6.3: 0.6 ≈ 8 cm, 2.5 ≈ 33 cm
parser.add_argument("--light_random", action="store_true")          # 6.3: random brightness + tint per chase
parser.add_argument("--show_view", action="store_true")             # 6.3: live window of the defender's camera
parser.add_argument("--record", type=str, default="")               # 6.3: save the view as an .mp4 file
parser.add_argument("--results_json", type=str, default="")         # 6.3: save results for run_robustness.py
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


# ── 6.3: make the RENDERED attacker bigger or smaller ─────────────────────────
# Finds the attacker's config (any config whose name or prim path contains "attacker")
# and multiplies its spawn scale BEFORE the scene is built. Only the picture changes:
# the simulator's true distance and capture radius stay exactly the same.
def scale_attacker(cfg, s):
    scaled = []
    for holder in (cfg, getattr(cfg, "scene", None)):
        if holder is None:
            continue
        for name in dir(holder):
            if name.startswith("_"):
                continue
            try:
                v = getattr(holder, name)
            except Exception:
                continue
            path = getattr(v, "prim_path", None)
            spawn = getattr(v, "spawn", None)
            if isinstance(path, str) and "attacker" in (name + path).lower() and spawn is not None \
                    and hasattr(spawn, "scale"):
                base = spawn.scale if spawn.scale is not None else (1.0, 1.0, 1.0)
                spawn.scale = tuple(b * s for b in base)
                scaled.append(f"{name} ({path}) scale {base} -> {spawn.scale}")
    if not scaled:
        raise RuntimeError("--attacker_scale: no attacker config with a spawn scale was found. "
                           "Run: findstr /n \"attacker\" quadcopter_env.py and paste the config lines.")
    for line in scaled:
        print(f"[6.3] attacker rendered x{s}: {line}")


if args.attacker_scale != 1.0:
    scale_attacker(env_cfg, args.attacker_scale)

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

# ── 6.3: random lighting at every chase ───────────────────────────────────────
# Collects every light in the scene once, remembers its original brightness, then at each
# reset sets a new brightness (30–150 % of the original) and a slight colour tint.
lights = []
if args.light_random:
    import omni.usd
    from pxr import Gf
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.HasAttribute("inputs:intensity"):
            lights.append((prim, prim.GetAttribute("inputs:intensity").Get()))
    print(f"[6.3] random lighting on {len(lights)} light(s): {[str(p.GetPath()) for p, _ in lights]}")


def randomize_lights():
    for prim, base in lights:
        prim.GetAttribute("inputs:intensity").Set(float(base) * float(np.random.uniform(0.3, 1.5)))
        if prim.HasAttribute("inputs:color"):
            prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*np.random.uniform(0.75, 1.0, 3).tolist()))
RADIUS = raw_env.cfg.capture_radius       # simulator capture distance in metres, set in Chapter 2.1

# ── 6.3: live view of what the defender sees ──────────────────────────────────
# Draws on each camera picture: the detector's box, the image centre, the TRUE distance
# (from the simulator), the camera's ESTIMATED distance and the time-to-contact.
# The estimated distance needs the attacker's real width, which a camera alone cannot
# know. Here we use the rendered drone's width (0.13 m × --attacker_scale), so the number
# shows how well the camera COULD judge distance if the width were known. The capture
# decision does not use it: that is time-to-contact, which needs no width.
VIEW = args.show_view or bool(args.record)
if VIEW:
    import cv2
    LENS = raw_env.cfg.cam_focal_mm / raw_env.cfg.cam_aperture_mm           # 3.1 Part E
    SPAN = getattr(raw_env.cfg, "attacker_span_m", 0.13) * args.attacker_scale
    view_state = {"window": args.show_view, "writer": None}


def put(img, text, y, color):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)   # outline
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def show_view(rgb, box, readings, true_dist, ttc, cam_captured, label, warming_up):
    if not VIEW or rgb is None:
        return
    img = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    cv2.drawMarker(img, (w // 2, h // 2), (200, 200, 200), cv2.MARKER_CROSS, 20, 1)   # "dead ahead"
    if box is not None:
        xc, yc, bw, bh = [float(v) for v in box]
        p1, p2 = (int(xc - bw / 2), int(yc - bh / 2)), (int(xc + bw / 2), int(yc + bh / 2))
        cv2.rectangle(img, p1, p2, (0, 0, 255) if cam_captured else (0, 255, 0), 2)
        cv2.line(img, (w // 2, h // 2), (int(xc), int(yc)), (0, 255, 255), 1)          # bearing
    green, gray, white, red = (0, 255, 0), (170, 170, 170), (255, 255, 255), (0, 0, 255)
    put(img, label, 20, white)
    put(img, f"true distance (simulator): {true_dist:.2f} m", 44, white)
    if warming_up:
        put(img, "first frames after reset: ignored", 68, gray)
    elif readings[6] == 1.0 and readings[2] > 0:
        put(img, f"camera estimate: {LENS * SPAN / readings[2]:.2f} m  (assumes {SPAN * 100:.0f} cm width)", 68, green)
    else:
        put(img, "attacker NOT seen - holding last reading", 68, gray)
    put(img, f"time-to-contact: {ttc:.2f} s   (capture below {TTC:.2f} s)" if ttc != float("inf")
        else "time-to-contact: --  (needs 3 sightings in a row)", 92, white)
    if cam_captured:
        put(img, "CAMERA SAYS: CAPTURED", 120, red)
    if args.record:
        if view_state["writer"] is None:
            view_state["writer"] = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                                   int(round(1 / DT)), (w, h))
        view_state["writer"].write(img)
    if view_state["window"]:
        try:
            cv2.imshow("Defender camera", img)
            cv2.waitKey(1)
        except cv2.error:
            view_state["window"] = False
            print("[6.3] cannot open a window (opencv-python-headless is installed). "
                  "Use --record view.mp4 instead, or: pip install opencv-python")


def run_round(splice):
    """Run args.episodes chases. splice=True → camera readings replace slots 6..12."""
    name = "camera" if splice else "calculated"
    print(f"\n── Round '{name}': splice {'ON' if splice else 'OFF'}, {args.episodes} chases "
          f"(camera time-to-contact threshold {TTC:.2f} s, capture radius {RADIUS:.2f} m) ──")
    res = {"captures": 0, "crashes": 0, "timeouts": 0, "confirmed": 0, "near": 0,
           "steps": 0, "seen": 0, "jitter": []}
    obs, _ = env.reset()
    raw_env.episode_length_buf.zero_()                      # full time for every chase (see note below)
    randomize_lights()                                      # 6.3, does nothing without --light_random
    bridge = CameraReadingBridge()
    ep = {"steps": 0, "seen": 0, "min_ttc": float("inf"), "last_dist": 0.0,
          "cam_says": False, "jitter": []}
    prev_action = None
    done_eps = 0

    while app.is_running() and done_eps < args.episodes:
        # 1) SEE — this step's rendered frame, (1, H, W, 3) uint8 → numpy HWC
        # The first pictures after a reset are rendered before the scene moved, so they still
        # show the end of the previous chase. Treat them as "nothing seen".
        rgb = raw_env._camera.data.output["rgb"][0].cpu().numpy()
        warming_up = ep["steps"] < SKIP_FRAMES
        box = None if warming_up else detector.detect(rgb)
        readings = bridge.update(box)                           # 7 numbers

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
        show_view(rgb, box, readings, ep["last_dist"], ttc, bridge.captured_ttc(TTC, DT),
                  f"{name} round | chase {done_eps + 1}/{args.episodes} | step {ep['steps']}", warming_up)
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
            randomize_lights()                               # 6.3, does nothing without --light_random
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

CONDITION = f"attacker x{args.attacker_scale:g}" + (", random light" if args.light_random else "")
print(f"\n================ 6.2 SUMMARY ({CONDITION}) ================")
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
if "camera" in rounds:
    c = rounds["camera"]
    print("\nCopy this line into project_notes.txt (6.3 robustness table):")
    print(f"6.3 [{CONDITION}] ({args.episodes} chases): camera capture {100*c['rate']:.0f}% | "
          f"crashes {c['crashes']} | camera saw {100*c['seen_share']:.0f}% | "
          f"confirmed {c['confirmed']}/{c['captures']} | nearly {c['near']}/{c['captures']} | "
          f"jitter {c['jitter_mean']:.3f}")
if len(rounds) == 2:
    print("\nCopy this line into project_notes.txt:")
    print(f"6.2 capture rate ({args.episodes} chases): calculated {100*rounds['calculated']['rate']:.0f}% | "
          f"camera {100*rounds['camera']['rate']:.0f}% | gap "
          f"{100*(rounds['calculated']['rate'] - rounds['camera']['rate']):.0f} points")

if args.results_json:                                       # 6.3: machine-readable results
    import json
    keep = ("episodes", "rate", "captures", "crashes", "timeouts", "confirmed", "near", "seen_share", "jitter_mean")
    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump({"condition": CONDITION, "attacker_scale": args.attacker_scale,
                   "light_random": args.light_random, "all_checks_ok": all_ok,
                   "rounds": {k: {x: v[x] for x in keep} for k, v in rounds.items()}}, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))   # numpy numbers → plain
    print(f"[6.3] results saved: {args.results_json}")

if VIEW:
    if view_state["writer"] is not None:
        view_state["writer"].release()
        print(f"[6.3] video saved: {args.record}")
    if args.show_view:
        cv2.destroyAllWindows()
env.close()
app.close()