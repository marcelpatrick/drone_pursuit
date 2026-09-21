"""
check_3_1.py  -  automatic Checkpoint 3.1 for drone_pursuit_tutorial.md
=======================================================================

WHAT IT DOES, IN ONE PICTURE
    project_notes.txt ──┐
                        ├──►  compare numbers  ──►  item 4 (+ decimation for item 1)
    your env cfg ───────┘
    your env  ──► run 64 drones for 300 steps, secretly recording every
                  camera reading before and after the delay buffer
              ──► item 1: no crash, 17 numbers every step, in the right slots
              ──► item 2: readings make sense, delayed values lag by the drawn delay
              ──► item 3: `visible` drops to 0, held values freeze, resets clear them

    Half the drones hover (zero commands) so the circling attacker sweeps across
    their camera; the other half fly randomly so distances vary.

WHERE IT LIVES
    C:\\projects\\drone_pursuit\\drone_pursuit\\scripts\\checks\\check_3_1.py

RUN (environment: env_drone, from any folder)
    python C:\\projects\\drone_pursuit\\drone_pursuit\\scripts\\checks\\check_3_1.py

OPTIONAL FLAGS
    --notes <path>     project_notes.txt, if not in the default places
    --num_envs 64      more drones = more chances to see every situation
    --steps 300        20 Hz x 300 = 15 s, enough for every env to reset once
    --show             open the viewer instead of running headless

NOTE: run this BEFORE adding 3.3's Gaussian reading noise. Noise makes held
values wiggle on purpose, so item 3 would report a false failure.
"""

import argparse
import math
import os
import re
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Automatic Checkpoint 3.1")
parser.add_argument("--task", default="Template-Drone-Pursuit-Direct-v0")
parser.add_argument("--notes", default=None, help="path to project_notes.txt")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--show", action="store_true", help="open the viewer")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.show
app = AppLauncher(args).app

import torch                                                     # noqa: E402
import gymnasium as gym                                          # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms        # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry          # noqa: E402
import drone_pursuit.tasks  # noqa: F401,E402  (registers your task)

# ─────────────────────────────────────────────────────────────────────────────
# Result bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
TITLES = {
    1: "Env steps with no shape errors at observation_space = 17 and decimation matching your measured rate",
    2: "Printed readings behave sensibly, and delayed values visibly lag",
    3: "`visible` drops to 0 when the attacker leaves the frame, with held values staying frozen",
    4: "Your cfg contains the three numbers from project_notes.txt, not the defaults",
}
RESULTS = {k: [] for k in TITLES}


def ok(i, msg):
    RESULTS[i].append(("PASS", msg, None))


def fail(i, msg, fix):
    RESULTS[i].append(("FAIL", msg, fix))


def warn(i, msg, fix=None):
    RESULTS[i].append(("WARN", msg, fix))


def print_report():
    icon = {"PASS": "  [ok]  ", "FAIL": "  [XX]  ", "WARN": "  [??]  "}
    print("\n" + "=" * 78)
    print(" CHECKPOINT 3.1 - AUTOMATIC REPORT")
    print("=" * 78)
    any_fail = False
    for i, title in TITLES.items():
        items = RESULTS[i]
        if any(s == "FAIL" for s, _, _ in items):
            verdict, any_fail = "FAIL", True
        elif not items or any(s == "WARN" for s, _, _ in items):
            verdict = "NOT FULLY CONFIRMED"
        else:
            verdict = "PASS"
        print(f"\n{i}. {title}\n   -> {verdict}")
        if not items:
            print("        (not run - an earlier failure stopped it)")
        for status, msg, fix in items:
            print(f"{icon[status]}{msg}")
            if fix:
                print(f"          Fix: {fix}")
    print("\n" + "=" * 78)
    print(" RESULT:", "FIX THE [XX] LINES ABOVE" if any_fail else "Checkpoint 3.1 passed")
    print("=" * 78 + "\n")
    return any_fail


# ─────────────────────────────────────────────────────────────────────────────
# Reading project_notes.txt
# ─────────────────────────────────────────────────────────────────────────────
def find_notes(path):
    cands = [path] if path else []
    base = os.environ.get("DRONE_PURSUIT_DIR")
    if base:
        cands += [os.path.join(base, "project_notes.txt"),
                  os.path.join(base, "drone_pursuit", "project_notes.txt")]
    cands += [r"C:\projects\drone_pursuit\drone_pursuit\project_notes.txt",
              r"C:\projects\drone_pursuit\project_notes.txt"]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def parse_notes(p):
    txt = open(p, encoding="utf-8", errors="ignore").read()
    out = {}
    m = re.search(r"control\s*rate\s*:\s*([\d.]+)\s*Hz", txt, re.I)
    out["hz"] = float(m.group(1)) if m else None
    m = re.search(r"video\s*delay\s*:\s*([\d.]+)\s*ms", txt, re.I)
    out["delay_mean_ms"] = float(m.group(1)) if m else None
    m = re.search(r"([\d.]+)\s*(?:\u2013|\u2014|-|to)\s*([\d.]+)\s*ms\s*range", txt, re.I)
    out["delay_lo_ms"], out["delay_hi_ms"] = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = re.search(r"telemetry\s*available\s*:([^\n]*)", txt, re.I)
    out["telemetry"] = m.group(1).strip() if m else None
    m = re.search(r"camera\s*:\s*(\d+)\s*x\s*(\d+)", txt, re.I)
    out["res"] = (int(m.group(1)), int(m.group(2))) if m else None
    m = re.search(r"camera\s*:[^\n]*?([\d.]+)\s*deg", txt, re.I)
    out["fov_deg"] = float(m.group(1)) if m else None
    return out


def obs_dim(space):
    if isinstance(space, int):
        return space
    try:
        return gym.spaces.flatdim(space)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 4 (and the decimation half of item 1): cfg vs project_notes.txt
# ─────────────────────────────────────────────────────────────────────────────
env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
phys_hz = 1.0 / env_cfg.sim.dt
dec = env_cfg.decimation
step_dt = env_cfg.sim.dt * dec

odim = obs_dim(env_cfg.observation_space)
if odim == 17:
    ok(1, "cfg observation_space = 17")
else:
    fail(1, f"cfg observation_space = {odim}, expected 17",
         "3.1 Part C: set observation_space = 17 in QuadcopterEnvCfg")

notes_path = find_notes(args.notes)
notes = parse_notes(notes_path) if notes_path else None
if notes is None:
    fail(4, "project_notes.txt not found",
         "pass --notes <path>, or create it as in 1.4 Step 7")
    fail(1, "cannot compare decimation - no measured rate available",
         "see item 4")
else:
    ok(4, f"read {notes_path}")
    hz = notes["hz"]

    # --- number 1: control rate -> decimation ---------------------------------
    if hz is None:
        fail(4, "no 'control rate : NN Hz' line in project_notes.txt",
             "add it exactly as in 1.4 Step 7")
        fail(1, "cannot compare decimation - control rate missing from notes", "see item 4")
    else:
        exp_dec = max(1, round(phys_hz / hz))
        ctrl = phys_hz / dec
        if dec == exp_dec:
            msg = f"decimation = {dec} -> {ctrl:.1f} Hz control, matches your measured {hz:g} Hz"
            ok(1, msg)
            ok(4, "control rate: " + msg)
        else:
            msg = f"decimation = {dec} gives {ctrl:.1f} Hz, but you measured {hz:g} Hz"
            fix = f"3.1 Part B: set decimation = {exp_dec}  (physics {phys_hz:.0f} Hz / {hz:g} Hz)"
            fail(1, msg, fix)
            fail(4, "control rate: " + msg, fix)

    # --- number 2: video delay -> obs_delay_min / obs_delay_max ---------------
    dmin = getattr(env_cfg, "obs_delay_min", None)
    dmax = getattr(env_cfg, "obs_delay_max", None)
    if dmin is None or dmax is None:
        fail(4, "cfg has no obs_delay_min / obs_delay_max",
             "3.1 Part D Edit 1: add both to QuadcopterEnvCfg")
    elif hz is not None and notes["delay_lo_ms"] is not None:
        lo_s = notes["delay_lo_ms"] / 1000 * hz
        hi_s = notes["delay_hi_ms"] / 1000 * hz
        ok_min = {math.floor(lo_s), round(lo_s)}
        ok_max = {round(hi_s), math.ceil(hi_s)}
        rng = f"{notes['delay_lo_ms']:g}-{notes['delay_hi_ms']:g} ms at {hz:g} Hz = {lo_s:.2f}-{hi_s:.2f} steps"
        if dmin in ok_min and dmax in ok_max and dmin <= dmax:
            ok(4, f"video delay: obs_delay_min = {dmin}, obs_delay_max = {dmax}  ({rng})")
        else:
            fail(4, f"video delay: cfg has obs_delay_min = {dmin}, obs_delay_max = {dmax}, but {rng}",
                 f"3.1 Part D Edit 1: obs_delay_min = {round(lo_s)}, obs_delay_max = {math.ceil(hi_s)}")
    elif hz is not None and notes["delay_mean_ms"] is not None:
        mean_s = notes["delay_mean_ms"] / 1000 * hz
        if dmin <= mean_s <= dmax:
            warn(4, f"video delay: only a mean ({notes['delay_mean_ms']:g} ms = {mean_s:.1f} steps) in notes; "
                    f"it sits inside your {dmin}-{dmax} range, but the range itself cannot be checked",
                 "add the measured range to project_notes.txt, e.g. '99-219 ms range'")
        else:
            fail(4, f"video delay: mean {mean_s:.1f} steps lies outside obs_delay_min..max = {dmin}..{dmax}",
                 "3.1 Part D Edit 1: recompute both from project_notes.txt")
    else:
        fail(4, "no 'video delay' line in project_notes.txt", "add it as in 1.4 Step 7")

    # --- number 3: telemetry -> the 17 inputs ----------------------------------
    if notes["telemetry"]:
        if odim == 17:
            ok(4, f"telemetry recorded ('{notes['telemetry'][:45]}...') and cfg uses the 17-input layout")
        else:
            fail(4, "telemetry recorded, but the cfg is not using the 17-input layout",
                 "3.1 Part C: observation_space = 17")
    else:
        fail(4, "no 'telemetry available' line in project_notes.txt", "add it as in 1.4 Step 7")

    # --- bonus: camera lens matches the notes ----------------------------------
    need = ["cam_width", "cam_height", "cam_focal_mm", "cam_aperture_mm", "attacker_span_m"]
    missing = [n for n in need if getattr(env_cfg, n, None) is None]
    if missing:
        fail(4, f"cfg is missing camera values: {', '.join(missing)}",
             "3.1 Part E: add the camera-model block to QuadcopterEnvCfg")
    else:
        fov = math.degrees(2 * math.atan(env_cfg.cam_aperture_mm / (2 * env_cfg.cam_focal_mm)))
        if notes["fov_deg"] is not None:
            if abs(fov - notes["fov_deg"]) <= 3:
                ok(4, f"camera: cfg lens gives {fov:.0f} deg, notes say {notes['fov_deg']:g} deg")
            else:
                good = env_cfg.cam_aperture_mm / (2 * math.tan(math.radians(notes["fov_deg"] / 2)))
                fail(4, f"camera: cfg lens gives {fov:.0f} deg, notes say {notes['fov_deg']:g} deg",
                     f"3.1 Part E: cam_focal_mm = {good:.1f}")
        if notes["res"]:
            a_n = notes["res"][0] / notes["res"][1]
            a_c = env_cfg.cam_width / env_cfg.cam_height
            if abs(a_n - a_c) > 0.02:
                fail(4, f"camera: cfg frame shape {env_cfg.cam_width}x{env_cfg.cam_height} does not match "
                        f"the drone's {notes['res'][0]}x{notes['res'][1]}",
                     "3.1 Part E: keep cam_width / cam_height the same ratio as the real camera")

    # --- "not the defaults printed here" ---------------------------------------
    if hz == 20 and notes["delay_lo_ms"] == 99 and notes["delay_hi_ms"] == 219:
        warn(4, "your notes contain the tutorial's EXAMPLE numbers exactly (20 Hz, 99-219 ms)",
             "if you copied them rather than measured, redo 1.4 Steps 5-6 with your Tello")

# ─────────────────────────────────────────────────────────────────────────────
# Build the env and install a "spy" on the delay buffer
# ─────────────────────────────────────────────────────────────────────────────
env = None
LOG = []
try:
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
except Exception:
    fail(1, "the environment could not be created:\n" + traceback.format_exc(limit=3),
         "read the last line of the error above; it names the file and line")

raw = env.unwrapped if env is not None else None
if raw is not None:
    REQUIRED = {
        "_camera_readings": "3.1 Part F", "_delayed_readings": "3.1 Part D Edit 3",
        "_reading_history": "3.1 Part D Edit 2", "_delay_steps": "3.1 Part D Edit 2",
        "_prev_bx": "3.1 Part D Edit 2", "_prev_by": "3.1 Part D Edit 2",
        "_prev_asz": "3.1 Part D Edit 2", "_attacker": "Chapter 2.1",
    }
    for name, part in REQUIRED.items():
        if not hasattr(raw, name):
            fail(1, f"env has no `{name}`", f"{part} adds it")
    if any(st == "FAIL" and msg.startswith("env has no") for st, msg, _ in RESULTS[1]):
        raw = None

if raw is not None:
    N, dev = raw.num_envs, raw.device
    exp_hist = (N, raw.cfg.obs_delay_max + 1, 7)
    if tuple(raw._reading_history.shape) != exp_hist:
        fail(2, f"_reading_history has shape {tuple(raw._reading_history.shape)}, expected {exp_hist}",
             "3.1 Part D Edit 2: torch.zeros(num_envs, obs_delay_max + 1, 7)")

    original = raw._delayed_readings

    def spy(fresh):
        out = original(fresh)
        rel_b, _ = subtract_frame_transforms(
            raw._robot.data.root_pos_w, raw._robot.data.root_quat_w, raw._attacker.data.root_pos_w)
        LOG.append({
            "fresh": fresh.detach().float().cpu().clone(),
            "out": out.detach().float().cpu().clone(),
            "delay": raw._delay_steps.detach().cpu().clone(),
            "reset": (raw.episode_length_buf == 0).detach().cpu().clone(),
            "dist": raw._dist.detach().float().cpu().clone(),
            "rel": rel_b.detach().float().cpu().clone(),
        })
        return out

    raw._delayed_readings = spy   # _get_observations now calls the spy

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 1: step the env, check every observation
    # ─────────────────────────────────────────────────────────────────────────
    shape_err = None

    def check_obs(obs, when):
        global shape_err
        if shape_err:
            return
        if not isinstance(obs, dict) or "policy" not in obs:
            shape_err = f"{when}: observations are not a dict with a 'policy' entry"
            fail(1, shape_err, "3.1 Part G must end with: return {\"policy\": obs}")
            return
        p = obs["policy"]
        if tuple(p.shape) != (N, 17):
            shape_err = f"{when}: observation shape is {tuple(p.shape)}, expected ({N}, 17)"
            fail(1, shape_err, "3.1 Part G: torch.cat must join 3 + 3 + 7 + 4 = 17 numbers; count your pieces")
            return
        if not torch.isfinite(p).all():
            bad = (~torch.isfinite(p)).any(0).nonzero().flatten().tolist()
            shape_err = f"{when}: NaN/inf in observation slots {bad}"
            fail(1, shape_err, "slots 6-12 = camera (Part F divides by distance); 13-16 = commands (Part A)")

    try:
        act_dim = gym.spaces.flatdim(raw.single_action_space)
        if act_dim != 4:
            fail(1, f"action space has {act_dim} numbers, expected 4", "3.1 Part A / action_space = 4")
        torch.manual_seed(0)
        obs, _ = env.reset()
        check_obs(obs, "after reset")
        half = N // 2
        for t in range(args.steps):
            a = torch.zeros(N, act_dim, device=dev)                  # first half hovers
            a[half:] = torch.rand(N - half, act_dim, device=dev) * 2 - 1   # second half wanders
            obs, _, _, _, _ = env.step(a)
            check_obs(obs, f"step {t}")
            if shape_err:
                break
        if not shape_err:
            ok(1, f"{args.steps} steps x {N} envs ran with no errors; every observation was ({N}, 17)")
    except Exception:
        shape_err = "crash"
        fail(1, "the env crashed while stepping:\n" + traceback.format_exc(limit=4),
             "the last line of the error names the file and line to fix")

    # slot order: are the 17 numbers where Chapter 7.2 expects them?
    if not shape_err and LOG:
        p = obs["policy"].float().cpu()
        noisy = getattr(raw.cfg, "observation_noise_model", None) is not None
        slots = [
            ("own speed", 0, 3, raw._robot.data.root_lin_vel_b),
            ("own tilt", 3, 6, raw._robot.data.projected_gravity_b),
            ("camera (delayed)", 6, 13, LOG[-1]["out"]),
            ("last commands", 13, 17, raw._actions),
        ]
        wrong = [f"{nm} (slots {a}-{b - 1})" for nm, a, b, ref in slots
                 if (p[:, a:b] - ref.float().cpu()).abs().max() > 1e-4]
        if not wrong:
            ok(1, "slot order correct: speed 0-2, tilt 3-5, camera 6-12, commands 13-16")
        elif noisy:
            warn(1, "observation noise is on, so slot order could not be compared exactly")
        else:
            fail(1, "these pieces are not in their slots: " + ", ".join(wrong),
                 "3.1 Part G: torch.cat order must be lin_vel_b, projected_gravity_b, delayed, self._actions")

    if not LOG:
        fail(2, "_get_observations never called _delayed_readings",
             "3.1 Part G: `delayed = self._delayed_readings(fresh)` and put `delayed` in torch.cat")

# ─────────────────────────────────────────────────────────────────────────────
# ITEMS 2 and 3: analyse the recording
# ─────────────────────────────────────────────────────────────────────────────
if raw is not None and len(LOG) > 5:
    try:
        Fr = torch.stack([r["fresh"] for r in LOG])      # (T, N, 7) before delay
        Out = torch.stack([r["out"] for r in LOG])       # (T, N, 7) after delay
        Dl = torch.stack([r["delay"] for r in LOG])      # (T, N)
        Rs = torch.stack([r["reset"] for r in LOG])      # (T, N) just reset?
        Ds = torch.stack([r["dist"] for r in LOG])       # (T, N)
        Rel = torch.stack([r["rel"] for r in LOG])       # (T, N, 3)
        T, N = Fr.shape[:2]
        bx, by, asz, vis = Fr[..., 0], Fr[..., 1], Fr[..., 2], Fr[..., 6]
        V = vis > 0.5
        env_idx = torch.arange(N)

        # --- a small printout of env 0, like the manual sanity check ----------
        print("\n env 0, first 15 steps  (fresh = camera now, delayed = what the policy sees)")
        print("  step   dist   bx_fresh  bx_delayed  visible  delay")
        for t in range(min(15, T)):
            print(f"  {t:4d}  {Ds[t, 0]:5.2f}   {bx[t, 0]:+7.3f}    {Out[t, 0, 0]:+7.3f}     "
                  f"{int(vis[t, 0])}       {int(Dl[t, 0])}")

        # ── ITEM 2a: delay buffer ───────────────────────────────────────────
        lo_c, hi_c = raw.cfg.obs_delay_min, raw.cfg.obs_delay_max
        if Dl.min() < lo_c or Dl.max() > hi_c:
            fail(2, f"drawn delays span {int(Dl.min())}-{int(Dl.max())}, outside {lo_c}-{hi_c}",
                 "3.1 Part D: torch.randint(obs_delay_min, obs_delay_max + 1, ...)")
        distinct = sorted(set(Dl.flatten().tolist()))
        if len(distinct) == 1 and hi_c > lo_c and N >= 8:
            warn(2, f"every env drew the same delay ({distinct[0]})",
                 "3.1 Part D: the delay should be drawn per env, in __init__ and _reset_idx")

        def lag_mismatch(shift):
            bad, example = 0, None
            c = torch.full((N,), -1, dtype=torch.long)
            for t in range(T):
                c = torch.where(Rs[t], torch.zeros_like(c), c + 1)
                d = Dl[t].long() + shift
                have = (d >= 0) & (d <= c)
                src = (t - d).clamp(min=0)
                exp = torch.where(have.unsqueeze(-1), Fr[src, env_idx], torch.zeros(N, 7))
                b = (Out[t] - exp).abs().amax(-1) > 1e-5
                bad += int(b.sum())
                if example is None and b.any():
                    e = int(b.nonzero()[0])
                    example = (t, e, int(Dl[t, e]), exp[e, 0].item(), Out[t, e, 0].item())
            return bad, example

        bad0, ex = lag_mismatch(0)
        total = T * N
        if bad0 == 0:
            ms = Dl.float().mean().item() * step_dt * 1000
            ok(2, f"delayed values lag fresh ones by exactly the drawn delay "
                  f"({lo_c}-{hi_c} steps, average {ms:.0f} ms) in all {total} checks")
        else:
            no_delay = int(((Out - Fr).abs().amax(-1) > 1e-5).sum())
            bad_m, _ = lag_mismatch(-1)
            bad_p, _ = lag_mismatch(+1)
            t, e, d, exp_v, got_v = ex
            detail = (f"{bad0}/{total} delayed readings are wrong "
                      f"(e.g. step {t}, env {e}, delay {d}: expected bx {exp_v:+.3f}, got {got_v:+.3f})")
            if no_delay == 0:
                fix = "the buffer returns the fresh reading unchanged - check Part D Edit 3 uses gather with _delay_steps"
            elif bad_m == 0:
                fix = "off by one step (too short) - Part D Edit 3: roll FIRST, then write slot 0, then gather"
            elif bad_p == 0:
                fix = "off by one step (too long) - Part D Edit 3: history must be obs_delay_max + 1 wide"
            else:
                fix = "Part D: check _reset_idx clears _reading_history[env_ids] and redraws _delay_steps"
            fail(2, detail, fix)

        # ── ITEM 2b: readings behave sensibly ───────────────────────────────
        nv = int(V.sum())
        if nv < 20:
            fail(2, f"the attacker was visible in only {nv} of {total} readings",
                 "3.1 Part F: check the visible condition and the minus signs in bearing_x / bearing_y")
        else:
            bad_rng = int(((bx[V].abs() > 1 + 1e-4) | (by[V].abs() > 1 + 1e-4) | (asz[V] <= 0)).sum())
            if bad_rng:
                fail(2, f"{bad_rng} visible readings have |bearing| > 1 or size <= 0",
                     "3.1 Part F: visible must require |bearing_x| < 1, |bearing_y| < 1, ang_size > 0.012")
            else:
                ok(2, f"all {nv} visible readings are inside the frame (bearings between -1 and +1)")

            # ang_size rises as distance falls
            d_v = Ds[V].clamp(min=0.05)
            if d_v.std() < 0.05:
                warn(2, "distance barely changed during the run; ang_size vs distance not tested",
                     "rerun with --steps 600")
            else:
                corr = torch.corrcoef(torch.stack([asz[V].log(), d_v.log()]))[0, 1].item()
                k = raw.cfg.cam_focal_mm / raw.cfg.cam_aperture_mm * raw.cfg.attacker_span_m
                ratio = (asz[V] * d_v / k).median().item()
                if corr > -0.9:
                    fail(2, f"ang_size does not rise as distance falls (correlation {corr:+.2f}, should be near -1)",
                         "3.1 Part F: ang_size must DIVIDE by self._dist, and Part G must set _dist before _camera_readings")
                elif abs(ratio - 1) > 0.05:
                    warn(2, f"ang_size rises as distance falls, but is {ratio:.2f}x the lens formula",
                         "3.1 Part E/F: check cam_focal_mm, cam_aperture_mm, attacker_span_m")
                else:
                    ok(2, f"ang_size rises as distance falls (correlation {corr:+.2f})")

            # bearing_x flips sign as the attacker crosses the frame
            pos, neg = int((bx[V] > 0.1).sum()), int((bx[V] < -0.1).sum())
            if pos and neg:
                ok(2, f"bearing_x takes both signs ({pos} right-of-centre, {neg} left-of-centre readings)")
            else:
                warn(2, "bearing_x never flipped sign in this run", "rerun with --steps 600 or --num_envs 128")

            # left/right and up/down not swapped
            for name, col, axis in (("bearing_x", bx, 1), ("bearing_y", by, 2)):
                side = Rel[V][:, axis]
                fwd = Rel[V][:, 0].clamp(min=0.05)
                m = (side / fwd).abs() > 0.05
                if m.sum() < 10:
                    continue
                agree = (torch.sign(col[V][m]) == -torch.sign(side[m])).float().mean().item()
                if agree > 0.95:
                    ok(2, f"{name} points the right way ({agree:.0%} of readings)")
                elif agree < 0.05:
                    fail(2, f"{name} is mirrored: the attacker's side and the reading's sign always disagree",
                         f"3.1 Part F: {name} = -(rel_b[:, {axis}] / safe_fwd) * ...  (keep the leading minus)")
                else:
                    fail(2, f"{name} sign matches the attacker's side only {agree:.0%} of the time",
                         "3.1 Part F: check which rel_b column each bearing uses (1 = left/right, 2 = up/down)")

        # ── ITEM 3: visible drops, held values freeze, resets clear ─────────
        if not bool(((vis == 0) | (vis == 1)).all()):
            fail(3, "`visible` contains values other than 0 and 1",
                 "3.1 Part F: return visible.float() from a True/False condition")

        not_reset = ~Rs[1:]
        drops = int((V[:-1] & ~V[1:] & not_reset).sum())
        if drops:
            ok(3, f"`visible` dropped from 1 to 0 {drops} times as the attacker left the frame")
        else:
            warn(3, "the attacker never left the frame during this run, so the drop could not be seen",
                 "rerun with --steps 600 or --num_envs 128")

        hidden = (~V[1:]) & not_reset
        if hidden.sum() == 0:
            warn(3, "no hidden steps to test held values on", "rerun with more --steps")
        else:
            moved = (Fr[1:, :, 0:3] - Fr[:-1, :, 0:3]).abs().amax(-1)[hidden]
            rates = Fr[1:, :, 3:6].abs().amax(-1)[hidden]
            if moved.max() > 1e-5:
                fail(3, f"held values moved while hidden in {int((moved > 1e-5).sum())} readings",
                     "3.1 Part G: the three torch.where(vis > 0.5, new, self._prev_*) lines must run BEFORE _prev_* is updated")
            elif rates.max() > 1e-5:
                fail(3, "values froze, but the three rate-of-change numbers were not 0 while hidden",
                     "3.1 Part G: compute d_bx, d_by, d_asz AFTER the torch.where hold lines")
            else:
                ok(3, f"held values stayed frozen (and their changes read 0) across {int(hidden.sum())} hidden readings")

        # resets must clear the held values (Part D's _reset_idx block)
        mid = Rs.clone()
        mid[0] = False
        if mid.sum() == 0:
            warn(3, "no env reset mid-run, so reset clearing was not tested", "rerun with more --steps")
        else:
            hid_r = mid & ~V
            vis_r = mid & V
            bad_h = int((Fr[..., 0:3].abs().amax(-1)[hid_r] > 1e-5).sum())
            bad_v = int(((Fr[..., 3:6] - Fr[..., 0:3]).abs().amax(-1)[vis_r] > 1e-5).sum())
            if bad_h or bad_v:
                fail(3, f"{bad_h + bad_v} fresh episodes started with the previous episode's last sighting",
                     "3.1 Part D, _reset_idx: zero self._prev_bx / _prev_by / _prev_asz for env_ids")
            else:
                ok(3, f"all {int(mid.sum())} mid-run resets started from a clean slate")

    except Exception:
        fail(2, "the checker itself crashed while analysing:\n" + traceback.format_exc(limit=3),
             "paste this message back into the chat")

failed = print_report()
if env is not None:
    env.close()
app.close()
sys.exit(1 if failed else 0)