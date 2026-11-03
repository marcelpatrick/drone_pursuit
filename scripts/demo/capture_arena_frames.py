# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\capture_arena_frames.py
# ── NEW file — the whole content, nothing to insert into.
# ── Loads : your Ch.3 checkpoint, passed on the command line
# ── Writes: data\arena_frames\frame_XXX_dDD.DDm_vN.png        (clean, for the detector test in 5.2)
# ──         data\arena_frames\readings.csv                     (the calculated readings, for 6.1 Step 2,
# ──                                                              plus the drone's real pixels, for 5.2 Step 3)
# ──         data\arena_frames_check\frame_XXX_dDD.DDm_vN.png  (red box = where the attacker should be,
# ──                                                              green box = the pixels the attacker really covers)
# ──   v1 = attacker in the camera's view right now, v0 = out of view (or too small to detect)
# ── EDITED after drone_v3's 5.2 test:
# ──   1. skips the picture of any step in which a chase ended (the picture showed the old chase,
# ──      the readings the new one)
# ──   2. the camera also records which object each pixel shows, so every row says how many pixels
# ──      the attacker really covers and whether it is big enough to grade (same 10 x 4 px rule as
# ──      convert_to_yolo.py uses for training labels)

import argparse, os, csv
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)   # your Ch.3 best_agent.pt
parser.add_argument("--num_frames", type=int, default=200)     # ← EDITED: 200 (was 40), so each distance row has enough frames
parser.add_argument("--every", type=int, default=5)            # 1 picture every 5 control steps = 0.25 s at 20 Hz
parser.add_argument("--out", type=str,
                    default=r"C:\projects\drone_pursuit\drone_pursuit\data\arena_frames")
parser.add_argument("--fixed_span", type=float, default=0.13)  # attacker width used in the calculation, metres.
                                                               # 0.13 = the Crazyflie drawn on screen, so the
                                                               # calculated size matches the picture (6.1 Step 2).
                                                               # Pass 0 to keep 3.2B's random width instead.
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True                                     # the camera renders nothing without this
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw
from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
import drone_pursuit.tasks  # noqa: F401  (registers your task)

TASK = "Template-Drone-Pursuit-Direct-v0"
SEG = "instance_id_segmentation_fast"                          # ← NEW: per pixel, the ID of the object it shows
MIN_LONG, MIN_THIN = 10, 4                                     # ← NEW: same limits as convert_to_yolo.py

env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
env_cfg.scene.num_envs = 1                                     # one chase is enough
if args.fixed_span > 0:                                        # lock the width: same in the calculation as on screen
    env_cfg.attacker_span_range = (args.fixed_span, args.fixed_span)
# ← NEW: for this capture only, the camera also outputs object IDs (training settings unchanged)
env_cfg.tiled_camera.data_types = ["rgb", SEG]
env_cfg.tiled_camera.colorize_instance_id_segmentation = False # plain ID numbers, not colours
env = gym.make(TASK, cfg=env_cfg, render_mode=None)
raw_env = env.unwrapped                                        # to reach the camera and the true distance
env = SkrlVecEnvWrapper(env)

# rebuild the skrl agent exactly as train.py does, then load your checkpoint
agent_cfg = load_cfg_from_registry(TASK, "skrl_cfg_entry_point")
runner = Runner(env, agent_cfg)
runner.agent.load(args.checkpoint)
runner.agent.set_running_mode("eval")

check_dir = args.out + "_check"
os.makedirs(args.out, exist_ok=True)
os.makedirs(check_dir, exist_ok=True)


def attacker_pixels():
    """← NEW: the attacker's real footprint in this picture: pixel count and tight box (or None)."""
    seg = raw_env._camera.data.output[SEG][0, ..., 0]           # (480, 640) object ID per pixel
    info = raw_env._camera.data.info[SEG]
    if isinstance(info, list):                                  # some versions keep one dict per camera
        info = info[0]
    ids = [int(k) for k, path in info["idToLabels"].items() if "/Attacker" in str(path)]
    if not ids:
        return 0, None
    mask = torch.isin(seg, torch.tensor(ids, device=seg.device, dtype=seg.dtype)).cpu().numpy()
    count = int(mask.sum())
    if count == 0:
        return 0, None
    ys, xs = np.nonzero(mask)
    return count, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


# the answer key for 6.1 Step 2 and 5.2 Step 3: one row per saved picture
csv_file = open(os.path.join(args.out, "readings.csv"), "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["name", "dist_m", "bx", "by", "asz", "visible",
                 "drone_px", "box_x", "box_y", "box_w", "box_h", "gradeable"])   # ← NEW columns

obs, _ = env.reset()
step, saved = 0, 0
while app.is_running() and saved < args.num_frames:
    with torch.no_grad():
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])  # best guess, no random variation
    obs, _, terminated, truncated, _ = env.step(actions)       # ← EDITED: keep the two "chase ended" flags
    step += 1
    if (terminated | truncated).any():                         # ← NEW: a chase ended and a new one began in this step.
        continue                                               #   The picture was taken before the restart,
                                                               #   the readings after it, so the two would not match.
    if step % args.every:                                      # not a saving step: keep flying
        continue

    rgb = raw_env._camera.data.output["rgb"][0].cpu().numpy()  # (480, 640, 3)
    if rgb.dtype != np.uint8:                                  # some versions return 0..1 floats
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    dist = raw_env._dist[0].item()                             # true distance, metres

    # where the attacker is in the frame RIGHT NOW (no delay, no noise), from 3.1 Part F
    bx_t, by_t, asz_t, vis_t = raw_env._camera_readings()
    bx, by, asz, visible = bx_t[0].item(), by_t[0].item(), asz_t[0].item(), vis_t[0].item()
    v = int(visible > 0.5)

    # ← NEW: what the picture really contains
    drone_px, box = attacker_pixels()
    if box:
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        box_x, box_y = (x0 + x1) / 2, (y0 + y1) / 2
    else:
        bw = bh = 0
        box_x = box_y = -1
    gradeable = int(v == 1 and max(bw, bh) >= MIN_LONG and min(bw, bh) >= MIN_THIN)

    name = f"frame_{saved:03d}_d{dist:05.2f}m_v{v}.png"
    img = Image.fromarray(rgb[..., :3])
    img.save(os.path.join(args.out, name))                     # clean copy: for the detector test
    writer.writerow([name, f"{dist:.3f}", f"{bx:.4f}", f"{by:.4f}", f"{asz:.4f}", v,
                     drone_px, f"{box_x:.1f}", f"{box_y:.1f}", bw, bh, gradeable])

    check = img.copy()                                         # marked copy: for you to look at
    draw = ImageDraw.Draw(check)
    if v:
        W, H = img.size
        xc, yc = W / 2 + bx * W / 2, H / 2 + by * H / 2        # bearing → pixel position
        half = max(asz * W / 2, 6)                             # at least 12 px wide, so you can see it
        draw.rectangle([xc - half, yc - half, xc + half, yc + half], outline=(255, 0, 0), width=2)
    if box:                                                    # ← NEW: the attacker's real pixels
        draw.rectangle([x0 - 2, y0 - 2, x1 + 1, y1 + 1], outline=(0, 200, 0), width=1)
    check.save(os.path.join(check_dir, name))

    print(f"saved {name}  bx={bx:+.2f} by={by:+.2f} size={asz:.3f}  "
          f"drone {drone_px} px, box {bw}x{bh}, gradeable {gradeable}", flush=True)
    saved += 1

csv_file.close()
print(f"done — {saved} frames and readings.csv in {args.out}", flush=True)
env.close()
app.close()