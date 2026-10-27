# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\capture_arena_frames.py
# ── NEW file — the whole content, nothing to insert into.
# ── Loads : your Ch.3 checkpoint, passed on the command line
# ── Writes: data\arena_frames\frame_XXX_dDD.DDm_vN.png        (clean, for the detector test)
# ──         data\arena_frames_check\frame_XXX_dDD.DDm_vN.png  (red box where the attacker should be)
# ──   v1 = attacker in the camera's view right now, v0 = out of view (or too small to detect)

import argparse, os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)   # your Ch.3 best_agent.pt
parser.add_argument("--num_frames", type=int, default=40)      # how many pictures to save
parser.add_argument("--every", type=int, default=5)            # 1 picture every 5 control steps = 0.25 s at 20 Hz
parser.add_argument("--out", type=str,
                    default=r"C:\projects\drone_pursuit\drone_pursuit\data\arena_frames")
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
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
env_cfg.scene.num_envs = 1                                     # one chase is enough
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

obs, _ = env.reset()
step, saved = 0, 0
while app.is_running() and saved < args.num_frames:
    with torch.no_grad():
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])  # best guess, no random variation
    obs, _, _, _, _ = env.step(actions)
    step += 1
    if step % args.every:                                      # not a saving step: keep flying
        continue

    rgb = raw_env._camera.data.output["rgb"][0].cpu().numpy()  # (480, 640, 3)
    if rgb.dtype != np.uint8:                                  # some versions return 0..1 floats
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    dist = raw_env._dist[0].item()                             # true distance, metres

    # where the attacker is in the frame RIGHT NOW (no delay), from 3.1 Part F
    bx_t, by_t, asz_t, vis_t = raw_env._camera_readings()
    bx, by, asz, visible = bx_t[0].item(), by_t[0].item(), asz_t[0].item(), vis_t[0].item()

    name = f"frame_{saved:03d}_d{dist:05.2f}m_v{int(visible > 0.5)}.png"
    img = Image.fromarray(rgb[..., :3])
    img.save(os.path.join(args.out, name))                     # clean copy: for the detector test

    check = img.copy()                                         # marked copy: for you to look at
    if visible > 0.5:
        W, H = img.size
        xc, yc = W / 2 + bx * W / 2, H / 2 + by * H / 2        # bearing → pixel position
        half = max(asz * W / 2, 6)                             # at least 12 px wide, so you can see it
        ImageDraw.Draw(check).rectangle([xc - half, yc - half, xc + half, yc + half],
                                        outline=(255, 0, 0), width=2)
    check.save(os.path.join(check_dir, name))

    print(f"saved {name}  bx={bx:+.2f} by={by:+.2f} size={asz:.3f}", flush=True)
    saved += 1

print(f"done — {saved} frames in {args.out}", flush=True)
env.close()
app.close()