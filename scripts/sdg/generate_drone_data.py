# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\generate_drone_data.py
# ── 4.1 version 3: no Replicator writer, trigger or orchestrator. The Python
# ──   loop moves the camera, renders with simulation_app.update(), and saves
# ──   the files itself — the same way Isaac Lab's own camera sensors work.

"""Standalone SDG: labeled images of a Crazyflie for detector training."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_frames", type=int, default=200)
parser.add_argument("--out_dir", type=str, default=r"C:\projects\drone_pursuit\drone_pursuit\data\raw")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True                       # cameras need the render pipeline
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything below runs inside the sim app ----
import json
import os
import random

import numpy as np
from PIL import Image

import omni.replicator.core as rep
import omni.usd
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))
stage = omni.usd.get_context().get_stage()

WARMUP_UPDATES = 30   # frames rendered once at start, so assets finish loading
FRAME_UPDATES = 8     # frames rendered per photo, so the image settles after a camera move

# ── SECTION 1 — the scene ───────────────────────────────────────────────────
# ground + two lights (both get randomized in 4.2)
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

# ambient fill — stands in for skylight bouncing around
dome_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
dome_cfg.func("/World/Light", dome_cfg)

# the sun — a single strong source coming from one direction
sun_cfg = sim_utils.DistantLightCfg(intensity=3000.0, angle=0.53)
sun_cfg.func("/World/Sun", sun_cfg)

# the subject of every photo: a Crazyflie, tagged with its class
DRONE_POS = (0.0, 0.0, 1.5)
drone_cfg = sim_utils.UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",
    semantic_tags=[("class", "drone")],          # ← this line is what produces the labels
)
drone_cfg.func("/World/Drone", drone_cfg, translation=DRONE_POS)

# ── SECTION 2 — camera and the two "sensors" that read each image ───────────
# a plain USD camera: 12 mm lens on a 20.955 mm sensor ≈ 82° wide, like a Tello
CAM_PATH = "/World/Camera"
cam = UsdGeom.Camera.Define(stage, CAM_PATH)
cam.GetFocalLengthAttr().Set(12.0)
cam.GetHorizontalApertureAttr().Set(20.955)
cam.GetVerticalApertureAttr().Set(20.955 * 480 / 640)     # 4:3, like the real drone
cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
cam_pose_op = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()

# render product = the 640x480 image the camera exposes onto
render_product = rep.create.render_product(CAM_PATH, (640, 480))

# annotators = readers attached to that image: one for colour, one for boxes
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
bbox_annot = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
rgb_annot.attach(render_product)
bbox_annot.attach(render_product)

# ── SECTION 3 — the capture loop ────────────────────────────────────────────
os.makedirs(args.out_dir, exist_ok=True)
rng = random.Random(args.seed)
target = Gf.Vec3d(*DRONE_POS)


def place_camera():
    """Put the camera at a random spot and turn it to face the drone."""
    eye = Gf.Vec3d(rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.5, 3.0))
    view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))   # z is "up"
    cam_pose_op.Set(view.GetInverse())


def render(n):
    """Render n frames. Plain app updates: nothing here can wait forever."""
    for _ in range(n):
        simulation_app.update()


print("warming up renderer...", flush=True)
place_camera()
render(WARMUP_UPDATES)
print("capturing", flush=True)

for i in range(args.num_frames):
    place_camera()
    render(FRAME_UPDATES)

    rgb = rgb_annot.get_data()
    bbox = bbox_annot.get_data()
    if rgb is None or rgb.size == 0:
        print(f"frame {i + 1}/{args.num_frames}  skipped (image not ready)", flush=True)
        continue

    # same file names BasicWriter uses, so 4.2 and 4.3 read them unchanged
    Image.fromarray(rgb[:, :, :3]).save(os.path.join(args.out_dir, f"rgb_{i:04d}.png"))
    np.save(os.path.join(args.out_dir, f"bounding_box_2d_tight_{i:04d}.npy"), bbox["data"])
    labels = {str(k): v for k, v in bbox["info"]["idToLabels"].items()}
    with open(os.path.join(args.out_dir, f"bounding_box_2d_tight_labels_{i:04d}.json"), "w") as f:
        json.dump(labels, f)
    with open(os.path.join(args.out_dir, f"bounding_box_2d_tight_prim_paths_{i:04d}.json"), "w") as f:
        json.dump(list(bbox["info"]["primPaths"]), f)

    print(f"frame {i + 1}/{args.num_frames}  boxes={len(bbox['data'])}", flush=True)

simulation_app.close()