# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\generate_drone_data.py
# ── 4.1, SimulationApp version. Starts Isaac Sim the way NVIDIA's Isaac Sim 5.1
# ──   Replicator "Getting Started Scripts" do: capture-on-play off, a USD-API
# ──   change before each photo (here: the camera position), rep.orchestrator.step()
# ──   to capture, BasicWriter to save, rep.orchestrator.wait_until_complete() at the end.
# ──   No randomisation of lights or objects yet: that is 4.2.

"""Standalone SDG: labelled images of a Crazyflie for detector training."""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_frames", type=int, default=200)
parser.add_argument("--out_dir", type=str, default=r"C:\projects\drone_pursuit\drone_pursuit\data\raw")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

# ── SECTION 0 — start Isaac Sim ─────────────────────────────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": args.headless})

# ---- everything below runs inside the sim app ----
import os
import random

import carb.settings
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)        # 1 unit = 1 metre, as in Isaac Lab
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)  # z is "up", as in Isaac Lab
rep.orchestrator.set_capture_on_play(False)      # save only when step() is called
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)   # DLSS Quality: NVIDIA's advice below 600x600
rng = random.Random(args.seed)

ASSETS = get_assets_root_path()
if ASSETS is None:
    raise RuntimeError("Isaac Sim asset root not found: check your internet connection")

RT_SUBFRAMES = 8      # renders per photo, so the image settles after the camera move
WARMUP_STEPS = 3      # photos taken before the writer is attached, so assets finish loading

# ── SECTION 1 — the scene ───────────────────────────────────────────────────
# the grey grid floor (the same file Isaac Lab's GroundPlaneCfg loads)
add_reference_to_stage(f"{ASSETS}/Isaac/Environments/Grid/default_environment.usd", "/World/ground")

# ambient fill — stands in for skylight bouncing around
dome = stage.DefinePrim("/World/Light", "DomeLight")
dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(2000.0)

# the sun — a single strong source coming from one direction
sun = stage.DefinePrim("/World/Sun", "DistantLight")
sun.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(3000.0)
sun.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(0.53)

# the subject of every photo: a Crazyflie, 1.5 m above the floor, tagged with its class
DRONE_POS = (0.0, 0.0, 1.5)
drone = stage.DefinePrim("/World/Drone", "Xform")             # holder we position
UsdGeom.Xformable(drone).AddTranslateOp().Set(Gf.Vec3d(*DRONE_POS))
add_reference_to_stage(f"{ASSETS}/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd", "/World/Drone/cf2x")
add_labels(drone, labels=["drone"], instance_name="class")      # ← this is what produces the labels

# ── SECTION 2 — camera, render product, writer and one annotator ────────────
# a plain USD camera: 12 mm lens on a 20.955 mm sensor ≈ 82° wide, like a Tello
CAM_PATH = "/World/Camera"
cam = UsdGeom.Camera.Define(stage, CAM_PATH)
cam.GetFocalLengthAttr().Set(12.0)
cam.GetHorizontalApertureAttr().Set(20.955)
cam.GetVerticalApertureAttr().Set(20.955 * 480 / 640)     # 4:3, like the real drone
cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
cam_pose_op = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
target = Gf.Vec3d(*DRONE_POS)


def place_camera():
    """Put the camera at a random spot and turn it to face the drone."""
    eye = Gf.Vec3d(rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.5, 3.0))
    view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))   # z is "up"
    cam_pose_op.Set(view.GetInverse())


render_product = rep.create.render_product(CAM_PATH, (640, 480))

# BasicWriter saves rgb_XXXX.png + the three bounding_box_2d_tight_* files
os.makedirs(args.out_dir, exist_ok=True)
writer = rep.writers.get("BasicWriter")
writer.initialize(output_dir=args.out_dir, rgb=True, bounding_box_2d_tight=True)

# a separate box annotator, only to print a per-photo check to the console
bbox_annot = rep.annotators.get("bounding_box_2d_tight")
bbox_annot.attach(render_product)

# ── SECTION 3 — the capture loop ────────────────────────────────────────────
print("warming up renderer...", flush=True)
for _ in range(WARMUP_STEPS):          # writer not attached yet: these photos are not saved
    place_camera()
    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)

# check 1: the drone loaded at its real size (about 0.1 m across)
size = (UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(drone).ComputeAlignedRange().GetSize())
print(f"drone size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m", flush=True)

writer.attach(render_product)
print("capturing", flush=True)

for i in range(args.num_frames):
    place_camera()                                          # 1. new camera spot
    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)        # 2. render; the writer saves 4 files
    boxes = bbox_annot.get_data()["data"]
    sizes = [f"{int(b['x_max'] - b['x_min'])}x{int(b['y_max'] - b['y_min'])}px" for b in boxes]
    print(f"frame {i + 1}/{args.num_frames}  boxes={len(boxes)}  {' '.join(sizes)}", flush=True)

# release resources, then wait until every file is on disk
writer.detach()
bbox_annot.detach()
render_product.destroy()
rep.orchestrator.wait_until_complete()
print("done", flush=True)
simulation_app.close()