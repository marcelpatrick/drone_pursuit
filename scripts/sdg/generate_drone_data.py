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
import glob                                  # ← NEW: finds your own files in data\bg\ and data\drones\
import math                                  # ← NEW: used to aim the camera in a random direction
import os
import random
from collections import Counter              # ← NEW: counts how often each drone look appears

import carb.settings
import omni.client                           # ← NEW: asks NVIDIA's server whether a file exists
import omni.kit.commands                     # ← NEW: used to create materials
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade   # ← EDITED: UsdShade and UsdLux added

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

# RT_SUBFRAMES = 8      # renders per photo, so the image settles after the camera move
RT_SUBFRAMES = 16     # renders per photo (4.2B: raised from 8 — new skies and textures need more renders)
WARMUP_STEPS = 3      # photos taken before the writer is attached, so assets finish loading

# ── SECTION 1 — the scene ───────────────────────────────────────────────────
ISAAC_NUCLEUS_DIR = f"{ASSETS}/Isaac"        # Isaac Sim's robots, props and pattern textures
NVIDIA_NUCLEUS_DIR = f"{ASSETS}/NVIDIA"      # NVIDIA's skies and material pictures


def make_box(path, size_x, size_y, size_z):
    """A box mesh whose corners sit at ±size/2 in metres, so a picture painted on it keeps its real scale."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    x, y, z = size_x / 2, size_y / 2, size_z / 2
    mesh.CreatePointsAttr([(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
                           (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 3, 2, 1,  4, 5, 6, 7,  0, 1, 5, 4,
                                      1, 2, 6, 5,  2, 3, 7, 6,  3, 0, 4, 7])
    mesh.CreateExtentAttr([(-x, -y, -z), (x, y, z)])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)     # sharp edges, not a rounded blob
    return mesh.GetPrim()


# the floor: a 200 m x 200 m slab, top surface at height 0, repainted every photo (Step 6)
floor = make_box("/World/Floor", 200.0, 200.0, 0.05)
UsdGeom.Xformable(floor).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.025))

# ambient fill — now also the sky: a 360° photograph, swapped every photo (Step 6)
dome = stage.DefinePrim("/World/Light", "DomeLight")
dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)
UsdLux.DomeLight(dome).CreateTextureFileAttr(
    Sdf.AssetPath(f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/qwantani_4k.hdr"))    # first sky
UsdLux.DomeLight(dome).CreateTextureFormatAttr(UsdLux.Tokens.latlong)             # a 360° photo
dome.CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(True)    # the camera sees the photo

# the sun — a single strong source coming from one direction
sun = stage.DefinePrim("/World/Sun", "DistantLight")
sun.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(3000.0)
sun.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(0.53)

# the camera aims here, and the drone look chosen for each photo is placed here (Step 5)
DRONE_POS = (0.0, 0.0, 1.5)


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


# BEFORE (4.1):
# def place_camera():
#     eye = Gf.Vec3d(rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.5, 3.0))
#     ...
# NEW!: AFTER:
def place_camera():
    """Put the camera 0.5–6 m from the drone, below or above it, turned to face it.
    The distance is drawn first, so near and far views appear about equally often."""
    while True:
        dist = rng.uniform(0.5, 6.0)                        # 1. how far: any distance, equally likely
        up = rng.uniform(-1.0, 1.0)                         # 2. which direction: up = vertical share,
        turn = rng.uniform(0.0, 2.0 * math.pi)              #    turn = compass angle
        flat = math.sqrt(1.0 - up * up)
        eye = target + Gf.Vec3d(dist * flat * math.cos(turn), dist * flat * math.sin(turn), dist * up)
        if 0.2 <= eye[2] <= 4.0:                            # 3. keep the camera 0.2–4 m above the floor
            break

    view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))   # z is "up"
    cam_pose_op.Set(view.GetInverse())



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

# ▼▼▼ NEW! ▼▼▼
# ── SECTION 2B — randomisers: change the scene before every photo (4.2) ──────
# Written in the style of NVIDIA's Isaac Sim 5.1 "Randomization Snippets":
# plain USD attribute edits, made just before each rep.orchestrator.step().
rand = random.Random(args.seed + 1)   # the dice for every scene change


def set_transform_attributes(prim, location=None, rotation=None, scale=None):
    """NVIDIA's helper (Randomization Snippets): add a transform op if missing, then set it.
    location in metres, rotation in degrees about x, then y, then z."""
    if location is not None:
        if not prim.HasAttribute("xformOp:translate"):
            UsdGeom.Xformable(prim).AddTranslateOp()
        prim.GetAttribute("xformOp:translate").Set(location)
    if rotation is not None:
        if not prim.HasAttribute("xformOp:rotateXYZ"):
            UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute("xformOp:rotateXYZ").Set(rotation)
    if scale is not None:
        if not prim.HasAttribute("xformOp:scale"):
            UsdGeom.Xformable(prim).AddScaleOp()
        prim.GetAttribute("xformOp:scale").Set(scale)


def create_omnipbr_material(mtl_path):
    """NVIDIA's helper (Randomization Snippets), shortened to the one input used here:
    an OmniPBR material whose base colour can be changed."""
    omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url="OmniPBR.mdl", mtl_name="OmniPBR", mtl_path=mtl_path)
    material_prim = stage.GetPrimAtPath(mtl_path)
    shader = UsdShade.Shader(omni.usd.get_shader_from_material(material_prim, get_prim=True))
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f)
    return UsdShade.Material(material_prim), shader


def random_color():
    return Gf.Vec3f(rand.random(), rand.random(), rand.random())


# a colour setting on both lights (their intensity settings already exist from Section 1)
dome.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
sun.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))

# distractors: 4 cubes + 4 spheres, 1 m across before scaling, each with its own paint,
# tagged "distractor" so 4.3 can drop their boxes
DISTRACTORS = []                       # (prim, shader)
for i in range(8):
    if i % 2 == 0:
        shape = UsdGeom.Cube.Define(stage, f"/World/Distractors/Cube_{i:02d}")
        shape.CreateSizeAttr(1.0)                                   # USD's default cube is 2 m
    else:
        shape = UsdGeom.Sphere.Define(stage, f"/World/Distractors/Sphere_{i:02d}")
        shape.CreateRadiusAttr(0.5)                                 # USD's default sphere is 2 m across
    shape.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])   # the size the renderer assumes
    prim = shape.GetPrim()
    add_labels(prim, labels=["distractor"], instance_name="class")
    material, shader = create_omnipbr_material(f"/World/Looks/Distractor_{i:02d}")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
    UsdGeom.Imageable(prim).MakeInvisible()                         # hidden until a randomiser shows it
    DISTRACTORS.append((prim, shader))


def randomise_lights():
    # ambient fill: brightness and a warm-to-cool tint
    dome.GetAttribute("inputs:intensity").Set(rand.uniform(500, 6000))
    dome.GetAttribute("inputs:color").Set(
        Gf.Vec3f(rand.uniform(0.7, 1.0), rand.uniform(0.7, 1.0), rand.uniform(0.6, 1.0)))
    # the sun: 10–80° elevation, any compass direction, golden to neutral, dim to blinding
    set_transform_attributes(sun, rotation=Gf.Vec3f(rand.uniform(-80, -10), 0.0, rand.uniform(0, 360)))
    sun.GetAttribute("inputs:intensity").Set(rand.uniform(1000, 12000))
    sun.GetAttribute("inputs:color").Set(Gf.Vec3f(1.0, rand.uniform(0.85, 1.0), rand.uniform(0.7, 1.0)))


def randomise_distractors():
    # show 3–8 of the 8 shapes; each shown one gets a new place, angle, size and colour
    shown = set(rand.sample(range(len(DISTRACTORS)), rand.randint(3, len(DISTRACTORS))))
    for i, (prim, shader) in enumerate(DISTRACTORS):
        if i not in shown:
            UsdGeom.Imageable(prim).MakeInvisible()
            continue
        s = rand.uniform(0.05, 0.4)                                 # 5–40 cm across
        set_transform_attributes(
            prim,
            location=Gf.Vec3d(rand.uniform(-5, 5), rand.uniform(-5, 5), rand.uniform(0.0, 3.5)),
            rotation=Gf.Vec3f(rand.uniform(0, 360), rand.uniform(0, 360), rand.uniform(0, 360)),
            scale=Gf.Vec3f(s, s, s),
        )
        shader.GetInput("diffuse_color_constant").Set(random_color())
        UsdGeom.Imageable(prim).MakeVisible()


def randomise_drone():
    # a banking drone looks different from a level one: ±20° tilt, any heading (4.2B replaces this)
    set_transform_attributes(
        drone, rotation=Gf.Vec3f(rand.uniform(-20, 20), rand.uniform(-20, 20), rand.uniform(0, 360)))


def randomise_scene():
    """Called once per photo, just before the camera moves."""
    randomise_lights()             # ← EXISTING (4.2)
    randomise_distractors()        # ← EXISTING (4.2)
    randomise_background()         # ← NEW: sky, floor, buildings, walls, trees
    randomise_drone_look()         # ← NEW: replaces randomise_drone()
# ▲▲▲ END OF INSERT ▲▲▲

# ▼▼▼ INSERT HERE! ▼▼▼
# ── SECTION 2C — background pools and objects (4.2B Step 4) ─────────────────
BG_DIR = r"C:\projects\drone_pursuit\drone_pursuit\data\bg"


def local_files(subfolder, extensions):
    """Every file in data\\bg\\<subfolder>\\ with one of the given extensions."""
    found = []
    for ext in extensions:
        found += glob.glob(os.path.join(BG_DIR, subfolder, f"*{ext}"))
    return [p.replace("\\", "/") for p in found]


def exists(path):
    """True if the file is on this computer, or answers on NVIDIA's asset server."""
    if os.path.isfile(path):
        return True
    result, _ = omni.client.stat(path)
    return result == omni.client.Result.OK


def keep_existing(label, paths, required=True):
    """Keep only paths that exist; print the ones that are missing."""
    usable = [p for p in paths if exists(p)]
    for p in paths:
        if p not in usable:
            print(f"[bg] {label}: NOT FOUND, skipped -> {p}")
    print(f"[bg] {label}: {len(usable)} usable")
    if required and not usable:
        raise RuntimeError(f"[bg] {label}: none usable - read the NOT FOUND lines above")
    return usable


SKIES = keep_existing("skies", [
    # outdoor
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/qwantani_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/mealie_road_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/noon_grass_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/champagne_castle_1_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    # indoor — for flights in a room or a gym
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/wooden_lounge_4k.hdr",
] + local_files("sky", [".hdr", ".exr"]))

FLOOR_TEX = keep_existing("floor textures", [
    f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Ground/textures/aggregate_exposed_diff.jpg",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_diff.jpg",
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_brick_grey.jpg",
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_wood_boards_brown.jpg",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
] + local_files("floor", [".jpg", ".jpeg", ".png"]))

WALL_TEX = keep_existing("wall textures", [
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_brick_grey.jpg",
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_wooden_wall.jpg",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/RustedMetal/RustedMetal_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Carbon/Steel_Carbon_BaseColor.png",
] + local_files("wall", [".jpg", ".jpeg", ".png"]))

# trees come from NVIDIA's general content library, which is separate from the Isaac Sim pack
NV_CONTENT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets"
PROP_USDS = keep_existing("trees & props", [
    f"{NV_CONTENT}/Vegetation/Trees/Red_Maple.usd",
    f"{NV_CONTENT}/Vegetation/Trees/Japanese_Cherry.usd",
    f"{NV_CONTENT}/Vegetation/Shrub/Boxwood.usd",
    f"{NV_CONTENT}/Vegetation/Shrub/Cedar_Shrub.usd",
] + local_files("props", [".usd", ".usda", ".usdc", ".usdz"]), required=False)

PROP_SIZE_M = (2.0, 12.0)   # every tree/prop is resized so its largest side is 2–12 m
PROP_TILT_X = 0.0           # set to 90.0 or -90.0 only if trees appear lying on their side


def strips(near, far):
    """Four ((x_min, y_min), (x_max, y_max)) rectangles, all at least `near` m from the drone."""
    return [
        ((-far,  near), ( far,  far)),     # north
        ((-far, -far),  ( far, -near)),    # south
        (( near, -near), ( far,  near)),   # east
        ((-far, -near), (-near,  near)),   # west
    ]


# near = 6 m camera limit + half the object's largest width (Step 1)
WALL_STRIPS     = strips(near=12.0, far=25.0)   # walls up to 12 m long  → half = 6 m
PROP_STRIPS     = strips(near=15.0, far=40.0)   # props up to 12 m wide  → half-diagonal ≈ 8.5 m
BUILDING_STRIPS = strips(near=22.0, far=70.0)   # up to 20 m x 20 m      → half-diagonal ≈ 14 m


def show(prim, visible):
    """Show or hide a prim and everything under it."""
    img = UsdGeom.Imageable(prim)
    img.MakeVisible() if visible else img.MakeInvisible()


def create_textured_material(mtl_path, prim):
    """An OmniPBR material that wears a picture, bound to `prim`. The texture inputs follow
    NVIDIA's full create_omnipbr_material helper (Randomization Snippets)."""
    material, shader = create_omnipbr_material(mtl_path)
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)   # paint by position, no UV map needed
    shader.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
    return shader


def set_texture(shader, file, rotate_deg=0.0):
    shader.GetInput("diffuse_texture").Set(Sdf.AssetPath(file))
    shader.GetInput("texture_rotate").Set(float(rotate_deg))


def measure(prim):
    """Largest side of a prim's visible geometry, in metres (0 if nothing loaded)."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    size = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
    return max(size[0], size[1], size[2]), size


def load_model(holder_path, usd):
    """Load a 3D model one level below a holder of our own, and return the holder.
    The script moves, turns and resizes the holder, never the model's own top level."""
    holder = stage.DefinePrim(holder_path, "Xform")
    add_reference_to_stage(usd, f"{holder_path}/model")
    return holder


build_rng = random.Random(7)       # fixed seed = the same building and wall sizes on every run

FLOOR_SHADER = create_textured_material("/World/Looks/Floor", floor)

BUILDINGS = []                     # (prim, height, shader)
for i in range(12):
    w, d, h = build_rng.uniform(6, 20), build_rng.uniform(6, 20), build_rng.uniform(4, 30)
    box = make_box(f"/World/Background/Building_{i:02d}", w, d, h)
    BUILDINGS.append((box, h, create_textured_material(f"/World/Looks/Building_{i:02d}", box)))
    show(box, False)

WALLS = []                         # (prim, height, shader)
for i in range(8):
    length, h = build_rng.uniform(6, 12), build_rng.uniform(2.5, 6)
    box = make_box(f"/World/Background/Wall_{i:02d}", length, 0.3, h)
    WALLS.append((box, h, create_textured_material(f"/World/Looks/Wall_{i:02d}", box)))
    show(box, False)

PROPS = []                         # (holder prim, scale that makes the model's largest side 1 m)
for i in range(8 if PROP_USDS else 0):
    usd = PROP_USDS[i % len(PROP_USDS)]
    path = f"/World/Background/Prop_{i:02d}"
    try:
        holder = load_model(path, usd)
        largest, size = measure(holder)
    except Exception as e:
        print(f"[bg] {path}: FAILED to load {usd} -> {e}; skipped")
        continue
    if not largest > 0:            # file opened but no geometry arrived: nothing to scale from
        print(f"[bg] {path}: {usd} loaded with no geometry; hidden and skipped")
        show(holder, False)
        continue
    print(f"[bg] {path}: {usd.rsplit('/', 1)[-1]} authored size {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}")
    show(holder, False)            # shown per photo by randomise_background()
    PROPS.append((holder, 1.0 / largest))
# ▲▲▲ END OF INSERT ▲▲▲

# ▼▼▼ INSERT HERE! ▼▼▼
# ── SECTION 2D — the attacker drone: several models × two looks (4.2B Step 5) ─
DRONE_DIR = r"C:\projects\drone_pursuit\drone_pursuit\data\drones"
DRONE_SIZE_M = (0.08, 0.35)        # largest side after resizing, in metres
RECOLOR_SHARE = 0.5                # share of photos that show a recoloured look

drone_files = []
for ext in (".usd", ".usda", ".usdc", ".usdz"):
    drone_files += [p.replace("\\", "/") for p in glob.glob(os.path.join(DRONE_DIR, f"*{ext}"))]

DRONE_SOURCES = [                  # (file, weight) — a higher weight is chosen more often
    (f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd", 3.0),        # ~9 cm, closest to a Tello
    (f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/Quadcopter/quadcopter.usd", 1.0),
    (f"{ISAAC_NUCLEUS_DIR}/Robots/NTNU/ARL-Robot-1/arl_robot_1.usd", 1.0),     # newer docs list it; may be absent in 5.1
    (f"{ISAAC_NUCLEUS_DIR}/Robots/NASA/Ingenuity/ingenuity.usd", 0.3),        # coaxial helicopter: rare on purpose
] + [(p, 3.0) for p in drone_files]                                            # your own models (Step 2)

usable = keep_existing("drone models", [p for p, _ in DRONE_SOURCES])
DRONE_SOURCES = [(p, w) for p, w in DRONE_SOURCES if p in usable]


def deinstance(root):
    """Switch instancing off under `root` so every part can take its own material."""
    while True:
        instances = [p for p in Usd.PrimRange(root) if p.IsInstance()]
        if not instances:
            return
        for p in instances:        # repeat: switching one off can reveal instances nested inside it
            p.SetInstanceable(False)


DRONE_LOOKS = []                   # dicts: holder prim, path, unit scale, weight, part shaders (empty = original paint)
for k, (usd, weight) in enumerate(DRONE_SOURCES):
    for look in ("original", "recolor"):
        path = f"/World/Drones/Drone_{k:02d}_{look}"
        try:
            holder = load_model(path, usd)
            add_labels(holder, labels=["drone"], instance_name="class")   # every look is class "drone"
            if look == "recolor":
                deinstance(holder)
            largest, size = measure(holder)
        except Exception as e:
            print(f"[drone] {path}: FAILED to load {usd} -> {e}; skipped")
            continue
        if not largest > 0:
            print(f"[drone] {path}: {usd} loaded with no geometry; hidden and skipped")
            show(holder, False)
            continue
        shaders = []
        if look == "recolor":      # one material per part, so each part can get its own colour
            meshes = [p for p in Usd.PrimRange(holder) if p.IsA(UsdGeom.Mesh)]
            for j, mesh in enumerate(meshes):
                material, shader = create_omnipbr_material(f"/World/Looks/Drone_{k:02d}_part_{j:03d}")
                UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material, UsdShade.Tokens.strongerThanDescendants)
                shaders.append(shader)
        show(holder, False)        # every look starts hidden; one is shown per photo
        share = RECOLOR_SHARE if look == "recolor" else 1.0 - RECOLOR_SHARE
        print(f"[drone] {path}: {usd.rsplit('/', 1)[-1]} size {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}"
              + (f", {len(shaders)} editable meshes" if look == "recolor" else ""))
        DRONE_LOOKS.append({"prim": holder, "path": path, "unit": 1.0 / largest,
                            "weight": weight * share, "shaders": shaders})

if not DRONE_LOOKS:
    raise RuntimeError("[drone] no drone model loaded - read the [drone] and [bg] lines above")

SHOWN = Counter()                  # how many saved photos each look appeared in, printed at the end
# ▲▲▲ END OF INSERT ▲▲▲

# ▼▼▼ INSERT HERE! ▼▼▼
def randomise_background():
    # sky: a different photo, turned to face a different way
    UsdLux.DomeLight(dome).GetTextureFileAttr().Set(Sdf.AssetPath(rand.choice(SKIES)))
    set_transform_attributes(dome, rotation=Gf.Vec3f(0.0, 0.0, rand.uniform(0, 360)))

    # floor: new picture at a new angle; hidden 1 photo in 5, so the sky photo's own ground shows
    set_texture(FLOOR_SHADER, rand.choice(FLOOR_TEX), rand.uniform(0, 360))
    show(floor, rand.random() < 0.8)

    # buildings and walls: stand on the floor (centre at half height) in their strip, any heading
    for group, strip_set in ((BUILDINGS, BUILDING_STRIPS), (WALLS, WALL_STRIPS)):
        for i, (box, h, shader) in enumerate(group):
            visible = rand.random() < 0.6
            show(box, visible)
            if not visible:
                continue
            (x0, y0), (x1, y1) = strip_set[i % 4]
            set_transform_attributes(box,
                                     location=Gf.Vec3d(rand.uniform(x0, x1), rand.uniform(y0, y1), h / 2),
                                     rotation=Gf.Vec3f(0.0, 0.0, rand.uniform(0, 360)))
            set_texture(shader, rand.choice(WALL_TEX), rand.choice([0.0, 90.0]))

    # trees & props: base on the floor, any heading, resized to 2–12 m
    for i, (holder, unit) in enumerate(PROPS):
        visible = rand.random() < 0.7
        show(holder, visible)
        if not visible:
            continue
        (x0, y0), (x1, y1) = PROP_STRIPS[i % 4]
        s = unit * rand.uniform(*PROP_SIZE_M)
        set_transform_attributes(holder,
                                 location=Gf.Vec3d(rand.uniform(x0, x1), rand.uniform(y0, y1), 0.0),
                                 rotation=Gf.Vec3f(PROP_TILT_X, 0.0, rand.uniform(0, 360)),
                                 scale=Gf.Vec3f(s, s, s))


_last_look = [None]


def randomise_drone_look():
    """Exactly one drone per photo: hide the previous look, show a new one at DRONE_POS."""
    if _last_look[0] is not None:
        show(_last_look[0]["prim"], False)
    d = rand.choices(DRONE_LOOKS, weights=[x["weight"] for x in DRONE_LOOKS])[0]
    s = d["unit"] * rand.uniform(*DRONE_SIZE_M)
    set_transform_attributes(d["prim"],
                             location=Gf.Vec3d(*DRONE_POS),
                             rotation=Gf.Vec3f(rand.uniform(-20, 20), rand.uniform(-20, 20), rand.uniform(0, 360)),
                             scale=Gf.Vec3f(s, s, s))
    for shader in d["shaders"]:    # recolour looks: every part its own colour
        shader.GetInput("diffuse_color_constant").Set(random_color())
    show(d["prim"], True)
    SHOWN[d["path"]] += 1
    _last_look[0] = d


def preload_assets():
    """Show every sky and texture once, so downloads finish before the first photo, not during it."""
    print(f"preloading {len(SKIES)} skies and {len(set(FLOOR_TEX + WALL_TEX))} textures...", flush=True)
    show(floor, True)
    for sky in SKIES:
        UsdLux.DomeLight(dome).GetTextureFileAttr().Set(Sdf.AssetPath(sky))
        for _ in range(5):
            simulation_app.update()
    for tex in sorted(set(FLOOR_TEX + WALL_TEX)):
        set_texture(FLOOR_SHADER, tex)
        for _ in range(3):
            simulation_app.update()
# ▲▲▲ END OF INSERT ▲▲▲

# ── SECTION 3 — the capture loop ────────────────────────────────────────────
def drone_and_distractor_boxes():
    """Split the boxes in the photo just captured into drone boxes and the rest."""
    out = bbox_annot.get_data()
    drone_ids = {int(k) for k, v in out["info"]["idToLabels"].items() if "drone" in str(v)}
    drone_boxes = [b for b in out["data"] if int(b["semanticId"]) in drone_ids]
    return drone_boxes, len(out["data"]) - len(drone_boxes)


preload_assets()                       # ← NEW: download every sky and texture before the first photo

print("warming up renderer...", flush=True)
for _ in range(WARMUP_STEPS):          # writer not attached yet: these photos are not saved
    randomise_scene()
    place_camera()
    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
SHOWN.clear()                          # ← NEW: count only the saved photos

writer.attach(render_product)
print("capturing", flush=True)

for i in range(args.num_frames):
    randomise_scene()                                       # 1. lights, distractors, background, drone look
    place_camera()                                          # 2. new camera spot
    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)        # 3. render; the writer saves 4 files
    drone_boxes, n_other = drone_and_distractor_boxes()     # 4. print a check
    sizes = " ".join(f"{int(b['x_max'] - b['x_min'])}x{int(b['y_max'] - b['y_min'])}px" for b in drone_boxes)
    look = _last_look[0]["path"].rsplit("/", 1)[-1]
    print(f"frame {i + 1}/{args.num_frames}  drone={len(drone_boxes)} {sizes}  other={n_other}  {look}", flush=True)

# release resources, then wait until every file is on disk
writer.detach()
bbox_annot.detach()
render_product.destroy()
rep.orchestrator.wait_until_complete()

for path, n in sorted(SHOWN.items()):   # ← NEW: how often each drone look appeared
    print(f"[drone] {path}: shown in {n} of {args.num_frames} photos", flush=True)
print("done", flush=True)
simulation_app.close()