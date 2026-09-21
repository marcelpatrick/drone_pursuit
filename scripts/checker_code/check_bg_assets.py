"""
check_bg_assets.py (v2) — tests every background AND drone asset URL used in 4.2B, and lists
what really exists in each NVIDIA folder so broken entries can be replaced with working ones.

Needs only plain Python 3 (no Isaac Sim). Run it in any env:
    python check_bg_assets.py
"""
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
ROOT = f"{BUCKET}/Assets/Isaac/5.1"          # = NUCLEUS_ASSET_ROOT_DIR on Isaac Sim 5.1
NVIDIA_NUCLEUS_DIR = f"{ROOT}/NVIDIA"
ISAAC_NUCLEUS_DIR = f"{ROOT}/Isaac"
NV_CONTENT = f"{BUCKET}/Assets"
PEGASUS = ("https://raw.githubusercontent.com/PegasusSimulator/PegasusSimulator/main/"
           "extensions/pegasus.simulator/pegasus/simulator/assets/Robots")

# ── the exact paths used in 4.2B Step 4 ─────────────────────────────────────
PATHS = {
    "skies": [
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/qwantani_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/mealie_road_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/noon_grass_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/champagne_castle_1_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
        f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/wooden_lounge_4k.hdr",
    ],
    "floor textures": [
        f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Ground/textures/aggregate_exposed_diff.jpg",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_diff.jpg",
        f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_brick_grey.jpg",
        f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_wood_boards_brown.jpg",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
    ],
    "wall textures": [
        f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_brick_grey.jpg",
        f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_wooden_wall.jpg",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/RustedMetal/RustedMetal_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Carbon/Steel_Carbon_BaseColor.png",
    ],
    "trees & props": [
        f"{NV_CONTENT}/Vegetation/Trees/Red_Maple.usd",
        f"{NV_CONTENT}/Vegetation/Trees/Japanese_Cherry.usd",
        f"{NV_CONTENT}/Vegetation/Shrub/Boxwood.usd",
        f"{NV_CONTENT}/Vegetation/Shrub/Cedar_Shrub.usd",
    ],
    # 4.2B Step 5 — models loaded straight from the Isaac Sim asset server
    "drone models (Isaac Sim)": [
        f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",
        f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/Quadcopter/quadcopter.usd",
        f"{ISAAC_NUCLEUS_DIR}/Robots/NTNU/ARL-Robot-1/arl_robot_1.usd",
        f"{ISAAC_NUCLEUS_DIR}/Robots/NASA/Ingenuity/ingenuity.usd",
    ],
    # 4.2B Step 2 — the two curl downloads into data\drones\
    "drone models (Pegasus downloads)": [
        f"{PEGASUS}/Iris/iris.usd",
        f"{PEGASUS}/Pegasus/pegasus_optimized.usdc",
    ],
    # pictures that iris.usd pulls from NVIDIA's server when it renders (without them it renders grey)
    "iris.usd material files": [
        f"{BUCKET}/Materials/Base/Carpet/Carpet_Gray.mdl",
        f"{BUCKET}/Materials/Base/Carpet/Carpet_Gray/Carpet_Gray_BaseColor.png",
        f"{BUCKET}/Materials/Base/Textiles/Linen_Blue.mdl",
        f"{BUCKET}/Materials/Base/Textiles/Linen_Blue/Linen_Blue_BaseColor.png",
    ],
}

# ── folders to inventory, and which files in them are useful ────────────────
FOLDERS = [
    ("skies",          f"{ROOT}/NVIDIA/Assets/Skies/",                           (".hdr", ".exr")),
    ("skies",          f"{ROOT}/Isaac/Materials/Textures/Skies/",                (".hdr", ".exr")),
    ("floor/wall tex", f"{ROOT}/Isaac/Materials/Textures/Patterns/",             (".jpg", ".png")),
    ("floor tex",      f"{ROOT}/NVIDIA/Materials/vMaterials_2/Ground/textures/", ("_diff.jpg", "_diff.png")),
    ("floor/wall tex", f"{ROOT}/NVIDIA/Materials/Base/",                         ("_BaseColor.png", "_BaseColor.jpg")),
    ("trees & props",  f"{BUCKET}/Assets/Vegetation/",                           (".usd",)),
    ("drone models",   f"{ROOT}/Isaac/Robots/Bitcraze/",                         (".usd", ".usda")),
    ("drone models",   f"{ROOT}/Isaac/Robots/IsaacSim/Quadcopter/",              (".usd", ".usda")),
    ("drone models",   f"{ROOT}/Isaac/Robots/NTNU/",                             (".usd", ".usda")),
    ("drone models",   f"{ROOT}/Isaac/Robots/NASA/",                             (".usd", ".usda")),
]

# searched across the whole Isaac/Robots folder: any file whose path mentions one of these words
DRONE_WORDS = ("drone", "quad", "copter", "uav", "aerial", "crazyflie", "iris", "tello", "dji", "arl_robot")


def exists(url):
    """HTTP HEAD: returns (True/False, short reason)."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:                     # DNS, proxy, timeout, no internet
        return False, f"{type(e).__name__}: {e}"


def list_keys(folder_url):
    """All object keys under a folder, via the S3 ListObjectsV2 API (follows pagination)."""
    prefix = folder_url[len(BUCKET) + 1:]
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        ns = {"s3": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        find = (lambda el, tag: el.findall(f"s3:{tag}", ns)) if ns else (lambda el, tag: el.findall(tag))
        keys += [c.find("s3:Key", ns).text if ns else c.find("Key").text for c in find(root, "Contents")]
        truncated = (root.find("s3:IsTruncated", ns) if ns else root.find("IsTruncated")).text == "true"
        if not truncated:
            return keys
        token = (root.find("s3:NextContinuationToken", ns) if ns else root.find("NextContinuationToken")).text


def main():
    print("=" * 78, "\nPART 1 — testing the paths used in 4.2B\n" + "=" * 78)
    broken = 0
    for label, urls in PATHS.items():
        print(f"\n[{label}]")
        for url in urls:
            ok, why = exists(url)
            broken += not ok
            print(f"  {'OK     ' if ok else 'MISSING'}  {why:<10} {url.replace(BUCKET, '')}")
    print(f"\n{broken} broken path(s)\n")

    print("=" * 78, "\nPART 2 — this is the list of all the available assets that exist in each folder (use these as replacements in case any of the path above are broken)\n" + "=" * 78)
    for label, folder, suffixes in FOLDERS:
        try:
            keys = [k for k in list_keys(folder) if k.lower().endswith(tuple(s.lower() for s in suffixes))
                    and "/.thumbs/" not in k]
        except Exception as e:
            print(f"\n[{label}] {folder.replace(BUCKET, '')}\n  could not list folder: {e}")
            continue
        print(f"\n[{label}] {folder.replace(BUCKET, '')}  — {len(keys)} file(s)")
        for k in keys:
            print(f"  {BUCKET}/{k}")

    print("\n" + "=" * 78, "\nPART 3 — any other drone-like robot file in Isaac/Robots (candidates for 4.2B Step 5)\n" + "=" * 78)
    try:
        keys = list_keys(f"{ROOT}/Isaac/Robots/")
        hits = [k for k in keys if k.lower().endswith((".usd", ".usda")) and "/.thumbs/" not in k
                and any(w in k.lower() for w in DRONE_WORDS)
                and "quadruped" not in k.lower()]            # "quad" also matches legged robots
        print(f"\n{len(hits)} candidate file(s) out of {len(keys)} files scanned")
        for k in hits:
            print(f"  {BUCKET}/{k}")
    except Exception as e:
        print(f"  could not list Isaac/Robots: {e}")


if __name__ == "__main__":
    sys.exit(main())
