# 🛩️ Drone Pursuit: A Sim-to-Real Follow-Along Tutorial
### Teaching one drone to see, find, and chase another — in Isaac Sim + Isaac Lab, on Windows

**The goal: a sim-to-real test.** You train a policy entirely in NVIDIA Isaac Sim and Isaac Lab, then fly it on a real DJI Tello that chases a real drone using a real camera. The simulation is the means; the flight is the point.

**Who this is for:** anyone who has completed the standard Isaac Lab introductory material — a first robot tutorial, a cartpole or Ant RL task, a Replicator synthetic-data pipeline, and an external project created with the template wizard. **No drone, hardware or telemetry experience is assumed.** Chapter 1.4 starts at plugging the drone in.

**Naming convention:** this is a *counter-drone defence* scenario. The **defender** is the drone we train — it carries the camera and intercepts. The **attacker** is the intruder, the target being chased.

**What you will have at the end:** a real Tello that takes off, finds another drone through its own camera, and closes on it — driven by a neural network that has never seen a real aircraft. Plus the two files that make it work: an object detector built entirely from synthetic images you generated, and a control policy trained against a simulated target.

**Hardware: about C$250.** A used Tello (~C$115), a cheap toy quadcopter as the target (C$50–70), spare propellers, batteries, and bright tape. Your laptop runs both models; nothing runs on the drone.

**Total time:** ~28 hours across 19 subchapters, each ≤1.5 hours of hands-on work. Training runs happen in the background. Every subchapter ends with a ✅ **Checkpoint** — a concrete test that tells you it worked before you move on.

**One training run, not two.** Chapter 1.4 measures the Tello *before* Chapter 3 designs the policy, so the policy is built around your actual drone from the start. This is the main reason the chapters are ordered as they are.

---
# Chapter 0 — The Big Picture (read this first, ~30 min)
<details>
     
<summary> expand </summary>

## 0.1 What are we actually building?

Three problems, stacked like layers:

```
┌──────────────────────────────────────────────────────────────┐
│  PROBLEM 3: INTEGRATION                                      │
│  "Chase what you SEE, not what the simulator TELLS you"      │
│  (bounding box → estimated position → policy input)          │
├──────────────────────────────────────────────────────────────┤
│  PROBLEM 2: PERCEPTION                                       │
│  "Find the other drone in a camera image"                    │
│  (synthetic data → object detection model)                   │
├──────────────────────────────────────────────────────────────┤
│  PROBLEM 1: CONTROL                                          │
│  "Fly toward a moving target without crashing"               │
│  (RL policy trained with PPO, as in the standard Isaac Lab tasks)        │
└──────────────────────────────────────────────────────────────┘
```

## 0.2 The one architectural decision that makes this project feasible

Here is the most important idea in this whole tutorial, so let's put it up front.

The "obvious" way to build this is to feed camera pixels directly into the RL policy — the drone learns to fly *from images*. **We are NOT doing that.** Training RL from pixels means rendering a camera for every one of your thousands of parallel environments on every step. It's 10–100× slower to train, much harder to debug (is the bug in flying or in seeing?), and needs a CNN policy you'd have to tune.

Instead, we use a pattern the robotics world calls **privileged training** (also called "teacher-student" or "state-based training with vision-based deployment"). The refinement that makes it work cleanly is this: **the policy's senses are defined in the camera's own language, but during training we produce those numbers with arithmetic instead of rendering.**

```
TRAINING TIME (Chapters 2–3)              DEMO TIME (Chapter 6)
─────────────────────────────             ─────────────────────────────
Simulator knows both drones'              Camera renders a real frame
exact positions (privileged)                        │
     │                                    YOLO detector returns a
     │  we PROJECT that truth                bounding box
     │  through the camera equations              │
     ▼                                            ▼
┌───────────────────────────┐         ┌───────────────────────────┐
│ 7 camera-native numbers:  │         │ THE SAME 7 NUMBERS,       │
│ where the attacker sits   │ ◄─ identical ─► measured from the   │
│ in frame, how big it      │  definition │ actual bounding box   │
│ looks, and how fast both  │         │                           │
│ of those are changing     │         │                           │
└───────────────────────────┘         └───────────────────────────┘
     │                                            │
     ▼                                            ▼
┌──────────┐                                ┌──────────┐
│ RL POLICY │  ◄────── same policy ────────► │ RL POLICY │ (unchanged!)
└──────────┘                                └──────────┘
     │                                            │
  thrust + body moments                     thrust + body moments
```

**Why this works:** the part of the policy's input that concerns the attacker is seven numbers describing what the camera reports (the rest is the defender's own motion and its last command). During training the simulator knows both positions, so we calculate what the camera would have reported — no frames are rendered, and training runs as fast as the plain hover task. At demo time the camera renders a frame and the detector supplies the same seven numbers. They are defined identically in both cases, so the policy responds the same way. This also lets you build and test the flight half and the vision half separately, following the standard synthetic-data pattern: **generate data in sim, train a model outside sim, bring the model back.**


## 0.3 The technology stack — what solves what, and why

| Problem | Technology | Why this choice (vs. alternatives) |
|---|---|---|
| **Flying a drone (in simulation)** | `CRAZYFLIE_CFG` asset from Isaac Lab's built-in `Isaac-Quadcopter-Direct-v0` task, with its force actions replaced by stick commands in 3.1 | Isaac Lab ships a working quadcopter hover task. We *modify* a proven task instead of building flight from scratch. **The Crazyflie is a stand-in, not the drone you will fly** — see §0.5. |
| **Flying a drone (in reality)** | A consumer WiFi camera drone such as a DJI Tello, commanded from your laptop over its Python SDK | Isaac Lab has no Tello model and does not need one. Chapter 3.1 replaces the simulated force model with the normalised stick commands a Tello accepts, after which the simulated airframe is a generic hovering body rather than a Crazyflie. |
| **The moving attacker** | A second Crazyflie, **kinematically scripted** (we set its position each step along a waypoint path) | An RL-vs-RL adversarial setup is a research project. A scripted intruder gives us a *predictable, tunable* difficulty level — exactly what the Multi-UAV pursuit-evasion literature does in its early curriculum stages. |
| **Closing the distance** | **PPO** via **skrl**, using Isaac Lab's standard `train.py`, with a **dense distance + closing-speed reward** | PPO/skrl is what the template wizard wires up for you. Dense rewards ("getting closer = points every step") train fast; sparse rewards ("points only on capture") often never take off. |
| **Synthetic data** | Isaac Sim **Replicator** (`omni.replicator.core`) with **semantic tags** + `bounding_box_2d_tight` annotator + **domain randomization** | The standard Replicator synthetic-data workflow — writer, annotator, randomisation — applied to a drone. |
| **Object detection** | **YOLOv8-nano** (Ultralytics), trained in a **separate conda env** | vs. TAO Toolkit (which you know): TAO needs Docker/WSL and heavier setup on Windows. Ultralytics is `pip install ultralytics` + one training command, runs natively on Windows, and YOLOv8n is small enough to run in real time. |
| **Getting the detector INTO Isaac Lab** | Export YOLO → **ONNX**, run with **onnxruntime** inside the Isaac Lab env | See §0.7. An ONNX file contains the trained weights and network structure with no Python dependencies, so it runs under onnxruntime alone. Installing Ultralytics into the Isaac Lab env would pull its own torch and could replace the one Isaac Lab needs. |
| **Turning a bounding box into policy input** | **Bearing + angular size** — the box's offset from the image centre and its share of the frame, plus how fast both are changing (§0.4) | Requires no knowledge of the attacker's real dimensions, so it works against any drone model. Also collapses Chapter 6's bridge from camera geometry to a handful of divisions. |
| **Onboard camera** | Isaac Lab **`TiledCameraCfg`** attached to the defender's body, run with `--enable_cameras` | Isaac Lab's vectorised camera API. Only one environment needs it at demo time, so rendering cost stays low. |

### Name etymology corner (quick hits you'll meet later)
- **YOLO** = *You Only Look Once*. Older detectors scanned an image region-by-region (many "looks"). YOLO's insight: one single neural network pass over the whole image predicts all boxes at once. The name is literally the algorithm.
- **ONNX** = *Open Neural Network eXchange*. A neutral file format for trained networks — like exporting a USD file: any tool that speaks the format can open it, regardless of what tool created it.
- **PPO** = *Proximal Policy Optimization*. "Proximal" (Latin *proximus*, "nearest") because each update is clipped to stay *near* the previous policy — small careful steps instead of wild jumps.
- **Replicator** — it *replicates* reality: generates many synthetic variations of a scene to stand in for real-world photos.

## 0.4 How the defender senses the attacker (the key design choice)

This section explains **what** the sensing scheme is, **what it's used for**, and **why** it was chosen over the alternatives. Everything in Chapters 3 and 6 follows from it.

### What the camera can and cannot tell you

The camera produces one rectangle per frame around the attacker. From it you can read where the rectangle sits in the image and how large it is — and nothing else. Distance in metres is *not* in there. A small drone nearby and a large drone far away produce an identical rectangle; a single camera at a single instant cannot separate them. Any scheme that claims to output metres has quietly assumed something extra.

### What we feed the policy

Seven numbers, all read straight off the rectangle, and nothing else about the attacker:

| # | Reading | What it means | What the defender does with it |
|---|---|---|---|
| 1 | Horizontal offset | How far left or right of centre the attacker appears | Steer to bring it toward the centre |
| 2 | Vertical offset | How far above or below centre it appears | Climb or descend to level with it |
| 3 | Share of view | How much of the frame width the attacker occupies | Small means far, large means near |
| 4–6 | Rate of change of each of the above | Whether it is drifting toward centre, and whether it is growing or shrinking | Tells the defender whether its current turn and throttle are working |
| 7 | Spotted flag | Whether the detector found anything at all this frame | Switch to a "lost sight of it" behaviour |

Number 7 matters more than it looks. When the detector comes up empty, the flag drops to zero and readings 1–3 are held at their previous values. Without the flag, the policy would receive stale numbers with no indication they were stale, and would have no way to learn that losing sight calls for different behaviour.

Numbers 4–6 are what separate chasing from intercepting. If reading 3 holds steady while reading 1 sweeps quickly, the defender is flying alongside the attacker at matched speed and never closing — a situation it must recognise and break out of.

The defender's own state — its velocity and which way gravity pulls — continues to come from the simulator, and that is legitimate: a real drone reads this from its onboard sensors. Only knowledge *about the attacker* has to come through the camera. (A cheap drone reports fewer of these than a simulator can provide; Chapter 7.4 removes whatever yours cannot supply.)

**Two more numbers join these seven before the policy sees them**, and both exist for hardware reasons explained in Chapter 3.1: the seven readings arrive *delayed*, because a real camera frame reaches your laptop a fraction of a second after it was taken, and the policy is also told what it commanded on the previous step, so it can account for decisions already in flight.

### Why this was chosen

**Because recovering metres from a picture requires the attacker's true width.** The conversion divides by that width, so facing a drone of a different size scales every distance estimate wrong by the same factor, with no error message. Readings 1–3 make no claim about metres, so there is nothing to get wrong.

**Because training and deployment use the same numbers.** The common failure in projects like this is a policy that reaches an 80% capture rate in Chapter 3 and then fails in Chapter 6, because it was trained on positions in metres and deployed on estimates derived from a rectangle. Here both phases compute the identical seven quantities, so the only difference is the detector's few-pixel error.

**Because it makes Chapter 6 simpler, not harder.** The conversion from bounding box to policy input is a few divisions. There is no camera geometry to get wrong, no focal length to derive, no unit test to write against ground truth.

**Because direction is what a camera can actually measure.** Guidance systems have used direction and rate of change of direction for decades for this reason. Distance in metres is not required to close on a target.

### What it costs

The policy is tied to the lens it trained with: a wider field of view makes the same rectangle correspond to a different distance, so changing it means retraining. And you can no longer print "estimated distance: 2.8 m" and compare it against the simulator — you watch the raw readings and judge the system by whether it captures. Both are acceptable in exchange for not needing the attacker's dimensions.

### Why the reward may still use true distance

**Observations must be obtainable from a camera. Rewards need not be.**

During training the reward function still uses the true distance between the drones. That is allowed because the reward is only ever read by the training algorithm, which uses it to adjust the network's weights; once training finishes, the reward function is never called again. The policy is therefore *scored* using information it never *receives*. This lets us keep a smooth, easy-to-tune reward while the policy still learns to act on camera readings alone.

## 0.5 What the real drone dictates

Everything in Chapters 2 and 3 is shaped by what a Tello can actually do. This section lists those constraints once, so later decisions look inevitable rather than arbitrary.

**The arrangement:** the Tello films, reports numbers about itself, and obeys commands. Your laptop runs the detector and the policy and sends commands back over WiFi. Nothing runs on the drone.

That is still a genuine sim-to-real transfer. What makes a transfer genuine is that the observations and actions match between simulation and reality — not where the arithmetic happens. Moving the computation onto the aircraft would cost hundreds of dollars and demonstrate nothing extra.

| What the Tello dictates | Consequence | Where it is handled |
|---|---|---|
| It accepts **stick commands**, not forces | The policy's four outputs become forward/back, left/right, up/down and turn | 3.1 Part A |
| It accepts commands at roughly **20–30 Hz** | The simulation must decide at the same rate, or every command means a different amount of movement | 1.4 measures it, 3.1 Part B sets it |
| Its video arrives **99–219 ms late** | The policy must be trained on stale readings, or it oscillates on hardware | 1.4 measures it, 3.1 Part D |
| It reports speeds and tilt but **no rotation rates** | Those cannot be observations, because nothing could supply them at flight time | 3.1 Part C leaves them out |
| Its camera sees **~83° across a 4:3 frame** | The simulated camera must match, or every bearing means a different angle | 3.1 Part E |
| It is a **cheap drone** — it drifts, its thrust sags with the battery | Those properties get randomised during training rather than guessed | 3.3 Step 0 |

### The one thing that is not a Tello

The simulation flies Isaac Lab's **Crazyflie** model, because that is the quadcopter Isaac Lab ships and there is no Tello asset. This matters less than it sounds:

- **The airframe does not transfer, and does not need to.** Once 3.1 replaces forces with stick commands, the simulated body is a generic hovering object responding to velocity commands — which is exactly what a Tello is from your code's point of view.
- **The attacker's appearance does matter.** A detector trained only on Crazyflie renders will not reliably find your toy quadcopter. Chapter 7.4 fixes this by retraining on footage of the real target.

What actually transfers is the policy's decision-making, and that only ever saw camera readings and its own motion — never an airframe.

## 0.6 The chapter map

```
Ch.1  FOUNDATIONS        Isolate the environment, verify the install, create
      (5 x 1.5h)         the project, build the vision env — then SET UP AND
                         MEASURE THE TELLO before any policy is designed
        |
Ch.2  THE ARENA          Two drones in one scene: defender (physics) and
      (2 x 1.5h)         attacker (scripted evasive path)
        |
Ch.3  THE POLICY         Stick commands, 17 camera-native observations at your
      (3 x 1.5h)         measured rate and delay, pursuit reward, PPO training
        |
Ch.4  SYNTHETIC DATA     Camera + semantic tags + randomisation
      (3 x 1.5h)         -> thousands of labelled drone images
        |
Ch.5  OBJECT DETECTION   Convert to YOLO format, train YOLOv8n, export to ONNX
      (2 x 1.5h)
        |
Ch.6  INTEGRATION        Reading converter, camera-only capture, simulated
      (3 x 1.5h)         demo — the dress rehearsal
        |
Ch.7  THE REAL TEST      Export the policy, build the flight script, fly in
      (4 x 1.5h)         stages, improve from recorded flights
```

**Chapters 3 and 4/5 are independent.** If a training run is cooking overnight, start Chapter 4 in parallel. The only hard dependency is Chapter 6, which needs both.

**Why the Tello comes first.** Chapter 1.4 measures three things — control rate, video delay, available telemetry — and Chapter 3 builds the policy around them. Measuring afterwards would mean training, discovering a mismatch, and training again.

## 0.7 Keeping the two environments from breaking each other

The single biggest source of pain in this kind of project is Python package conflicts. Our defense is **strict environment separation**:

```
   env_isaaclab  ──clone (1.0)──►  env_drone            drone_vision
   (untouched,                     (this project)       (built clean)
    other projects)                      │                    │
                                         ▼                    ▼
                              ┌────────────────────┐  ┌────────────────────┐
                              │ Python 3.11        │  │ Python 3.11        │
                              │ isaacsim 5.1       │  │ ultralytics        │
                              │ isaaclab (editable)│  │ its own torch      │
                              │ torch 2.7.0+cu128  │  │  (version doesn't  │
                              │ skrl, tensorboard  │  │   matter here)     │
                              │ drone_pursuit (-e) │  │                    │
                              │ + onnxruntime      │  │ OUT: best.onnx     │
                              └────────▲───────────┘  └─────────┬──────────┘
                                       │                        │
                                       └── the .onnx FILE ──────┘
                                           crosses over.
                                           Packages never do.
```

**Rule 1:** never `pip install ultralytics` (or anything torch-touching) into `env_drone`.
**Rule 2:** the only thing we add to `env_drone` is `onnxruntime` — pure inference, no torch dependency.
**Rule 3:** before every chapter that introduces a tool, we run a ≤5-minute smoke test (marked 🧪) to prove compatibility *before* investing hours.
**Rule 4:** `drone_vision` is built from scratch, never cloned — a copy of Isaac Lab's stack would be partially upgraded by Ultralytics, which is the confusing middle state we are trying to avoid.
**Rule 5:** `env_isaaclab` is left alone entirely. It stays available for your other Isaac Lab work, and it is what you re-clone from if `env_drone` ever becomes unusable.

## 0.8 Prerequisites checklist

You need, before Chapter 1:

- **Windows 10/11**, NVIDIA RTX GPU (8 GB+ VRAM recommended — cameras and YOLO both want VRAM), recent NVIDIA driver
- **Isaac Sim 5.1** installed via pip. Isaac Sim 5.x requires **Python 3.11**
- **Isaac Lab 2.3.x** installed from source into a conda environment. The code here targets the 2.3 API (`isaaclab.*` module names, not the older `omni.isaac.lab.*`)
- **conda** (Miniconda or Anaconda), and familiarity with creating and activating environments
- ~30 GB free disk (datasets + checkpoints + YOLO runs)
- Optional but recommended: **VS Code** with the Python extension pointed at each env

This tutorial uses one helper script, **`check_setup.py`**, which verifies your environment automatically. It reads `requirements-lock.txt` and `isaaclab_commit.txt` from `C:\projects\drone_pursuit` — the outer folder — which is its default. If you keep those two files elsewhere, set `DRONE_PURSUIT_DIR` to that location. Place it in your Isaac Lab repository root. It checks package versions, GPU and CUDA availability, every Isaac Lab API the tutorial calls, drift against your locked package list, and — with `--quick` — runs a short real training job to confirm the learning loop works. From subchapter 1.4 onward, `--hardware` also checks the drone packages and confirms the three Tello measurements are recorded. Project files live in `C:\projects\drone_pursuit`; adjust that path throughout if you prefer another location.

Two conventions used throughout:

**Every path is written in full, from `C:` onward.** Two roots appear repeatedly, and if yours differ, substitute them consistently:

| Root | Meaning |
|---|---|
| `C:\Users\[YOUR_USER]\IsaacLab` | your Isaac Lab repository |
| `C:\projects\drone_pursuit\drone_pursuit` | your project root, created by the wizard in 1.2 |
| `C:\projects\drone_pursuit` | the outer folder holding only `constraints.txt`, `requirements-lock.txt` and `isaaclab_commit.txt` |

**Every step begins with the conda environment it runs in**, marked like this:

> **Environment:** `env_drone`

Some steps only edit files and need no environment; those say so. Commands use `isaaclab.bat`, the Windows equivalent of the `./isaaclab.sh` seen in Linux-oriented documentation.

---
</details>

# Chapter 1 — Foundations: Environments & Compatibility

## 1.0 Isolate the project before installing anything (≤1.5h)

<details> 

<summary>Expand: </summary>

> **What / Why / How it contributes:** Every later subchapter installs packages or edits code. This one sets up three protections first: a private copy of the conda environment, a pinned Isaac Lab commit, and a frozen package list. Together they mean a later breakage can be traced to a specific change instead of guessing, and a ruined environment can be restored in minutes instead of hours.

### Why the order in this subchapter matters

You are about to copy your environment. **Copying a broken environment copies the breakage**, so the first thing to do is confirm the original is healthy — not the last.

```
   Step 1  verify env_isaaclab is healthy   ← check BEFORE copying
              │
   Step 2  pin Isaac Lab's git state        ← the code the copy will point at
              │
   Step 3  clone  env_isaaclab → env_drone  ← the copy itself
              │
   Step 4  verify the clone is healthy      ← check AFTER copying
              │
   Step 5  freeze the package list          ← record what "working" looks like
```

### Step 1 — Confirm the source environment is healthy

> **Environment:** `env_isaaclab` — your existing Isaac Lab environment. `env_drone` does not exist yet.

Runs `check_setup.py` against your existing Isaac Lab environment to prove it is healthy before you copy it. Copying a broken environment copies the breakage, and every later chapter runs inside that copy.

*Run from:* `C:\Users\[YOUR_USER]\IsaacLab`
```bat
conda activate env_isaaclab
cd C:\Users\[YOUR_USER]\IsaacLab
isaaclab.bat -p C:\Users\[YOUR_USER]\IsaacLab\check_setup.py --headless --skip-training
```

Replace `env_isaaclab` with whatever your working Isaac Lab environment is called.

You want **0 failures**. Several warnings are expected at this point: `drone_vision` and `onnxruntime` belong to later subchapters, and the conda-env warning is expected because `env_drone` does not exist yet.

If anything fails, fix it before continuing.

### Step 2 — Pin Isaac Lab's git state

> **Environment:** any. These are git commands and do not depend on the active conda environment.

Creates a private git branch from Isaac Lab's current commit and records the commit ID (a SHA, the unique fingerprint of a snapshot). Your project calls Isaac Lab's source directly, so this stops a future `git pull` from changing function signatures underneath your Chapter 2 code.

Your Isaac Lab is an editable install from a source checkout, which means your code calls whatever is in that folder *right now*. A `git pull` on `main` can change function signatures without pip recording anything — the package version would not move, but your Chapter 2 code would break.

```bat
cd C:\Users\[YOUR_USER]\IsaacLab
git switch -c drone-pursuit-base
## -> Used git switch -c drone-pursuit-base_aug26 as a previous branch with the same name had already been created.
git rev-parse HEAD
```

The first command creates a branch from your current commit and switches to it. `main` can now move freely without affecting you. Copy the SHA that `git rev-parse` prints into your project notes:

*Run from:* `any folder`
```bat
mkdir C:\projects\drone_pursuit
git rev-parse HEAD > C:\projects\drone_pursuit\isaaclab_commit.txt
```

That file is what lets you return to this exact state months later, and it is what the checker compares against.

### Step 3 — Clone the environment

> **Environment:** start from `base` (run `conda deactivate` first), because you cannot clone an environment while it is active. You end this step inside `env_drone`.

Copies `env_isaaclab` into a new `env_drone`, then removes any other projects that came along. From here every command in this tutorial runs in `env_drone`, so nothing you install for this project can disturb your other Isaac Lab work.

*Run from:* `any folder`
```bat
conda deactivate
conda create -n env_drone --clone env_isaaclab
```

This takes a few minutes and produces an exact copy of a stack you have just verified. Nothing is downloaded or re-resolved, so no version can shift.

**Why clone rather than build fresh.** A from-scratch install would resolve current versions of everything, including some that are known to break this stack, and Isaac Sim's wheels take up to an hour to download. Cloning preserves the exact torch and CUDA pairing that the checker confirmed works on your GPU.

**Then remove any other projects that came along.** A shared Isaac Lab environment often has other external projects installed into it, and those register their own task names into the same registry as yours. List what is there and uninstall anything unrelated to this project:
- in this case, I removed `isaac_lab_tutorial` `isaaclab_arena`

*Run from:* `any folder`
```bat
conda activate env_drone
pip list | findstr isaac
pip uninstall -y isaac_lab_tutorial isaaclab_arena
## pip uninstall -y <any-other-external-project-names>
```

Leave the Isaac Lab extensions themselves (`isaaclab`, `isaaclab_tasks`, `isaaclab_assets`, `isaaclab_rl`, `isaaclab_mimic`) — those are the framework.

**From here on, every command in this tutorial runs in `env_drone`** unless it explicitly says `drone_vision`.

### Step 4 — Confirm the clone is healthy

> **Environment:** `env_drone`

Runs the checker inside the new copy, this time with `--quick` so it also trains briefly. It confirms the clone works and that the git pin from Step 2 is visible to the checker.

*Run from:* `C:\Users\[YOUR_USER]\IsaacLab`
```bat
conda activate env_drone
cd C:\Users\[YOUR_USER]\IsaacLab
isaaclab.bat -p C:\Users\[YOUR_USER]\IsaacLab\check_setup.py --headless --quick
```

`--quick` adds a short real training run, so this also confirms that the quadcopter task trains and saves a checkpoint. Two rows should differ from Step 1: the conda env now reads `env_drone`, and the Isaac Lab git check shows the `drone-pursuit-base` branch with a matching commit.

Expect this to take three to four minutes. The script prints a progress counter so you can tell it is still working.

### Step 5 — Record the working state and lock setuptools

> **Environment:** `env_drone`

Freezes the current package list and sets an upper bound on setuptools. The frozen list is what the checker compares against when something breaks later, and the bound stops a future install from silently upgrading a package that tensorboard depends on.

One command does both jobs:

```bat
cd C:\projects\drone_pursuit && pip freeze > C:\projects\drone_pursuit\requirements-lock.txt && (echo setuptools^<81) > C:\projects\drone_pursuit\constraints.txt && pip config --site set install.constraint C:\projects\drone_pursuit\constraints.txt
```

Three things happen:

**`requirements-lock.txt`** captures every package version at the moment everything works. Not for reinstalling from — for **comparing against**. When something breaks in Chapter 5, the checker diffs the current state against this file and tells you which packages moved.

**`constraints.txt`** constrains any new package installation with pip install from updating `setuptools` to a version higher than 81, which would break things. 

Confirm it took, and confirm the scope:

*Run from:* `any folder`
```bat
pip config --site list
```

You should see `install.constraint=...`. Now check that it did not leak wider:

*Run from:* `any folder`
```bat
conda activate env_isaaclab
pip config --site list
conda activate env_drone
```

The middle command should print nothing. That is the proof the setting lives in `env_drone` alone.

If setuptools ever slips past the bound anyway, you will see `ModuleNotFoundError: No module named 'pkg_resources'`, and the repair is `pip install "setuptools<81"`. The checker tests this in Section 1.

> ✅ **Checkpoint 1.0**
> 1. `check_setup.py --skip-training` reported 0 failures in your original environment
> 2. `git branch --show-current` in the Isaac Lab folder shows `drone-pursuit-base`
> 3. `isaaclab_commit.txt` exists and contains a SHA
> 4. `conda env list` shows `env_drone`, and the checker passes inside it
> 5. `requirements-lock.txt` and `constraints.txt` exist, and `pip config list` shows the constraint

---
</details>

## 1.1 Understand the flight code you will build on (≤1.5h)

<details> 
<summary>Expand: </summary>
> **What / Why / How it contributes:** Isaac Lab ships a working quadcopter task. Rather than building drone flight from scratch, this project copies that task and changes what it chases. This subchapter confirms the task flies on your machine, then walks through its source file — because in Chapter 1.2 that file becomes your project, and in Chapter 2 you will edit almost every method in it.

### Step 1 — Watch the built-in task fly

> **Environment:** `env_drone`

Plays back the checkpoint the checker trained in 1.0, so you see Isaac Lab's quadcopter actually fly. This is your known-good reference: if your own copy misbehaves in Chapter 2, you can run this one to tell whether the problem is your code or your setup.

Subchapter 1.0 already trained this task and verified a checkpoint was saved. Load that checkpoint and watch it:

*Run from:* `C:\Users\[YOUR_USER]\IsaacLab`
```bat
conda activate env_drone
cd C:\Users\[YOUR_USER]\IsaacLab
isaaclab.bat -p C:\Users\[YOUR_USER]\IsaacLab\scripts\reinforcement_learning\skrl\play.py --task Isaac-Quadcopter-Direct-v0 --num_envs 32
```

A viewport opens with 32 Crazyflies flying toward small goal markers. They will be unstable — the checker trains only briefly — but they should move toward their goals rather than tumbling immediately.

**If you want a longer run first**, train for more iterations before playing:

*Run from:* `C:\Users\[YOUR_USER]\IsaacLab`
```bat
isaaclab.bat -p C:\Users\[YOUR_USER]\IsaacLab\scripts\reinforcement_learning\skrl\train.py --task Isaac-Quadcopter-Direct-v0 --num_envs 2048 --headless --max_iterations 1000
```

Reduce `--num_envs` if you hit an out-of-memory error, and note the largest value that works — Chapter 3 reuses it.

### Note — two folders called `scripts`, and both are correct

From Chapter 1.2 onward you will use two similarly named paths. They are different folders:

| Path | Belongs to | Used in |
|---|---|---|
| `C:\Users\[YOUR_USER]\IsaacLab\scripts\reinforcement_learning\skrl\train.py` | The Isaac Lab repository | Chapter 1.1 only |
| `scripts\skrl\train.py`, run from your project root | Your project | Chapter 1.2 onward |

If a command reports that a file does not exist, check which of the two folders you are in before anything else. This is the most common cause of that error in this tutorial.

### Step 2 — Read the source file

> **Environment:** none needed — you are reading a file, not running anything.

Walks through `quadcopter_env.py`, the file you copy into your project in 1.2 and edit throughout Chapters 2 and 3. Reading it now means those edits are modifications to something you understand rather than instructions you follow blindly.

Open:

```
C:\Users\[YOUR_USER]\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\quadcopter\quadcopter_env.py
```

This is the Direct workflow you have seen in the cartpole tutorial: a config class holding all the tunable values, and an environment class implementing `_setup_scene`, `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones` and `_reset_idx`.

| What you see in the file | What it is, why it is there, and what would break without it |
|---|---|
| `class QuadcopterEnvCfg(DirectRLEnvCfg)` | **A settings sheet.** Every adjustable number for this task lives here as a plain named value: how long an attempt lasts, how many drones train at once, how strong the rotors are, how behaviour is scored. Nothing here does anything — it only records choices. **Why separate from the code below:** you can change how the task behaves without touching how it works, and Isaac Lab can make variants by copying this sheet and overriding one value. **Without it:** all these numbers would be scattered as raw figures through the code, so tuning would mean hunting through methods, and every variant would be a copy of the whole file. |
| `CRAZYFLIE_CFG` (imported from `isaaclab_assets`) | **A ready-made recipe for putting a Crazyflie into the world.** Isaac Sim stores 3D models as USD files — a file format describing shapes, joints and materials. That file alone is not enough to simulate: something must also state the drone's mass, how stiff its joints are, how carefully the physics engine should solve it, and where it starts. This recipe bundles the file's web address together with all those physical settings, so any task can say "give me a Crazyflie" in one line. **Without it:** every task using a Crazyflie would hand-write forty lines of loading and physics settings, and those copies would drift apart as people tweaked them. |
| `robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")` | **Taking that recipe and choosing where the drone will live in the scene.** An *articulation* is the word for a robot made of connected moving parts (from Latin *articulus*, a joint) — as opposed to a single solid object. `.replace(...)` makes a copy of the recipe with one detail changed, leaving the original untouched for other tasks. The changed detail is the drone's address in the scene, and `env_.*` is a pattern meaning "every environment" — so this single line places one drone into all four thousand copies. **Without the pattern:** you would need one line per environment. |
| `def _setup_scene(self)` | **Builds the world once, before the simulation starts.** It creates the drone from the recipe, adds a floor and a light, and then duplicates the whole arrangement thousands of times. The duplication line is what makes parallel training possible: you describe one arena and receive four thousand. It also registers the drone with the scene, which is what keeps the drone's position and speed readings refreshed every step. **Without registering it:** the drone would appear on screen but its readings would never update — a silent failure where nothing errors and nothing behaves sensibly. |
| `def _pre_physics_step(self, actions)` | **Translates the neural network's output into physical push.** The network emits four numbers between −1 and +1 that mean nothing on their own. This method decides what they mean: the first becomes an upward push measured in newtons, the other three become twisting forces. It also rescales the first number so that −1 means "rotors off" rather than "pull downward", because a real rotor can stop but cannot suck a drone into the floor. **Without it:** the network's raw numbers would reach the physics engine as forces — unscaled to the drone's weight, unclamped when the untrained network emits extremes, and capable of negative thrust the policy would happily learn to exploit. |
| `def _apply_action(self)` | **Hands those forces to the physics engine, repeatedly.** It runs on every physics tick, which happens more often than the network decides — so one decision gets applied several times. **The surprising part:** the force is applied to the drone's central body, not to its four rotors. The rotors spin visually but no airflow or blade behaviour is calculated; the simulation jumps straight to the *result* of rotor physics, which is one net push and three net twists. **Without the repetition:** forces in this physics engine do not persist between ticks, so the drone would receive a single nudge and then coast. |
| `def _get_observations(self)` | **Assembles everything the network is allowed to know.** Twelve numbers: how fast the drone is moving, how fast it is rotating, which direction is down, and where the goal sits relative to the drone. That last part is the important one — the goal is expressed from the drone's own point of view ("two metres ahead and slightly to my left") rather than as a fixed location in the world. **Why that matters:** the first description is learnable once and works anywhere in the arena; the second would have to be memorised location by location, and a drone that learned to hover at one spot would be useless elsewhere. **Without this method:** the network would receive nothing and could only act on noise. |
| `self._desired_pos_w = torch.zeros(...)` in `__init__` |This line creates that table that represents the position of the defender drone target in space coordinates (left–right, forward–back, up–down) - first as a static object, later it will be dynamic position of the attacker drone. One row per drone, one column per coordinate. For now, we will just fill it with zeros to allocate memory space. |
| `def _get_rewards(self)` | **Scores how good the last moment was, one number per drone.** It rewards being near the goal and mildly penalises moving fast or spinning. **Why the penalties exist:** with only a closeness reward, the quickest way to score is to hurl yourself at the goal and overshoot repeatedly; the penalties buy stability, so the drone learns to arrive and stay. **Who reads this:** only the training algorithm, which uses it to adjust the network's weights.|
| `def _get_dones(self)` | It is the method that decides which episodes have terminated (drone failed) or truncated (ran out of time) **Without this method:** attempts would never end, and crashed drones would lie on the floor generating useless experience forever. |
| `def _reset_idx(self, env_ids)` | **Restarts the attempts that just ended — and only those.** `env_ids` is a list of which drones finished. Out of four thousand, perhaps thirty crashed this step while the rest are mid-flight, so every line inside indexes by that list. It reports statistics for the finished attempts, picks a fresh random goal for each, and returns those drones to their starting position and zero speed. **Why the goal is random rather than fixed:** a fixed goal would let the policy memorise one flight path instead of learning to fly wherever the target is. **Without the index list:** writing the whole table would teleport four thousand healthy drones back to the start mid-flight. |
 
---
 
## The order things run in
 
```
ONCE, AT STARTUP
   _setup_scene()      build the world, duplicate it thousands of times
   __init__()          reserve memory for everything reused later
 
THEN, REPEATEDLY — one environment step = one network decision
   _pre_physics_step(actions)     four numbers  ->  push and twist
        |
   _apply_action()  -> physics tick   ┐  repeated `decimation` times,
   _apply_action()  -> physics tick   ┘  because forces do not persist
        |
   _get_observations()   what the network senses next
   _get_rewards()        how well it just did      (training only)
   _get_dones()          which attempts ended
        |
   _reset_idx(finished)  restart only those, with new random goals
```
 
## The five lines that become the pursuit project
 
1. `robot: ArticulationCfg = CRAZYFLIE_CFG.replace(...)` — a second line like this adds the attacker drone.
2. `self.scene.articulations["robot"] = self._robot` in `_setup_scene` — the attacker gets registered the same way.
3. `self._desired_pos_w = torch.zeros(...)` — the stored goal is replaced by reading the attacker's live position.
4. The goal term inside `_get_observations` — replaced by camera-style readings: where the target appears in view and how large it looks.
5. The distance term inside `_get_rewards` — "be near the goal" becomes "close on a moving target and capture it".
Everything else — the timing, the duplication, the reset mechanics, the force conversion — carries over essentially unchanged.

Two things in this file matter more than the rest.

**Drones are driven by external forces, not joint efforts.** In cartpole, effort is applied to a joint and PhysX moves the cart. Here there is no joint to push: the policy's four numbers become a force pushing the body upward and three torques rotating it. This is the main structural difference from joint-driven tasks, and it carries unchanged into your project.

**Observations are expressed in the drone's own frame.** The goal position is converted with `subtract_frame_transforms` so the policy receives "the target is 2 m ahead and slightly left of my nose" rather than "the target is at world coordinates (14.2, −3.7, 1.5)". The first is learnable anywhere in the arena; the second would have to be memorised per location. This is also why the camera in Chapter 6 fits so cleanly: a camera naturally sees the world in the body frame of the drone carrying it.

**`_desired_pos_w` is the line this whole project turns on.** In Chapter 2 that fixed hover goal becomes a second drone that moves. Everything else — the rewards, the observations, the camera — follows from that single substitution.

> ✅ **Checkpoint 1.1**
> 1. `play.py` shows drones flying toward goal markers
> 2. You can point to the line where the goal position enters the observation (look for `desired_pos_b`)
> 3. You can explain why the drone receives thrust and moments rather than joint efforts

---
</details>

## 1.2 Create the external project: `drone_pursuit` (≤1.5h)

> **What / Why / How it contributes:** We scaffold a clean external project with the template wizard, then copy Isaac Lab's quadcopter task folder into it and point the project at that task instead of the wizard's cartpole. Working in an external project means our code survives Isaac Lab updates, lives in its own git repository, and leaves the Isaac Lab source untouched. By the end, your own copy of the hover task trains under your own task name: the stage on which the pursuit is built.

### Step 1 — Run the template wizard

> **Environment:** `env_drone`

Runs Isaac Lab's project generator, which creates an external project folder with working scripts, packaging and RL wiring. External means your code lives outside the Isaac Lab repository, so Isaac Lab updates cannot break it and your work has its own git history.

```bat
cd C:\Users\[YOUR_USER]\IsaacLab
isaaclab.bat --new
```

Wizard answers (arrow keys + space + enter):
- **Type of project:** `External`
- **Project path:** somewhere outside the Isaac Lab repo, e.g. `C:\projects\drone_pursuit`
- **Project name:** `drone_pursuit`
- **Workflow:** `Direct | single-agent`  ← we drive the attacker by script, so from RL's point of view there is only ONE agent (the defender). True multi-agent (IPPO/MAPPO) is a Chapter 6 "next steps" topic.
- **RL library:** `skrl`   → algorithm `PPO`

### Step 2 — Install the external project

> **Environment:** `env_drone`
Installs your project as an editable Python package, which is what makes its task discoverable by `train.py`. The `fc` comparison afterwards catches any package pip silently upgraded — the most common way a working environment breaks.

Install, with `env_drone` active:

*Run from:* `any folder`
```bat
conda activate env_drone
cd C:\projects\drone_pursuit\drone_pursuit
python -m pip install -e C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit
```

(`-e` means editable: source edits take effect without reinstalling.)

Confirm nothing shifted underneath you:
- Pip resolves dependencies while installing, and it can upgrade or replace packages you didn't ask about - and cause versioning mismatches that can break things in this project, downstream. This test compares the environment before and after, so you find out immediately if pip changed anything important, inadvertently.
- So it's good to run this test every time something new is installed to the project. 

```bat
cd C:\projects\drone_pursuit\drone_pursuit
pip freeze > C:\projects\drone_pursuit\drone_pursuit\requirements-after-install.txt
fc C:\projects\drone_pursuit\requirements-lock.txt C:\projects\drone_pursuit\drone_pursuit\requirements-after-install.txt
```

`fc` is Windows' file-compare. You expect to see `drone_pursuit` added plus a few small packages. If **torch, numpy, setuptools, or protobuf** appears in that diff, stop** — something re-resolved it, and everything downstream depends on the CUDA build you verified in 1.0.

### Step 3 — Understand the folder layout, and create the folders the tutorial needs

> **Environment:** any. These are folder commands.

Explains the two-level folder structure the wizard creates and which of the two every path in this tutorial refers to. Mixing them up is the most common source of "file not found" from here to Chapter 7.

The wizard nests the project inside a folder of the same name, so you now have two levels. They hold different things, and mixing them up is the most common source of "file not found" later:

```
C:\projects\drone_pursuit\                 ← the OUTER folder, created in 1.0
│   constraints.txt                          setup records only. Nothing is run
│   requirements-lock.txt                    from here after Chapter 1.0.
│   isaaclab_commit.txt
│
└── drone_pursuit\                          ← the PROJECT ROOT. Run every
    │                                          command from here from now on.
    ├── scripts\                             your scripts
    │   ├── skrl\                            train.py, play.py (wizard-made)
    │   ├── sdg\                             Chapter 4 — data generation
    │   └── demo\                            Chapters 6 and 7
    ├── source\drone_pursuit\                the task code you will edit
    │     └── ...\tasks\direct\quadcopter\    ← after Step 5, this is the task
    ├── models\                              the two ONNX files — the transfer
    ├── data\                                images, labels, real footage
    ├── flights\                             Chapter 7 flight recordings
    ├── logs\                                training runs (auto-created)
    └── runs\                                YOLO runs (auto-created)
```

**Whenever this tutorial writes a path starting `C:\projects\drone_pursuit\`, it means the project root** — the inner folder — except for `constraints.txt` and `isaaclab_commit.txt`, which stay in the outer one.

Some of those folders do not exist yet. **Each is created in the subchapter that first needs it**, so nothing is made before you know what it is for. Watch for a short `mkdir` line at the start of those subchapters.

Two kinds of folder never need creating: `data\yolo\` is made by Chapter 4.3's conversion script, and `logs\` and `runs\` are made by the training tools when they first write to them.

**Why the manual ones matter.** Windows will not invent a folder for you, and Python's `open()` fails if the parent folder is missing. The `flights\` folder in Chapter 7 is the sharpest case: the flight script opens its log immediately after takeoff, so a missing folder crashes it with a drone already in the air.

### Step 4 — Verify the template task runs

> **Environment:** `env_drone`

Trains the wizard's placeholder cartpole for 20 iterations. Nothing here concerns drones — it proves the plumbing works (registration, environment creation, skrl) before Step 5 swaps the task, so any later failure is the swap and not the scaffolding.

The wizard generates a cartpole placeholder task registered as `Template-Drone-Pursuit-Direct-v0` (check the exact name in `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\drone_pursuit\__init__.py`):

*Run from:* `any folder`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\list_envs.py
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\train.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 64 --headless --max_iterations 20
```

If the placeholder cartpole trains for 20 iterations (or 640 timesteps), your project plumbing (registration → gym.make → env → skrl) is sound. 
Verify, after the progress bar reaches 100%, it should say `640/640`

### Step 5 — Copy the quadcopter task into your project

> **Environment:** none needed — you are copying a folder and editing two lines.

The wizard gave you a working *project* wrapped around the wrong *task*. Everything outside the task folder is already correct: the package installs, the scripts run, the registration works, skrl connects. What sits inside is a cartpole — a cart on a rail balancing a pole — and this project needs a drone. You replace it by copying Isaac Lab's whole quadcopter folder in and re-pointing one import.

```
   WHAT THE WIZARD BUILT              WHAT YOU WANT
   ┌────────────────────────┐         ┌────────────────────────┐
   │ project plumbing  ✅   │         │ project plumbing  ✅   │  ← unchanged
   │  ┌──────────────────┐  │   ==>   │  ┌──────────────────┐  │
   │  │ cartpole task    │  │         │  │ quadcopter task  │  │  ← replaced
   │  └──────────────────┘  │         │  └──────────────────┘  │
   └────────────────────────┘         └────────────────────────┘
```

#### 5a — Copy the folder

Copy this folder:

```
C:\Users\[YOUR_USER]\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\quadcopter
```

Into this folder:

```
C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\
```

Delete folder `drone_pursuit` in:
```
C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct
```
(The project creator wizard creates by default a folder with the cartpole task and names it after your project. Delete it to avoid confusion with the folder for the quadcopter task we will create.)

Result:

```
tasks\direct\
    quadcopter\           ← what you just copied
        __init__.py
        quadcopter_env.py
        agents\skrl_ppo_cfg.yaml
```

The `agents\` folder comes along, so the drone's training settings arrive with it — there is no separate settings file to copy.

#### 5b — Point the project at the new task

Open:

```
C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\__init__.py
```

Swap which task gets imported:

```python
# from . import drone_pursuit   # <<< COMMENT OUT (or delete) the wizard's cartpole
from . import quadcopter        # <<< ADD THIS LINE
```

Registration happens as a side effect of importing, so this line decides which task names exist. The cartpole folder stays on disk, harmless.

#### 5c — Rename the task

Open:

```
C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\__init__.py
```

Change the `id` only:

```python
gym.register(
    id="Template-Drone-Pursuit-Direct-v0",   # <<< was "Isaac-Quadcopter-Direct-v0"
    entry_point=f"{__name__}.quadcopter_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env:QuadcopterEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

**Why rename:** `Isaac-Quadcopter-Direct-v0` is already registered by Isaac Lab. Two tasks with the same name collide and the run fails. The `Template-` prefix also makes `list_envs.py` show your tasks separately from NVIDIA's hundreds.

**Leave the class names alone.** The entry-point strings read as *file : class* and already match — renaming would mean editing these strings too, for no benefit.

#### 5d — Verify

*Run from:* `any folder`
```bat
conda activate env_drone
python C:\projects\drone_pursuit\drone_pursuit\scripts\list_envs.py
```

`Template-Drone-Pursuit-Direct-v0` should appear. If it does not, check the import line in 5b.

⚠️ If an error names something you commented out, delete the `__pycache__` folders under `tasks\direct\` and retry.

#### The file you will edit from here on

Everything from Chapter 2 onward edits this one file:

```
C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py
```

It holds both classes — `QuadcopterEnvCfg` (the settings) and `QuadcopterEnv` (the task).

### Step 6 — Train your copy of the hover task

> **Environment:** `env_drone`

Runs your copied task end to end, to prove the swap in Step 5 worked before Chapter 2 starts editing code. `list_envs.py` only proved the task registers; this proves the environment builds, the drone flies, and a checkpoint saves and reloads.

- train.py: proves the environment builds and learns — but runs `--headless`, so you see nothing fly. It also writes a checkpoint.

*Run from:* `any folder`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\train.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 2048 --headless --max_iterations 20
```
- play.py: runs a new simulation where drones act using what they learned during training. They no longer learn here. It is a net new simulation, not a recording of the training.

*Run from:* `any folder`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\play.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 32
```

Same wobbly hover as 1.1 — but now it is *your* copy, in *your* repository, under *your* task name. From here on you edit these files freely; Isaac Lab's originals stay untouched.

Note the log folder has changed. Step 4's run filed itself under `cartpole_direct`, because that name came from the wizard's settings file. Now that the quadcopter's own `agents\skrl_ppo_cfg.yaml` is in use, runs are filed under the quadcopter's experiment name instead.

> ✅ **Checkpoint 1.2**
> 1. `list_envs.py` shows your task name
> 2. Your copy of the quadcopter task trains and plays
> 3. `git init` + first commit done; Chapter 2 edits `quadcopter_env.py` heavily

- Commit first checkpoint:
```bat
cd C:\projects\drone_pursuit\drone_pursuit
git init
git add .
git commit -m "quadcopter task copied and registered - baseline before Chapter 2"
```
- Also commit to GitHub if you have an account:

*Run from:* `any folder`
```bat
git remote add origin https://github.com/[YOUR_GITHUB_USER]/drone_pursuit.git
git add -A
git commit -m "short description of what changed"
git push -u origin main
```

---

## 1.3 Build the vision env and test every tool boundary (≤1.5h)

> **What / Why / How it contributes:** We create the second conda env (`drone_vision`) that will train YOLOv8 in Chapter 5, and — critically — we run ALL the cross-tool compatibility tests NOW, before investing hours in data generation. We prove: (1) Ultralytics trains on your GPU, (2) a YOLO model exports to ONNX, (3) that ONNX file runs inside `env_drone` via onnxruntime. If the full round trip works with a toy model today, it will work with your real model in Chapter 6.

### Step 1 — Create the vision env

> **Environment:** start from `base`, because you cannot create an environment from inside another. You end this step inside `drone_vision`.

Creates `drone_vision`, the second conda environment, and installs Ultralytics (the library that trains YOLO). It exists so that YOLO's torch never meets Isaac Lab's, and it is where you train the detector in Chapter 5.

**Create this one from scratch — do not clone `env_drone`.** The entire purpose of this environment is that its torch is unrelated to Isaac Lab's, so starting from a copy of Isaac Lab's stack would defeat it and invite a confusing partial upgrade.

*Run from:* `any folder`
```bat
conda create -n drone_vision python=3.11 -y
conda activate drone_vision
pip install ultralytics
```

Ultralytics pulls in its own torch. Its version is irrelevant here, because this env never runs Isaac Lab.

### Step 2 — 🧪 Smoke test #2: GPU training round-trip (5 min of compute)

> **Environment:** `drone_vision`

Trains YOLOv8n on `coco8` (a bundled 8-image dataset Ultralytics downloads on first use) for 3 epochs. It proves the install works end to end and, more importantly, that training actually reaches your GPU — a failure you would otherwise only notice in Chapter 5, as hours instead of minutes.

*Run from:* `C:\projects\drone_pursuit\drone_pursuit`
```bat
yolo detect train data=coco8.yaml model=yolov8n.pt epochs=3 imgsz=640
```

Watch the console: it should name your CUDA device, something like `Ultralytics 8.4.138  Python-3.11.16 torch-2.11.0+cu128 CUDA:0 ([YOUR GPU], 16376MiB)`, and the `GPU_mem` column must be non-zero. Three epochs on 8 images takes ~1–2 minutes.

Afterwards you can delete the toy leftovers — they are not yours:

*Run from:* `any folder`
```bat
rmdir /s /q C:\projects\drone_pursuit\drone_pursuit\datasets
rmdir /s /q C:\projects\drone_pursuit\drone_pursuit\runs\detect\train
```

### Step 3 — 🧪 Smoke test #3: export to ONNX

> **Environment:** `drone_vision`

Tests that a trained model in `drone_vision` can be converted to ONNX (a file format that stores trained neural networks). We will later save the detector we train in `drone_vision` into an `.onnx` file to transfer it to `env_drone`, to be used by the Isaac Lab simulation in Chapter 6.

*Run from:* `C:\projects\drone_pursuit\drone_pursuit`
```bat
yolo export model=yolov8n.pt format=onnx imgsz=640
```

In the output, look for `ONNX: export success  13.4s, saved as 'yolov8n.onnx'`. This writes the file in the current folder.

Create the models folder and copy it there — this is where both exported models will live:

*Run from:* `any folder`
```bat
mkdir C:\projects\drone_pursuit\drone_pursuit\models
copy yolov8n.onnx C:\projects\drone_pursuit\drone_pursuit\models\yolov8n.onnx
```

### Step 4 — 🧪 Smoke test #4: run the ONNX file inside the Isaac Lab env

> **Environment:** `env_drone` — this is the crossing point: the file made in `drone_vision` is opened in `env_drone`.

Opens the ONNX file from the *other* environment and runs one frame of random numbers through it. This is the boundary the whole two-environment design exists to protect, so proving it now with a toy model means Chapter 6 will work with your real one.

Create the file and open it in Notepad:

```bat
cd C:\projects\drone_pursuit\drone_pursuit
notepad test_onnx.py
```

Notepad offers to create it — click **Yes**. Paste this in, save (Ctrl+S) and close:

```python
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession(
    r"C:\projects\drone_pursuit\drone_pursuit\models\yolov8n.onnx",
    providers=["CPUExecutionProvider"],
)
inp = sess.get_inputs()[0]
print("input:", inp.name, inp.shape)

dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
out = sess.run(None, {inp.name: dummy})
print("output:", out[0].shape)
print("ONNX round-trip works inside env_drone")
```

Run it:

*Run from:* `any folder`
```bat
conda activate env_drone
pip install onnxruntime
python test_onnx.py
```

Expect:

```
input: images [1, 3, 640, 640]
output: (1, 84, 8400)
ONNX round-trip works inside env_drone
```

Two things to understand here, because they matter in Chapter 6:
- **`(1, 3, 640, 640)`** = (batch, channels, height, width). YOLO wants RGB, values 0–1, channels-first. Isaac Lab's camera gives (batch, H, W, channels) uint8, so a small conversion is needed — written in Chapter 6.1. (This smoke test uses the stock square model; your own detector is exported at 480×640 in Chapter 5.2, to match the drone's 4:3 camera.)
- **`(1, 84, 8400)`** = for each of 8400 candidate boxes: 4 box coordinates + 80 class scores. Our custom model will have **1 class**, so its output will be `(1, 5, 6300)`.

Also verify torch survived: `python -c "import torch; print(torch.cuda.is_available())"` must still print `True`. (`onnxruntime` has no torch dependency, so it will — this check is your habit-forming version-conflict audit.)

### Step 5 — 🧪 Smoke test #5: cameras inside Isaac Lab

> **Environment:** `env_drone`

Runs a built-in camera-based task to confirm Isaac Lab can render on your GPU. Chapters 4, 5.2 and 6 all depend on rendering, so a driver or VRAM problem is worth finding in five iterations rather than mid-way through generating 2500 frames.

Run the built-in camera-based cartpole to confirm the rendering pipeline works on your GPU (this is the `--enable_cameras` machinery from the Isaac Lab camera docs):

```bat
cd C:\Users\[YOUR_USER]\IsaacLab
isaaclab.bat -p C:\Users\[YOUR_USER]\IsaacLab\scripts\reinforcement_learning\skrl\train.py --task Isaac-Cartpole-RGB-Camera-Direct-v0 --num_envs 64 --headless --enable_cameras --max_iterations 5
```

We only need 5 iterations — success = it runs without a rendering/VRAM error. If OOM: drop `--num_envs` (camera envs are VRAM-hungry; the docs benchmark ~512 cameras on an RTX 4090 — scale expectations to your card).

> ✅ **Checkpoint 1.3** — every boundary tested:
> 1. `drone_vision` env trains YOLO on GPU
> 2. `yolov8n.onnx` exists and runs under onnxruntime in `env_drone`
> 3. torch+CUDA still healthy in `env_drone` after installing onnxruntime
> 4. Camera-based training runs with `--enable_cameras`
>
> **Every tool now provably talks to every other tool it needs to.** Nothing in Chapters 2–6 introduces a new compatibility risk.

---

## 1.4 Set up the Tello and measure it (≤1.5h)

> **What / Why / How it contributes:** You measure the real drone **before** designing the policy, not after. Three numbers come out of this subchapter — how late its video arrives, how fast it accepts commands, and which telemetry it can report — and all three go straight into Chapter 3. Measuring first means you train once. Measuring afterwards would mean training, discovering a mismatch, and training again.

### Step 1 — Install the SDK

> **Environment:** `env_drone`

Installs `djitellopy`, the Python library that speaks the Tello's command protocol, and OpenCV for its video. Every remaining step in 1.4 uses it, and Chapter 7.2's flight script imports the same two packages.

*Run from:* `any folder`
```bat
conda activate env_drone
pip install -c C:\projects\drone_pursuit\constraints.txt djitellopy opencv-python
```

**If you have never used a hardware SDK:** it is an ordinary Python library. `djitellopy` wraps the drone's wire protocol so that `drone.takeoff()` sends the text `takeoff` as a UDP packet to `192.168.10.1:8889` and waits for `ok`. Without it you would write that socket code yourself. Three kinds of call are all you need:

| Kind | Examples | Purpose |
|---|---|---|
| **Ask** | `get_battery()`, `get_speed_x()`, `get_roll()` | Read telemetry the drone reports about itself |
| **Command** | `takeoff()`, `land()`, `send_rc_control(a,b,c,d)`, `emergency()` | Tell it to act |
| **Stream** | `streamon()`, `get_frame_read()` | Start and read the video |

### Step 2 — Close the drone's network before connecting your laptop

> **Environment:** `env_drone`

Sets a password on the drone's WiFi and hardens how your laptop joins it. By default that network is open and its command port accepts flight commands from anything on it, so this happens before your laptop connects at all.

The Tello broadcasts a WiFi access point with **no password**, and its command port accepts flight commands from anything on that network. Your laptop then joins that open network.

**Set a password first.** The SDK has a command for it:

```python
from djitellopy import Tello
drone = Tello()
drone.connect()
drone.send_control_command("wifi drone_pursuit_net ChooseAStrongPasswordHere")
# the drone reboots its access point — rejoin with the new name and password
```

⚠️ Write the password down; recovery means a factory reset.

**Then three more things, in order of value:**

- **Set the network profile to Public on Windows** when prompted, which disables file and printer sharing. Check afterwards under Settings → Network & Internet → WiFi. Confirm Defender Firewall is on for public networks.
- **Stay on one network adapter.** A second USB adapter lets you keep internet while flying, but puts your laptop on the drone's network and your home network simultaneously, bridging them. With one adapter, flying and internet are mutually exclusive — and that enforced isolation is the strongest protection available here. Buy the adapter for convenience if you want; not for security.
- **For a second-hand unit:** factory-reset it and update the firmware through the official phone app *before* connecting your laptop at all. That replaces whatever the previous owner left on it.

### Step 3 — Prove the link, then fly under program control

> **Environment:** `env_drone`

Two short scripts: one that only reads the battery, one that takes off and lands. They confirm your laptop can both query and command the drone — the foundation everything in Chapter 7 is built on.

```python
# hello_drone.py — nothing spins
from djitellopy import Tello
drone = Tello()
drone.connect()
print(f"battery: {drone.get_battery()} %")
```

A battery percentage means your laptop and drone are talking. Then:

```python
# takeoff_test.py
import time
from djitellopy import Tello
drone = Tello()
drone.connect()
if drone.get_battery() < 30:
    raise SystemExit("charge before flying")
drone.takeoff()
time.sleep(5)
drone.land()
```

**The drone already has an emergency stop** — `drone.emergency()` cuts the motors immediately. Test it once, at low altitude over something soft. You will not write your own; Chapter 7 only adds a keyboard trigger for this existing one. There is also a passive safety net: if the drone receives no command for 15 seconds it lands itself, so a crashed script does not produce a runaway drone.

### Step 4 — MEASUREMENT 1: what telemetry can it report?

> **Environment:** `env_drone`

Prints what the drone can tell you about itself. Two findings shape Chapter 3: speeds arrive in centimetres per second, and rotation rates are not available at all — so they cannot be observations.

```python
print("speed x/y/z :", drone.get_speed_x(), drone.get_speed_y(), drone.get_speed_z())
print("attitude    :", drone.get_roll(), drone.get_pitch(), drone.get_yaw())
```

Two findings, both of which shape Chapter 3:

**Speeds are in centimetres per second.** Your simulation works in metres. This is a natural place to introduce a factor-of-100 error.

**There are no angular rates.** You get attitude *angles* — roll, pitch, yaw — but not how fast they are changing. So the policy cannot be given rotation rates, because at flight time nothing could supply them. Chapter 3.1 therefore leaves them out of the observation list entirely, rather than training on a number the drone cannot report.

That is the whole reason this subchapter comes before Chapter 3.

### Step 5 — MEASUREMENT 2: how late is the video?

> **Environment:** `env_drone`

Measures how long a camera frame takes to reach your laptop, using a stopwatch on screen. Chapter 3.1 trains the policy on readings delayed by exactly this much, because a policy trained on instant readings oscillates on real hardware.

```python
# see_camera.py
import cv2
from djitellopy import Tello

drone = Tello()
drone.connect()
drone.streamon()
reader = drone.get_frame_read()      # background thread, keeps only the NEWEST frame

while True:
    frame = reader.frame
    if frame is not None:
        cv2.imshow("drone camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
drone.streamoff()
cv2.destroyAllWindows()
```

**`get_frame_read()` is the line that matters.** It discards old frames and keeps only the latest. Reading frames sequentially from a queue builds a backlog and your delay grows without limit — that is the cause of the one-second lags people report.

**Measure it with a stopwatch:**

1. Open a millisecond stopwatch on screen.
2. Point the drone's camera at it.
3. Put the video window beside the stopwatch on the same screen.
4. Screenshot both. The difference between the two visible times is your delay.

Take **ten samples**; record minimum, mean and maximum. Published measurements for this drone give 99–219 ms with a mean near 175 ms, and an independent study found 80–120 ms — the spread between studies of identical hardware is exactly why you measure your own.

**Anything near a second means your buffering is wrong**, not your drone.

### Step 6 — MEASUREMENT 3: how fast can you command it?

> **Environment:** `env_drone`

Counts how many stick commands per second the drone accepts. Chapter 3.1 sets the simulation's decision rate to match, because the same command held for longer moves the drone further than the policy expects.

```python
# measure_rate.py — propellers may stay on; neutral commands do not spin them
import time
from djitellopy import Tello

drone = Tello()
drone.connect()
N = 500
start = time.time()
for _ in range(N):
    drone.send_rc_control(0, 0, 0, 0)
print(f"{N / (time.time() - start):.1f} commands per second")
```

Expect roughly 20–30 for a Tello. **This becomes your control rate**, and Chapter 3 will train at it.

### Step 7 — Write the three numbers down

> **Environment:** none needed — you are writing a text file.

Records the control rate, video delay and available telemetry in `project_notes.txt`. Chapter 3.1 reads all three into the policy's configuration, so this file is the handoff between the hardware half and the simulation half.

Put them in `project_notes.txt`. Chapter 3.1 uses all three:

```
control rate        : 20 Hz            → decimation = 5 in the cfg
video delay         : 175 ms mean, 99–219 ms range
                      → 2–5 control steps at 20 Hz
telemetry available : speeds (cm/s), attitude angles. NO angular rates.
camera              : 960x720, ~83 deg horizontal field of view
```

**Converting delay into steps:** `delay_steps = delay_seconds × control_rate`. At 20 Hz, 175 ms is 3.5 steps; the 99–219 ms range spans roughly 2 to 5 steps.

### Step 8 — Confirm the checker agrees

> **Environment:** `env_drone`

Runs the checker with `--hardware`, which verifies the drone packages are installed and the three measurements are actually recorded. If any is missing, Chapter 3.1 has nothing to build the policy around.

```bat
cd C:\Users\[YOUR_USER]\IsaacLab
isaaclab.bat -p C:\Users\[YOUR_USER]\IsaacLab\check_setup.py --headless --skip-training --hardware
```

Section 11 of the report should now show `djitellopy` and `opencv-python` installed, and all three measurements present in `project_notes.txt`. Those three rows are what Chapter 3 depends on — if any is missing, Chapter 3.1 has nothing to build the policy around.

> ✅ **Checkpoint 1.4**
> 1. Drone's WiFi has a password you set; Windows profile is Public; single adapter
> 2. Took off and landed under program control; `emergency()` tested
> 3. Three measurements recorded in `project_notes.txt`
> 4. `check_setup.py --hardware` shows Section 11 fully green
> 5. You know which telemetry the drone cannot provide, and why that matters for Chapter 3


---

# Chapter 2 — The Arena: Two Drones, One Scene

## 2.1 Add the attacker to the scene (≤1.5h)

> **What / Why / How it contributes:** We put a second Crazyflie into every cloned environment. The defender stays a physics-driven articulation (RL will fly it); the attacker becomes a scene actor whose position WE control. This subchapter is pure scene-building — no rewards or motion yet. It matters because everything later (chasing, seeing, detecting) needs two drones reliably spawning in all 2048 parallel envs without physics explosions.

### Concept first: two robots in the Direct workflow

You know the pattern for one robot: an `ArticulationCfg` in the env cfg, instantiated in `_setup_scene`, registered in `self.scene.articulations`. Two robots = literally the same pattern twice, with two different prim paths under each env namespace:

```
/World/envs/env_0/
   ├── Robot     ← defender (physics + external forces from policy)
   └── Attacker  ← attacker (we write its pose every step)
/World/envs/env_1/
   ├── Robot
   └── Attacker
   ...
```

The cloning system (`clone_environments`) replicates *everything* under `env_.*`, so the attacker rides along for free — the same mechanism that replicates goal markers and other scene objects.

### Step 1 — Extend the env cfg

> **Environment:** none needed — you are editing files.

Adds a second Crazyflie to the configuration — the attacker — plus the pursuit settings (capture distance, arena size, attacker speed). One line reuses the same spawn recipe as the defender, with a different address in the scene.

In your task file, add a second articulation config next to the existing `robot` one. The Crazyflie asset config is reused; only the prim path and spawn position change:

```python
# quadcopter_env.py — the config class near the top of the file
from isaaclab_assets import CRAZYFLIE_CFG          # the spawn recipe you already use
from isaaclab.assets import ArticulationCfg

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # ... keep everything from the quadcopter cfg (sim, scene, action_space=4, etc.)

    # defender — unchanged from the hover task
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # NEW: attacker — same drone, different prim path, spawned 4 m away at 1.5 m altitude
    attacker: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Attacker",
        init_state=ArticulationCfg.InitialStateCfg(pos=(4.0, 0.0, 1.5)),
    )

    # NEW: pursuit geometry knobs (plain attributes, like the reward scales you know)
    capture_radius = 0.35        # meters — "caught" if closer than this
    arena_radius = 8.0           # meters — episode fails if defender strays this far
    attacker_speed = 0.6         # m/s along its path (we'll tune this in Ch. 3)
```

**Why 0.35 m?** A Crazyflie is ~9 cm rotor-to-rotor. 0.35 m means the two airframes are roughly overlapping. Requiring actual mesh contact would mean the capture bonus is almost never triggered during early training.

Also bump `env_spacing` in the scene cfg to at least `2 * arena_radius` (e.g. 16.0) so neighboring envs' drones never visually overlap into each other's future camera views.

### Step 2 — Instantiate it in `_setup_scene`

> **Environment:** none needed — you are editing files.

Creates the attacker object and registers it with the scene. Registration is what keeps its position readings refreshed each step; without it the drone appears on screen but its data never updates.

```python
def _setup_scene(self):
    self._robot = Articulation(self.cfg.robot)
    self._attacker = Articulation(self.cfg.attacker)               # NEW
    self.scene.articulations["robot"] = self._robot
    self.scene.articulations["attacker"] = self._attacker          # NEW
    # ... rest unchanged: terrain/ground, clone_environments, lights
```

### Step 3 — Freeze the attacker (for now)

> **Environment:** none needed — you are editing files.

Pins the attacker in place by writing its pose every step, so it does not simply fall. This is temporary scaffolding — 2.2 replaces the fixed pose with a moving path — but it lets you verify the scene before adding motion.

The attacker is a physics object, so with no controller it will simply fall. Until 2.2 gives it a scripted path, pin it in place by re-writing its root state every step. Add to `_apply_action` (or a small helper called from it):

```python
def _apply_action(self):
    # defender: unchanged (thrust + moments)
    self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
    # attacker: hold pose (temporary — replaced by trajectory in 2.2)
    hold = self._attacker.data.default_root_state.clone()
    hold[:, :3] += self.scene.env_origins            # local spawn pos → world coords
    self._attacker.write_root_pose_to_sim(hold[:, :7])
    self._attacker.write_root_velocity_to_sim(torch.zeros_like(hold[:, 7:]))
```

**Flag — this is different from what you know:** for the defender we apply *forces* and let PhysX integrate motion (dynamic control). For the attacker we *write the pose directly* every step (kinematic control — from Greek *kinema*, "motion": describing motion without the forces causing it). Teleporting a body each step means PhysX doesn't simulate it falling — which is exactly what we want for a scripted actor. The trade-off: a kinematic attacker won't get knocked around on contact. Fine for us — "capture" is a distance check, not a physical collision.

### Step 4 — Reset logic

> **Environment:** none needed — you are editing files.

Returns the attacker to its start position whenever an episode restarts, mirroring what already happens for the defender. Without it, the attacker would stay wherever the last episode left it.

In `_reset_idx`, reset the attacker alongside the defender (mirror the existing robot-reset lines):

```python
def _reset_idx(self, env_ids):
    # ... existing defender reset ...
    a_state = self._attacker.data.default_root_state[env_ids].clone()
    a_state[:, :3] += self.scene.env_origins[env_ids]
    self._attacker.write_root_pose_to_sim(a_state[:, :7], env_ids)
    self._attacker.write_root_velocity_to_sim(a_state[:, 7:], env_ids)
```

### Step 5 — Look at it

> **Environment:** `env_drone`

Runs a few iterations with the viewport open, so you can see both drones in the scene. This is the checkpoint for Chapter 2.1: two drones spawning correctly across every parallel environment, before any rewards or motion exist.

Don't train — just *render* the scene with random actions using the play script's `--headless`-off mode on an untrained checkpoint, or quickest: run training for 5 iterations **without** `--headless`:

*Run from:* `any folder`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\train.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 16
```

You should see 16 arenas, each with a jittering defender (random policy) and an attacker hovering frozen at (4, 0, 1.5).

> ✅ **Checkpoint 2.1**
> 1. No spawn errors across 16 envs
> 2. Attacker visibly present and motionless at its spawn point in every env
> 3. Defender still jitters under the (untrained) policy — proving you didn't break its force pipeline

---

## 2.2 Give the attacker a life: scripted evasive motion (≤1.5h)

> **What / Why / How it contributes:** A frozen target teaches a policy to fly to a POINT — that's just the hover task with extra steps. A moving target forces the policy to learn interception. We give the attacker a parametric 3D path (circle + vertical bobbing) with per-env randomized phase and direction, so every parallel env presents a different chase. This is also our difficulty dial: one number (attacker_speed) takes us from "training wheels" to "genuinely hard."

### Concept: why a parametric path (and not attacker RL)?

Three candidate attackers (the thing being chased), in order of complexity:
1. **Parametric path** (sin/cos waypoint loop) — deterministic, tunable, zero training. ← **us**
2. Reactive script (flee from defender) — better, but can create degenerate loops early in training.
3. RL attacker (adversarial / MARL) — a research-scale problem. Chen, Yu et al. (Tsinghua, arXiv:2409.15866) do this with MAPPO and curriculum generation. See Chapter 6.3 for how to get there.

A useful rule: **change one hard thing at a time.** The hard thing in Chapter 3 is the pursuit reward. So the attacker stays simple.

### Step 1 — The trajectory math (visual, no trig anxiety)

> **Environment:** none needed — this step is explanation only.

Explains the path the attacker will fly — a circle with a vertical bob — and the two lines of arithmetic that produce it. Its speed becomes your difficulty dial in Chapter 3.3.

Picture a point moving around a circle of radius `R` while gently bobbing up and down:

```
      TOP VIEW                        SIDE VIEW
   .──────────.                    z
  /            \                 1.9 ┈╭─╮┈┈┈┈╭─╮┈    ← bobbing ±0.4 m
 │      ●──────│──→ x           1.5 ─┤  ╰────╯  ├─
  \     center /                 1.1 ┈┈┈┈┈┈┈┈┈┈┈┈
   '──────────'
   R = 3 m circle                horizontal loop + vertical wave
```

`sin` and `cos` are just "the x and y coordinates of a point walking around a circle" — that's all the math we need:

```python
# angle grows over time → point moves around the circle
theta = phase + direction * (attacker_speed / R) * t     # radians
x = R * cos(theta);  y = R * sin(theta)                  # the circle
z = 1.5 + 0.4 * sin(0.7 * t + phase)                     # the bob
```

`attacker_speed / R` converts "meters per second along the path" into "radians per second," so the speed knob means what it says.

### Step 2 — Implement it

> **Environment:** none needed — you are editing files.

Replaces the frozen pose from 2.1 with the moving path, and randomises where each attacker starts and which way it circles. Randomising stops the policy memorising one specific chase instead of learning to intercept movement.

In `__init__`, allocate per-env randomization buffers:

```python
self._atk_phase = torch.zeros(self.num_envs, device=self.device)
self._atk_dir = torch.ones(self.num_envs, device=self.device)     # +1 or -1 (CW/CCW)
self._atk_t = torch.zeros(self.num_envs, device=self.device)      # per-env clock
self._atk_radius = 3.0
```

Replace the "hold pose" block from 2.1 with the trajectory (called every `_apply_action`, which runs at physics rate — advance the clock by the physics dt):

```python
def _move_attacker(self):
    self._atk_t += self.physics_dt
    w = self.cfg.attacker_speed / self._atk_radius
    theta = self._atk_phase + self._atk_dir * w * self._atk_t
    pos = torch.zeros(self.num_envs, 3, device=self.device)
    pos[:, 0] = self._atk_radius * torch.cos(theta)
    pos[:, 1] = self._atk_radius * torch.sin(theta)
    pos[:, 2] = 1.5 + 0.4 * torch.sin(0.7 * self._atk_t + self._atk_phase)
    pos += self.scene.env_origins                                   # local → world

    pose = self._attacker.data.root_pose_w.clone()
    pose[:, :3] = pos                                               # keep orientation as-is
    self._attacker.write_root_pose_to_sim(pose)

    # also store the attacker's velocity (finite difference) — Chapter 3 puts it in the obs
    if not hasattr(self, "_atk_prev_pos"):
        self._atk_prev_pos = pos.clone()
    self._atk_vel = (pos - self._atk_prev_pos) / self.physics_dt
    self._atk_prev_pos = pos.clone()
```

Everything is a **batched torch tensor across all envs** — no Python loops. The same vectorisation applies to every reward and observation function in this project.

In `_reset_idx`, randomize the chase each episode (this is domain randomization for *behavior*, the same philosophy as your visual domain randomization in SDG):

```python
n = len(env_ids)
self._atk_phase[env_ids] = torch.rand(n, device=self.device) * 2 * math.pi
self._atk_dir[env_ids] = torch.where(torch.rand(n, device=self.device) > 0.5, 1.0, -1.0)
self._atk_t[env_ids] = 0.0
```

Why randomize phase and direction? If every env's attacker started at the same spot going the same way, the policy could memorize "always bank left at t=3 s" instead of learning "intercept whatever moves." Randomization forces the general skill — exactly why you randomized lighting and textures for the pallet jack.

### Step 3 — Watch the attackers move

> **Environment:** `env_drone`

Renders the scene again to confirm the attackers orbit smoothly and differently in each environment. Smooth motion matters because the camera readings in Chapter 3 are computed from it.

Run 16 envs without `--headless` again. Now the attackers sweep circles at different phases/directions while defenders jitter randomly.

> ✅ **Checkpoint 2.2**
> 1. Attackers orbit smoothly (no teleport-stutter — if stuttery, confirm `_move_attacker` runs every physics step, not every env step)
> 2. Different envs show different phases and directions
> 3. After a forced reset (let episodes time out), trajectories re-randomize
> 4. Commit to git: "arena complete"

---

# Chapter 3 — Closing the Distance: The Pursuit Policy

## 3.1 What the defender commands, and what it senses (≤1.5h)

> **What / Why / How it contributes:** This defines the two interfaces between the policy and the world — the four numbers it outputs and the seventeen it receives — using the measurements you took in 1.4. Every choice here is dictated by what a Tello can actually do and report. Getting them right now means the policy you train in 3.3 is the policy you fly in Chapter 7, with no second training run.

### Part A — The four outputs: stick commands

> **Environment:** none needed — Parts A to G are all file edits. You run nothing until the sanity check at the end of 3.1.

Isaac Lab's hover task outputs one thrust and three torques. **A Tello does not accept forces.** It accepts four normalised channels and runs its own stabiliser underneath:

```
   channel 0   forward / backward
   channel 1   left / right
   channel 2   up / down
   channel 3   turn (yaw rate)
```

That built-in stabiliser is a large amount of balancing work you no longer have to learn — and a large amount of behaviour the simulation must now imitate.

Replace the thrust-and-torque conversion in `_pre_physics_step` with one that treats the actions as **desired velocities** and applies whatever force reaches them:

```python
def _pre_physics_step(self, actions):
    self._actions = actions.clone().clamp(-1.0, 1.0)

    desired_vel_b = self._actions[:, :3] * self.cfg.max_speed        # m/s
    desired_yaw_rate = self._actions[:, 3] * self.cfg.max_yaw_rate   # rad/s

    # stands in for the Tello's own stabiliser
    vel_error = desired_vel_b - self._robot.data.root_lin_vel_b
    force_b = self.cfg.vel_gain * vel_error * self._robot_mass
    force_b[:, 2] += self._robot_weight                              # hold altitude

    self._thrust[:, 0, :] = quat_apply(self._robot.data.root_quat_w, force_b)
    yaw_error = desired_yaw_rate - self._robot.data.root_ang_vel_b[:, 2]
    self._moment[:, 0, 2] = self.cfg.yaw_gain * yaw_error
```

```python
max_speed = 2.0          # m/s — conservative; a Tello can do more
max_yaw_rate = 1.5       # rad/s
vel_gain = 3.0           # how hard the stand-in stabiliser corrects
yaw_gain = 0.05
```

**Why approximate rather than model the real stabiliser.** The Tello's control law is undocumented. What must match is the *interface* — four normalised numbers in, roughly velocity-like behaviour out. Section 3.3's randomisation covers the gap between this approximation and reality, and 7.4 corrects it from real flight logs.

**After this change the simulated airframe stops being a Crazyflie** in any meaningful sense. It is a generic hovering body that responds to velocity commands, which is what a Tello is from your code's point of view.

### Part B — The control rate matches your measurement

From 1.4, Step 6. If you measured 20 commands per second:

```python
# sim.dt = 1/100, so decimation 5 gives 20 Hz control
decimation = 5
```

**This cannot be skipped.** At 20 Hz each command persists two and a half times longer than at 50 Hz, so identical policy output produces much larger movement. A policy trained at 50 Hz and flown at 20 Hz overshoots consistently.

### Part C — The seventeen inputs

Defines everything the policy is allowed to know: its own motion, the seven camera readings, and the command it issued last step. Every entry is something a Tello can actually supply, which is what makes the same list usable in Chapter 7 without retraining.

```
obs (17 numbers per env):
 [0:3]   defender linear velocity (body frame)   ← Tello: get_speed_x/y/z, cm/s → m/s
 [3:6]   gravity direction        (body frame)   ← Tello: computed from roll and pitch
 [6]     bearing_x     how far left/right of centre the attacker appears, −1 … +1  ─┐
 [7]     bearing_y     how far above/below centre,                        −1 … +1   │
 [8]     ang_size      share of the frame width it fills,                   0 … 1   ├ DELAYED
 [9]     d_bearing_x   change in bearing_x since last step                          │
 [10]    d_bearing_y   change in bearing_y since last step                          │
 [11]    d_ang_size    change in ang_size   ← the "am I closing?" signal             │
 [12]    visible       1.0 if the detector found it this frame, else 0.0            ─┘
 [13:17] previous action — the four commands issued last step
```

Set `observation_space = 17`.

**Three things are absent, each for a reason from 1.4:**

**No angular rates.** A Tello reports attitude angles but not how fast they are changing. Training on a number the drone cannot supply would produce a policy that fails on hardware in a way you could not diagnose. The policy can fly without them: rotating the drone sweeps the attacker across the frame, so the camera readings already reveal rotation.

**No positions in metres, anywhere.** Recovering metres from a photograph requires knowing the attacker's true width. Bearings and frame-share make no such claim, so a different target drone changes nothing.

**No instantaneous readings.** They arrive late, on purpose — see Part D.

### Part D — The readings arrive late, and the policy is told its last command

**The delay.** A Tello's video reaches your laptop 99–219 ms after it was captured. At 20 Hz control that is 2 to 5 decisions of staleness. A policy trained on instant readings has no reason to account for this, and on hardware it oscillates: it steers toward where the target was, finds it has moved, over-corrects, repeats.

Training with the delay present teaches the policy to **lead** the target rather than track it.

```python
# from project_notes.txt — YOUR measurement, converted to control steps
obs_delay_min = 2
obs_delay_max = 5
```

In `__init__`:

```python
self._reading_history = torch.zeros(
    self.num_envs, self.cfg.obs_delay_max + 1, 7, device=self.device
)
self._delay_steps = torch.randint(
    self.cfg.obs_delay_min, self.cfg.obs_delay_max + 1,
    (self.num_envs,), device=self.device
)
```

Each step, push fresh readings in and take delayed ones out:

```python
def _delayed_readings(self, fresh):          # fresh: (num_envs, 7)
    self._reading_history = torch.roll(self._reading_history, shifts=1, dims=1)
    self._reading_history[:, 0] = fresh
    idx = self._delay_steps.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 7)
    return self._reading_history.gather(1, idx).squeeze(1)
```

Re-randomise `_delay_steps` for reset environments in `_reset_idx`, and zero their history.

**Randomising rather than fixing the delay is the point.** A constant lag can be cancelled exactly; a Tello's varies by roughly ±37 ms within a single flight. Training across the range produces tolerance rather than a policy tuned to a number it will never see twice.

**The previous action.** With delayed readings, some commands have been issued but are not yet visible in anything the policy can see. Feeding back what it just commanded lets it reason about those instead of re-issuing them. Four numbers, standard practice for delayed control.

### Part E — The camera model must match the Tello's lens

Sets the simulated camera's focal length and frame shape to match the real drone's. Bearings are measured as a fraction of the frame, so a different field of view makes the same reading mean a different angle — silently, with no error.

```python
# camera model — MUST match both the TiledCameraCfg and the real drone's lens
cam_width = 640
cam_height = 480                  # 4:3, matching the Tello's 960x720
cam_focal_mm = 12.0               # gives ~83 deg horizontal field of view
cam_aperture_mm = 20.955          # PinholeCameraCfg default horizontal aperture
attacker_span_m = 0.13            # the target's real width — used ONLY to simulate the camera
```

**Why 12 mm and not Isaac Lab's default 24 mm.** Bearings are *normalised* across the frame: +0.5 means "halfway from centre to the right edge." How many degrees that represents depends entirely on the lens. Isaac Lab's default gives roughly 47° of horizontal view; a Tello sees about 83°. Train on 47° and fly on 83° and every bearing means nearly twice the angle it did in training — the defender would under-steer on every correction, silently.

For a different camera, solve:

```
focal_mm = aperture_mm / (2 x tan(FOV / 2))
         = 20.955 / (2 x tan(41.3 deg)) = 12.0     for an 83 deg lens
```

**Aspect ratio too.** A Tello outputs 4:3. Rendering square and then squashing a 4:3 photograph into it would distort every bearing. Render 640×480, resize real frames to 640×480, and the geometry stays honest.

**This is the most likely silent failure in the whole project**, because nothing errors — the numbers quietly mean something else.

### Part F — Calculating the readings during training

During training we compute the seven readings from the true relative position — going *forward* through the camera model: given where the attacker really is, how large would it appear?

This is legitimate. The simulated camera really is photographing a target of known dimensions, and rendering the frame would produce this same rectangle, only far more slowly. What we never do is the *backward* step — taking a rectangle and dividing by an assumed size to produce metres. That step appears nowhere in this project.

```python
from isaaclab.utils.math import subtract_frame_transforms

def _camera_readings(self):
    """Project ground truth through the camera model → what the detector would report."""
    rel_b, _ = subtract_frame_transforms(
        self._robot.data.root_pos_w, self._robot.data.root_quat_w,
        self._attacker.data.root_pos_w,
    )
    fwd = rel_b[:, 0]
    safe_fwd = fwd.clamp(min=0.05)

    f_px = self.cfg.cam_width * self.cfg.cam_focal_mm / self.cfg.cam_aperture_mm
    half_w = self.cfg.cam_width / 2.0
    half_h = self.cfg.cam_height / 2.0

    bearing_x = -(rel_b[:, 1] / safe_fwd) * (f_px / half_w)
    bearing_y = -(rel_b[:, 2] / safe_fwd) * (f_px / half_h)
    ang_size = (f_px * self.cfg.attacker_span_m / self._dist.clamp(min=0.05)) / self.cfg.cam_width

    visible = (fwd > 0.05) & (bearing_x.abs() < 1.0) & (bearing_y.abs() < 1.0) & (ang_size > 0.012)
    return bearing_x, bearing_y, ang_size, visible.float()
```

The `ang_size > 0.012` threshold is a box about 8 pixels wide, below which your Chapter 5 detector will usually return nothing. Simulating that blind spot means the policy meets the "too small to detect" case thousands of times during training and learns a response, instead of encountering it first on a real flight.

### Part G — Assembling the observation

Puts the seventeen numbers together in the order the policy expects, holding the last reading whenever the attacker is not visible. Chapter 7.2's flight script assembles the same seventeen in the same order from real hardware.

```python
def _get_observations(self) -> dict:
    bx, by, asz, vis = self._camera_readings()

    # hold the previous reading wherever the attacker is not currently visible
    bx  = torch.where(vis > 0.5, bx,  self._prev_bx)
    by  = torch.where(vis > 0.5, by,  self._prev_by)
    asz = torch.where(vis > 0.5, asz, self._prev_asz)

    d_bx, d_by, d_asz = bx - self._prev_bx, by - self._prev_by, asz - self._prev_asz
    self._prev_bx, self._prev_by, self._prev_asz = bx.clone(), by.clone(), asz.clone()

    fresh = torch.stack([bx, by, asz, d_bx, d_by, d_asz, vis], dim=-1)
    delayed = self._delayed_readings(fresh)

    obs = torch.cat([
        self._robot.data.root_lin_vel_b,          # 3
        self._robot.data.projected_gravity_b,     # 3
        delayed,                                  # 7
        self._actions,                            # 4
    ], dim=-1)
    return {"policy": obs}
```

Allocate `_prev_bx / _prev_by / _prev_asz` in `__init__`, and zero them for reset environments — otherwise a fresh episode inherits the previous one's last sighting and its first `d_ang_size` is nonsense.

Also cache the true distance; the reward and the termination check both need it:

```python
self._dist = torch.linalg.norm(
    self._attacker.data.root_pos_w - self._robot.data.root_pos_w, dim=1
)
```

### Sanity check before moving on

Run a handful of non-headless iterations printing `bearing_x`, `ang_size`, `_dist` and `visible` for env 0. Three things must hold: **ang_size rises as `_dist` falls**, **bearing_x flips sign** as the attacker crosses the frame, and **the delayed readings lag the fresh ones** by the expected number of steps.

> ✅ **Checkpoint 3.1**
> 1. Env steps with no shape errors at `observation_space = 17` and `decimation` matching your measured rate
> 2. Printed readings behave sensibly, and delayed values visibly lag
> 3. `visible` drops to 0 when the attacker leaves the frame, with held values staying frozen
> 4. Your cfg contains the three numbers from `project_notes.txt`, not the defaults printed here


---

## 3.2 Reward design: teaching "get closer" (≤1.5h)

> **What / Why / How it contributes:** The reward function is the entire curriculum — the policy becomes whatever the reward pays for, including its loopholes. We build a dense reward from three ingredients (closing speed, proximity, capture bonus) plus stability penalties inherited from the hover task, and we define the episode-ending events. This subchapter is mostly THINKING, deliberately: reward bugs cost you full training runs, the most expensive kind of bug in this project.

### First, the rule that makes this section legal

The observations changed in 3.1, but **the reward does not have to**. The reward is read only by the training algorithm, which uses it to adjust the network's weights. Once training ends it is never called again — `play.py` and your Chapter 6 demo never evaluate it. So it may freely use the true distance between the drones, even though the policy itself never receives that number.

```
 OBSERVATIONS ──► must be obtainable from a camera at deployment  (strict)
 REWARDS      ──► may use anything the simulator knows            (free)
```

Keeping the reward metric is not a compromise — it is what lets us keep a dense, smooth, easy-to-tune learning signal while the policy learns to act on camera readings alone.

### The reward, ingredient by ingredient

```python
# in the cfg — the tuning dials
closing_reward_scale = 2.0      # per m/s of speed TOWARD the target
proximity_reward_scale = 1.5    # smooth "warmth" signal as distance shrinks
capture_bonus = 200.0           # jackpot, once, at capture
crash_penalty = -50.0           # hit the floor / left the arena
action_rate_penalty = -0.02     # sim-to-real: penalise jerky command changes
# keep the hover task's small lin_vel / ang_vel penalties (they fight jitter)
```

```python
def _get_rewards(self) -> torch.Tensor:
    to_target = self._attacker.data.root_pos_w - self._robot.data.root_pos_w
    dir_to_target = to_target / self._dist.unsqueeze(1).clamp(min=1e-6)

    # 1) CLOSING SPEED: my velocity, projected onto the target direction.
    #    +1.0 means "approaching at 1 m/s"; negative means fleeing. Paid EVERY step.
    closing = (self._robot.data.root_lin_vel_w * dir_to_target).sum(dim=1)

    # 2) PROXIMITY: 1 - tanh(dist/4) — a smooth 0..1 "warmth" that rises as you approach.
    proximity = 1.0 - torch.tanh(self._dist / 4.0)

    # 3) CAPTURE: the jackpot
    captured = self._dist < self.cfg.capture_radius

    # 4) SMOOTHNESS: penalise how much the command changed since last step
    action_rate = torch.sum(torch.square(self._actions - self._prev_actions), dim=1)

    reward = (
        self.cfg.closing_reward_scale * closing
        + self.cfg.proximity_reward_scale * proximity
        + self.cfg.capture_bonus * captured.float()
        + self.cfg.action_rate_penalty * action_rate
    ) * self.step_dt
    self._prev_actions = self._actions.clone()
    return reward
```

(Scaling by `step_dt`, like the built-in tasks do, keeps reward magnitudes comparable if you ever change the control frequency.)

**Why three ingredients instead of just the capture bonus?** With the bonus alone, an untrained policy applying random thrust would have to stumble within 0.35 m of a moving attacker before receiving anything other than zero. Across thousands of episodes that may never happen, and PPO cannot improve a policy whose returns are identical everywhere — this is a **sparse reward**. `closing` and `proximity` pay out on every single step, so even a bad policy learns which direction is better. The capture bonus then supplies the final push from "nearby" to "in contact."

**Why `tanh` for proximity?** It squashes distance into a bounded 0–1 curve: steep payoff gains near the target, flat far away. Unbounded `1/dist` rewards explode at tiny distances and destabilize PPO. `tanh` (hyperbolic tangent, the S-curve used inside neural networks) is the standard bounded squash; the hover task uses the same trick for its position reward.

**Why penalise the action rate?** This one exists for Chapter 7. In simulation, a policy can change its command wildly between steps at no cost — motors respond instantly. Real motors have inertia and real radio links drop packets, so a jittery command stream produces a drone that shakes rather than flies. Published sim-to-real work identifies command smoothness as one of the decisive factors in whether a policy transfers at all. Keep the weight small: too large and the defender becomes sluggish and stops manoeuvring.

**Reward-hacking preview** (you know this failure mode from Ant): watch for the policy learning to *orbit* the attacker at ~1 m — proximity pays well there, closing averages zero, and capture never happens. Fix if you see it: raise `closing_reward_scale` or shrink the `4.0` in the tanh to steepen the near-field gradient.

### Episode endings (`_get_dones`)

```python
def _get_dones(self):
    captured = self._dist < self.cfg.capture_radius                       # success
    crashed = self._robot.data.root_pos_w[:, 2] < 0.1                     # floor
    escaped = self._dist > self.cfg.arena_radius                          # lost the plot
    died = crashed | escaped | captured        # all three END the episode now
    time_out = self.episode_length_buf >= self.max_episode_length - 1
    return died, time_out
```

Ending the episode **on capture** matters: if the episode continued, the drone would sit inside the capture radius farming bonus — a reward exploit you'd only discover after a confusing training run.

> ✅ **Checkpoint 3.2** — code compiles & steps; and you can answer: *"If I set closing_reward_scale to 0, what degenerate behavior might appear?"* (Answer: hovering at the tanh sweet spot — proximity pays without ever closing.)

---

## 3.3 Randomise, train, diagnose (≤1.5h hands-on + background compute)

> **What / Why / How it contributes:** We launch the real training run, learn which TensorBoard curves diagnose a pursuit task specifically, and apply a simple curriculum: train against a slow attacker first, then raise its speed. Deliverable: a checkpoint where the defender reliably intercepts the moving attacker using ground-truth observations — the "flight brain" that Chapter 6 will connect to synthetic eyes.

### Step 0 — Randomise what you cannot know about the drone

> **Environment:** none needed — you are editing files.

Varies mass, thrust and drift between attempts, so each one trains against a slightly different drone. You cannot measure these precisely on a cheap drone, and a policy that works across a range will work on the actual one.

Before training, randomise the physical properties you cannot measure precisely. In `_reset_idx`, so each attempt trains against a slightly different drone:

```python
n = len(env_ids)
# mass varies with battery charge and wear
self._mass_scale[env_ids] = 1.0 + (torch.rand(n, device=self.device) - 0.5) * 0.2
# available thrust falls as the battery drains
self._thrust_scale[env_ids] = 0.85 + torch.rand(n, device=self.device) * 0.3
# cheap drones drift — a small constant push in a random direction
self._drift[env_ids] = (torch.rand(n, 3, device=self.device) - 0.5) * 0.15
```

Also add small Gaussian noise to the camera readings before they enter the delay buffer. The detector's rectangle jitters by a few pixels between frames, and a policy that never saw noisy bearings will chase the jitter.

**Randomisation matters more than accuracy here.** You will never model a C$115 drone correctly. A policy that works across a wide band of possible drones will work on the actual one; a policy tuned to your best guess fails wherever that guess was wrong.

**A note on wind.** This models drift as a small *constant* push, which represents a drone's imperfect trim well but wind poorly — real wind is gusty and changes direction. This tutorial assumes calm conditions, and that assumption is load-bearing: on a breezy day the disturbance would exceed anything the policy trained against, and the honest response is to wait for better weather. Flying in wind would mean making `_drift` vary *during* an episode — a harder task needing its own training run.

**Expect a lower capture rate than an unrandomised run, and welcome it.** You have made the task harder in exactly the ways reality is harder. A policy capturing 60–70% under randomisation is far more likely to fly than one capturing 95% under ideal conditions, because the second was never solving the real problem.

### Step 1 — Curriculum, the manual way

> **Environment:** `env_drone`

Starts training against a slow attacker. Beginning easy matters because at full speed an untrained policy almost never makes contact, so the capture bonus is never collected and training settles for hovering.

Start easy: in the cfg set `attacker_speed = 0.3`. Then:

*Run from:* `any folder`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\train.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 2048 --headless --max_iterations 1500
tensorboard --logdir C:\projects\drone_pursuit\drone_pursuit\logs\skrl
```

Expect roughly 30–90 min depending on GPU. This is a great moment to start Chapter 4 in a second terminal — it's fully independent.

### Step 2 — Which metrics matter for a pursuit task

> **Environment:** none needed — you are editing files.

Adds pursuit-specific numbers to your TensorBoard logging and explains which pattern means what. Total reward alone can climb while the defender never actually catches anything.

Beyond the usual `Total reward` climb, add these to `self.extras["log"]` in `_get_dones`/`_get_rewards` (same logging pattern as the built-in tasks) and watch:

| Metric | Healthy | Sick pattern → diagnosis |
|---|---|---|
| **capture rate** (fraction of episodes ending in capture) | climbs past 50–80% | stuck at 0% while reward climbs → orbiting exploit (see 3.2) |
| **mean final distance** | falls toward capture_radius | plateaus ≈ some fixed radius → orbiting again, or attacker simply faster than max defender speed |
| **episode length** | *falls* as captures come sooner | pinned at max → nobody's catching anybody |
| **crash rate** | < 10% after early phase | high forever → stability penalties too weak vs. closing reward (kamikaze diving) |
| **lost-sight fraction** (mean of `visible`) | rises toward ~0.9 as the policy learns to keep the attacker in frame | falling → the defender is flying blind and doesn't realise it matters |

### Step 3 — Raise the difficulty

> **Environment:** `env_drone`

Once capture rate > ~80% at speed 0.3: stop, set `attacker_speed = 0.6`, and **resume** from the checkpoint (`--checkpoint C:\projects\drone_pursuit\drone_pursuit\logs\skrl\<run-folder>\checkpoints\best_agent.pt` on the train script) rather than restarting. Repeat toward 1.0 m/s if you're ambitious. This staged-difficulty idea is curriculum learning in its simplest form — the adaptive-environment-generator in the Tsinghua paper is the industrial-strength version of this same instinct.

### Step 4 — Harvest the capture calibration (5 minutes, saves you an hour later)

> **Environment:** `env_drone`

Measures what share of the frame the attacker fills at the moment of capture. Chapter 6 has no simulator to ask for distance, so it declares victory using this number instead — and right now is the only time both quantities are available together.

Chapter 6 needs to declare victory using the camera alone, which means knowing **what share of the frame the attacker occupies at the moment of capture**. You have both quantities right now, so measure it while you can.

Add two lines to your logging so that on every step where `_dist` first drops below `capture_radius`, you record the corresponding `ang_size`. Run `play.py` for a few dozen episodes and look at the distribution.

You will get a spread rather than a single value — the attacker's rectangle is wider seen face-on than edge-on. Pick from that spread according to the error you'd rather make: the **low end** declares capture eagerly and occasionally claims one it didn't earn; the **median** is the balanced choice; the **high end** only ever confirms certain captures but silently misses some real ones. Write your choice into the cfg as `capture_ang_size`. Keep a `project_notes.txt` in the project folder for values like this one that you derive by measurement rather than install — the lockfile cannot capture them.

You are not choosing this number freely — you chose `capture_radius` in metres back in 2.1, and the camera geometry determines what that looks like in pixels. You are simply going and reading off the answer.

### Step 5 — Watch the trained chase

> **Environment:** `env_drone`

Plays the trained policy so you can see whether it intercepts rather than tail-chases. Corner-cutting toward where the attacker is going is the visible sign that the rate-of-change readings are doing their job.

*Run from:* `any folder`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\play.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 16
```

Watch for the *lead*: a well-trained defender cuts the corner toward where the attacker is *going*. That's the target-velocity observation earning its keep.

> ✅ **Checkpoint 3.3 — MILESTONE: Problem 1 (control) SOLVED**
> 1. Capture rate > 80% at attacker_speed ≥ 0.6
> 2. Play shows visible interception (corner-cutting), not just tail-chasing
> 3. Best checkpoint path recorded in `project_notes.txt` — Chapter 6 needs it
> 4. `capture_ang_size` measured and written into the cfg

---

# Chapter 4 — Synthetic Data: Manufacturing a Drone-Photo Factory

## 4.1 The SDG scene: camera + semantic tags (≤1.5h)

> **What / Why / How it contributes:** We build a standalone Replicator data-generation script. A Crazyflie gets a semantic tag ("drone"), a Replicator camera photographs it, and the bounding_box_2d_tight annotator auto-labels every frame. Why standalone instead of inside the RL env? Data generation and RL training have different needs (pretty rendering vs. speed); separating them lets each be simple. Output: the machine that Chapter 4.2 will crank.

### Concept: why synthetic images come pre-labelled

In the real world, someone draws boxes around drones in thousands of photos by hand. In sim, the renderer *already knows* every pixel's identity — annotation is free and pixel-perfect. Your job reduces to: (1) tell Replicator which prims mean "drone" (**semantic tags**), (2) point cameras from varied poses, (3) randomize everything else so the detector learns "drone-ness," not "this exact scene."

### Step 1 — The script skeleton

> **Environment:** `env_drone` (the `mkdir` needs no environment; the script does)

Builds the standalone script that photographs a drone and labels it automatically. It runs outside the RL environment because data generation wants pretty rendering while training wants speed, and Chapter 5 trains the detector on its output.

Create the folder for data-generation scripts, and the folder its output will go to:

```bat
cd C:\projects\drone_pursuit\drone_pursuit
mkdir C:\projects\drone_pursuit\drone_pursuit\scripts\sdg C:\projects\drone_pursuit\drone_pursuit\data\raw
```

Then create `C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\generate_drone_data.py`. It uses the standard `AppLauncher` pattern for standalone Isaac Lab scripts:

```python
"""Standalone SDG: labeled images of a Crazyflie for detector training."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_frames", type=int, default=200)
parser.add_argument("--out_dir", type=str, default=r"C:\projects\drone_pursuit\drone_pursuit\data\raw")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True                       # cameras need the render pipeline
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything below runs inside the sim app ----
import omni.replicator.core as rep
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))

# ground + two lights (both get randomized in 4.2)
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

# ambient fill — stands in for skylight bouncing around
dome_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
dome_cfg.func("/World/Light", dome_cfg)

# the sun — a single strong source coming from one direction
sun_cfg = sim_utils.DistantLightCfg(intensity=3000.0, angle=0.53)
sun_cfg.func("/World/Sun", sun_cfg)

# the subject of every photo: a Crazyflie, tagged with its class
drone_cfg = sim_utils.UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",
    semantic_tags=[("class", "drone")],          # ← this line is what produces the labels
)
drone_cfg.func("/World/Drone", drone_cfg, translation=(0.0, 0.0, 1.5))
```

Note the asset path: Isaac Sim 5.x moved it to `Robots/Bitcraze/Crazyflie/cf2x.usd` (older tutorials say `Robots/Crazyflie/` — a rename documented in the Isaac Lab release notes; good example of why we pin doc versions).

### Step 2 — Replicator camera + writer

> **Environment:** none needed — you are editing a file.

Adds the camera and the writer that saves each frame with its bounding box. The camera settings match the Tello's lens exactly, for the same reason Chapter 3.1 gives: a different field of view changes what every reading means.

```python
# a Replicator camera and a render product (the "film" it exposes onto)
camera = rep.create.camera(focal_length=12.0)   # ~83 deg FOV, matching a Tello
render_product = rep.create.render_product(camera, (640, 480))   # 4:3, matching the real drone

# BasicWriter: saves RGB + tight 2D boxes for every captured frame
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir=args.out_dir,
    rgb=True,
    bounding_box_2d_tight=True,      # "tight" = shrink-wrapped to visible pixels
)
writer.attach([render_product])
```

**tight vs loose boxes:** *loose* boxes the whole object even if half-hidden; *tight* shrink-wraps only visible pixels. Detectors are trained on what's visible → tight.

### Step 3 — First manual capture

> **Environment:** `env_drone`

Generates 20 frames as a trial run before the full batch. Twenty is enough to spot a broken setup, and cheap enough to throw away.

```python
with rep.trigger.on_frame(max_execs=args.num_frames):
    with camera:
        rep.modify.pose(
            position=rep.distribution.uniform((-3, -3, 0.5), (3, 3, 3.0)),
            look_at="/World/Drone",
        )
rep.orchestrator.run_until_complete()
simulation_app.close()
```

Run it (note: `-p` through isaaclab.bat if your project scripts expect it, else plain python in the env):

*Run from:* `any folder`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\generate_drone_data.py --num_frames 20 --headless
```

### Step 4 — Inspect like a QA engineer

> **Environment:** none needed — you are looking at files in Explorer.

Opens the generated frames and label files by hand. A labelling problem found here costs minutes; the same problem found in Chapter 5 looks like a detector that will not train no matter what you do.

Open `C:\projects\drone_pursuit\drone_pursuit\data\raw\`. You should find `rgb_0000.png`…, plus `bounding_box_2d_tight_0000.npy` and a matching `..._labels.json` per frame. Open a few PNGs: is the drone visible, from varied angles/distances? Load one `.npy` (it's a structured array with `x_min, y_min, x_max, y_max` plus a `semanticId` that maps through the labels JSON to `"drone"`).

> ✅ **Checkpoint 4.1** — 20 frames exist; boxes in the .npy visually match the drone's position when you sketch them mentally over the PNG; labels JSON contains your "drone" class.

---

## 4.2 Domain randomization: make the detector unfoolable (≤1.5h)

> **What / Why / How it contributes:** Twenty photos of one drone in one gray world would train a detector that only works in that gray world. We now randomize lighting, background, distractor objects, and camera pose so the ONLY constant across thousands of frames is the drone itself — forcing the network to key on drone shape, not scenery. This is standard domain randomisation, tuned for a small flying object: heavy distance variation and both sky and ground backgrounds.

### What to randomize, and the *why* behind each dial

| Randomize | Range idea | Because at demo time… |
|---|---|---|
| Camera distance | 0.5–6 m from drone | the attacker appears at wildly different scales as the chase closes |
| Camera elevation | below AND above drone | a defender sees the attacker against **sky** when below it and against **ground** when above — two totally different background statistics |
| Dome light intensity/color | 500–6000, warm↔cool | arena lighting in your RL scene isn't fixed forever |
| Drone yaw/pitch | full yaw, ±20° pitch | a banking drone looks different from a level one |
| Distractor objects | 3–8 random shapes with random colors/textures scattered around | teaches "this is NOT a drone" — without negatives, everything vaguely dark becomes a drone |
| **Sun direction and intensity** | full azimuth, 10–80° elevation, wide intensity range | outdoors, a single strong directional light produces harsh shadows, silhouettes and washed-out highlights that a dome light never creates |

Implementation — wrap the randomizations in the same `on_frame` trigger:

```python
# distractors: a pool of primitive shapes, shuffled every frame
distractors = rep.create.group([
    rep.create.cube(count=4, semantics=[("class", "distractor")]),
    rep.create.sphere(count=4, semantics=[("class", "distractor")]),
])

with rep.trigger.on_frame(max_execs=args.num_frames):
    with camera:
        rep.modify.pose(
            position=rep.distribution.uniform((-6, -6, 0.2), (6, 6, 4.0)),
            look_at="/World/Drone",
        )
    with rep.get.prims(path_pattern="/World/Light"):          # ambient fill
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(500, 6000))
        rep.modify.attribute("inputs:color", rep.distribution.uniform((0.7, 0.7, 0.6), (1.0, 1.0, 1.0)))
    with rep.get.prims(path_pattern="/World/Sun"):            # the directional sun
        rep.modify.pose(rotation=rep.distribution.uniform((-80, 0, 0), (-10, 0, 360)))
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(1000, 12000))
        rep.modify.attribute("inputs:color", rep.distribution.uniform((1.0, 0.85, 0.7), (1.0, 1.0, 1.0)))
    with rep.get.prims(path_pattern="/World/Drone"):
        rep.modify.pose(rotation=rep.distribution.uniform((-20, -20, 0), (20, 20, 360)))
    with distractors:
        rep.modify.pose(
            position=rep.distribution.uniform((-5, -5, 0), (5, 5, 3.5)),
            scale=rep.distribution.uniform(0.05, 0.4),
        )
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
```

(We tag distractors with their own class but will simply *not* teach YOLO that class — they exist purely as visual noise. Their boxes get filtered out in 4.3's conversion.)

### Why the sun needs its own light, separate from the dome

A **dome light** illuminates evenly from every direction at once — the look of an overcast sky or a well-lit room. A **distant light** is a single source infinitely far away, with all its rays parallel. That is what the sun is, and it produces three things a dome light cannot:

- **A lit side and a dark side.** With the sun behind the attacker, the drone becomes a near-black silhouette against bright sky. This is the hardest case your detector will face outdoors, and without a directional light it never sees it.
- **Cast shadows**, which give the detector a shape it must learn to ignore.
- **Blown-out highlights**, where bright sky saturates the sensor and detail disappears.

The rotation range `(-80, 0, 0)` to `(-10, 0, 360)` sweeps the sun through a full circle of compass directions at elevations from 10° (low, near sunrise or sunset, long shadows and frequent backlighting) to 80° (near overhead, midday). The warm-to-white colour range spans golden low sun through neutral midday light.

**The intensity ceiling of 12000 is deliberately high.** You want a meaningful share of frames where the image is genuinely difficult — overexposed, the attacker reduced to a dark shape. Those frames are what teach the detector to survive the conditions you will actually fly in.

**What this still does not reproduce** is lens flare: the streaks and rings from light scattering inside a real lens, which appear when the sun is in or near the frame. Rendering does not simulate it, and a real Tello camera pointed near the sun produces it heavily. If you find the detector failing specifically when flying toward the sun, that is the cause — and Chapter 7.4's retraining on real footage is the fix. Or, more simply, fly with the sun behind you.

### The production run

> **Environment:** `env_drone`

*Run from:* `any folder`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\generate_drone_data.py --num_frames 2500 --headless
```

~2000–3000 frames is a solid single-class dataset. More helps, but variety matters more than raw count. Expect this to take a while — start it, then begin the QA tooling below.

> ✅ **Checkpoint 4.2** — flipping through 30 random production frames you see: near/far drones, sky and ground backgrounds, bright and dark scenes, distractors present, **some frames with the drone strongly backlit and nearly a silhouette**, and the drone always identifiable by YOU (if a human can't find it, don't expect the network to).
>
> If no frame looks harshly lit, the sun is not being randomised — check that `/World/Sun` exists and that the `rep.get.prims` pattern matches it.

---

## 4.3 Dataset QA + conversion to YOLO format (≤1.5h)

> **What / Why / How it contributes:** Raw Replicator output isn't what Ultralytics eats. We convert (Replicator pixel-corner boxes → YOLO's normalized center-x/center-y/width/height), filter junk (0-KB files, frames where the drone is invisible or under ten pixels, distractor boxes), and split train/val. Label errors introduced here surface in Chapter 5 as a low mAP that no amount of extra training will fix.

### The format translation, visualized

```
REPLICATOR (absolute pixel corners)        YOLO (normalized center + size)
(x_min,y_min)                              one line per object in a .txt:
   ┌─────────┐                             class  x_c     y_c     w      h
   │  drone  │                             0      0.512   0.430   0.146  0.118
   └─────────┘(x_max,y_max)                        └── all divided by image size (0..1) ──┘
```

Create `C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\convert_to_yolo.py`:

```python
import json, random, shutil
from pathlib import Path
import numpy as np

RAW = Path(r"C:\projects\drone_pursuit\drone_pursuit\data\raw")
OUT = Path(r"C:\projects\drone_pursuit\drone_pursuit\data\yolo")
IMG_W, IMG_H = 640, 480
MIN_BOX_PX = 10          # boxes smaller than this are unlearnable specks — drop the frame

kept, dropped = 0, 0
samples = []
for rgb in sorted(RAW.glob("rgb_*.png")):
    idx = rgb.stem.split("_")[-1]
    npy = RAW / f"bounding_box_2d_tight_{idx}.npy"
    lbl = RAW / f"bounding_box_2d_tight_labels_{idx}.json"
    if not npy.exists() or rgb.stat().st_size == 0:          # skip empty or corrupt frames
        dropped += 1; continue
    boxes = np.load(npy)
    id2label = {int(k): v["class"] for k, v in json.loads(lbl.read_text()).items()}
    lines = []
    for b in boxes:
        if id2label.get(int(b["semanticId"])) != "drone":    # distractor boxes: ignore
            continue
        w, h = b["x_max"] - b["x_min"], b["y_max"] - b["y_min"]
        if w < MIN_BOX_PX or h < MIN_BOX_PX:
            continue
        xc, yc = (b["x_min"] + w / 2) / IMG_W, (b["y_min"] + h / 2) / IMG_H
        lines.append(f"0 {xc:.6f} {yc:.6f} {w/IMG_W:.6f} {h/IMG_H:.6f}")
    if lines:                                                # keep only frames with a visible drone
        samples.append((rgb, lines)); kept += 1
    else:
        dropped += 1

random.seed(42); random.shuffle(samples)
n_val = max(1, int(0.15 * len(samples)))
for split, chunk in [("val", samples[:n_val]), ("train", samples[n_val:])]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    for rgb, lines in chunk:
        shutil.copy(rgb, OUT / "images" / split / rgb.name)
        (OUT / "labels" / split / f"{rgb.stem}.txt").write_text("\n".join(lines))
print(f"kept {kept}, dropped {dropped}, val {n_val}")
```

Then the dataset descriptor `C:\projects\drone_pursuit\drone_pursuit\data\yolo\drone.yaml`:

```yaml
path: C:\projects\drone_pursuit\drone_pursuit\data\yolo
train: images\train
val: images\val
names:
  0: drone
```

Finally, the QA step that catches label bugs *before* a wasted training run — draw 10 boxes onto their images and eyeball them (tiny script with PIL, or Ultralytics' own dataset visualizer once you're in Chapter 5). A conversion bug (e.g. swapped w/h) is instantly obvious visually and invisible numerically.

> ✅ **Checkpoint 4.3 — MILESTONE: Problem 2's fuel is loaded**
> 1. ~85/15 train/val split on disk in YOLO layout
> 2. Drop-rate sane (< ~20%; much higher → your camera randomization frames the drone out too often — tighten `look_at` ranges)
> 3. 10/10 spot-checked boxes hug the drone

---

# Chapter 5 — Object Detection: Training the Eyes

## 5.1 Train YOLOv8-nano on your synthetic drones (≤1.5h)

> **What / Why / How it contributes:** In the drone_vision env, we fine-tune YOLOv8n — a network pretrained on millions of everyday photos — to specialize in one thing: spotting a Crazyflie. Fine-tuning (vs. training from scratch) means the network already understands edges, shapes and lighting; it only needs to learn "drone." That's why a few thousand synthetic images suffice. Deliverable: best.pt, your detector.

### Why nano?

YOLOv8 comes in sizes n/s/m/l/x (nano→xlarge). Chapter 6 runs the detector *inside the simulation loop* — nano (~3M parameters) infers in a few milliseconds and keeps the demo snappy. One distinctive object class is exactly the regime where nano is sufficient; larger models earn their cost on visually cluttered, many-class scenes.

### Train

> **Environment:** `drone_vision`

*Run from:* `C:\projects\drone_pursuit\drone_pursuit`
```bat
conda activate drone_vision
cd C:\projects\drone_pursuit\drone_pursuit
yolo detect train data=C:\projects\drone_pursuit\drone_pursuit\data\yolo\drone.yaml model=yolov8n.pt epochs=60 imgsz=640 batch=16 name=drone_v1
```

20–40 min typically. While it runs, open `C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\` and watch artifacts appear — especially `train_batch0.jpg` (augmented training samples with boxes: your last chance to catch label bugs) and `results.png` (the loss/metric curves).

### Read the metrics like you read TensorBoard

- **box_loss / cls_loss** — should fall steadily. (Localization error / classification error.)
- **mAP50** — *mean Average Precision at 50% IoU*. Unpack: **IoU** (Intersection over Union) scores box overlap 0–1; "at 50" counts a detection correct if overlap ≥ 0.5; **AP** integrates precision across confidence thresholds; **m**ean averages over classes (we have one). For a single distinctive object on partly-synthetic-matching backgrounds, expect **mAP50 > 0.9**.
- **mAP50-95** — same, averaged over stricter overlap thresholds; it'll be lower; > 0.6 is fine for us (we need "roughly where," not surgical corners).

If mAP50 < 0.8, the fix is almost always **data, not hyperparameters**: check drop-rate, box QA, and whether hard cases (distant drone against ground clutter) exist in training.

> ✅ **Checkpoint 5.1** — mAP50 ≥ 0.9 on val; `val_batch0_pred.jpg` shows tight, confident boxes.

---

## 5.2 Stress-test and export to ONNX (≤1.5h)

> **What / Why / How it contributes:** A detector can score above 0.9 mAP on its own validation split and still miss the attacker in the pursuit arena, because the SDG scene and the arena differ in lighting, background and typical viewing distance. We capture frames from the defender's actual camera during a chase, test on those, then export to ONNX so the model can run inside env_drone.

### Step 1 — Capture ground-truth-free test frames

> **Environment:** `env_drone` — this runs Isaac Lab to capture the frames.

Attaches a camera to the defender and saves frames from an actual chase. These are the images the detector will really face, and they differ from the training set in lighting, background and typical distance.

First, somewhere to put them:

*Run from:* `any folder`
```bat
mkdir C:\projects\drone_pursuit\drone_pursuit\data\arena_frames
```


Quickest path: temporarily add a `TiledCameraCfg` to your pursuit env (this is also a dry run for Chapter 6):

```python
# in QuadcopterEnvCfg, in quadcopter_env.py
from isaaclab.sensors import TiledCameraCfg

tiled_camera: TiledCameraCfg = TiledCameraCfg(
    prim_path="/World/envs/env_.*/Robot/body/front_cam",       # rides on the defender's body
    offset=TiledCameraCfg.OffsetCfg(pos=(0.04, 0.0, 0.01), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.05, 30.0)),
    width=640, height=480,
)
```

Instantiate it in `_setup_scene` (`self._camera = TiledCamera(self.cfg.tiled_camera)` + `self.scene.sensors["camera"] = self._camera`), then run `play.py` with your Chapter-3 checkpoint, `--num_envs 1 --enable_cameras`, and a few lines in the loop (or a tiny callback script) dumping `self._camera.data.output["rgb"]` to PNGs during a chase. Save ~20 frames across the approach.

### Step 2 — Test the detector on them

> **Environment:** `drone_vision`

Runs the detector on those arena frames and shows you the predictions. Scoring well on its own validation split proves little; this is the check that reveals whether it works where it will actually run.

*Run from:* `C:\projects\drone_pursuit\drone_pursuit`
```bat
conda activate drone_vision
yolo predict model=C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\weights\best.pt source=C:\projects\drone_pursuit\drone_pursuit\data\arena_frames save=True conf=0.4
```

Study the saved predictions. Typical finding: great when the attacker is near, misses when it's a 12-pixel speck far away. If misses are frequent at the distances that matter (< ~5 m), loop back to 4.2 with more far-range camera samples — one more SDG batch + `yolo detect train ... resume` style fine-tune usually closes it. Testing on frames from the environment where the detector will actually run is the check most often skipped, and the one that catches this.

### Step 3 — Export

> **Environment:** `drone_vision`

Converts the trained detector into `drone_detector.onnx` at the drone's frame shape. This file is one of the two that cross into `env_drone`, and Chapter 6.1 loads it.

*Run from:* `C:\projects\drone_pursuit\drone_pursuit`
```bat
yolo export model=C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\weights\best.pt format=onnx imgsz=480,640
copy C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\weights\best.onnx C:\projects\drone_pursuit\drone_pursuit\models\drone_detector.onnx
```

**Why `imgsz=480,640` and not plain `640`.** Training letterboxes images internally, which is fine. But at flight time you want the detector's input to be exactly the frame shape you feed it — 640 wide by 480 high — so the rectangle it returns is already in that coordinate space. Exporting square would mean padding every frame and then subtracting the padding back out of every box, which is arithmetic with no upside and one more place to be wrong.

Re-run the 1.3 onnxruntime smoke test in `env_drone` against this file. The input shape should now read `[1, 3, 480, 640]`, and the output `(1, 5, 6300)`: 4 box numbers + 1 class score, for each candidate box.

> ✅ **Checkpoint 5.2 — MILESTONE: Problem 2 (perception) SOLVED**
> 1. Detector finds the attacker in real arena frames at chase-relevant distances
> 2. `drone_detector.onnx` loads and runs inside env_drone, input `[1, 3, 480, 640]`, output `(1, 5, 6300)`

---

# Chapter 6 — Integration: See → Estimate → Chase

## 6.1 The reading converter: rectangle → the seven numbers (≤1.5h)

**"Converter" (called a bridge elsewhere in robotics) just means a small piece of code that sits between two subsystems and translates one's output into the other's input.** Here the detector outputs a rectangle in pixels; the policy expects seven normalised numbers. Nothing decides anything — it only rescales and remembers.

> **What / Why / How it contributes:** We build the translator between the detector's output and the policy's input. Because Chapter 3 defined the policy's input as camera readings, this translator only rescales the rectangle against the image dimensions — no conversion, no assumption about the attacker's size. We then verify it against the training-time projection, so any Chapter 6 weirdness later can be blamed on the detector or the policy, never on this file.

### What the bridge has to do

The detector hands you a rectangle: a centre point and a size, both in pixels. The policy wants the seven readings from §0.4. The entire translation is *divide by the image dimensions*:

```
   the 640-wide camera image
 ┌──────────────────────────────┐
 │              ┌──┐            │   rectangle centre, measured from the
 │              │▪▪│            │   image centre → horizontal + vertical bearing
 │      ·  ·  · │+ │ ·  ·  ·    │
 │              └──┘            │   rectangle width, as a fraction of the
 │                              │   image width → angular size
 └──────────────────────────────┘
   image centre = dead ahead
```

No focal length, no aperture, no physical dimensions. This follows from §0.4: the policy was trained on quantities already expressed in image terms, so a real rectangle needs only rescaling.

### Step 1 — Write the bridge

> **Environment:** none needed — you are creating a folder and a file.

Creates the converter that turns the detector's rectangle into the seven numbers the policy expects. It only rescales and remembers — no interpretation — which is why nothing in it can be got wrong.

Create the folder for the demo and flight scripts, which Chapters 6 and 7 both use:

*Run from:* `any folder`
```bat
mkdir C:\projects\drone_pursuit\drone_pursuit\scripts\demo
```

Then create `C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_bridge.py`:

```python
import numpy as np, onnxruntime as ort


class DroneDetector:
    """Runs the exported YOLO model and returns the single best rectangle."""

    def __init__(self, onnx_path, conf_thresh=0.4):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.conf = conf_thresh

    def detect(self, rgb_uint8_hwc):
        """480x640x3 uint8 (H, W, C) → (x_center, y_center, width, height) in pixels, or None."""
        x = rgb_uint8_hwc.astype(np.float32) / 255.0        # scale to 0..1
        x = np.transpose(x, (2, 0, 1))[None]                # HWC → 1,C,H,W
        out = self.sess.run(None, {self.input_name: x})[0][0]   # (5, 8400)
        boxes, scores = out[:4, :], out[4, :]
        best = int(np.argmax(scores))
        if scores[best] < self.conf:
            return None                                     # attacker not seen this frame
        return boxes[:, best]


class CameraReadingBridge:
    """Rectangle → the seven numbers the policy was trained on. Holds last value on a miss."""

    def __init__(self, img_w=640, img_h=480):   # must match the frame you feed it
        self.img_w, self.img_h = img_w, img_h
        self.bx = self.by = self.asz = 0.0
        self.has_seen = False

    def update(self, bbox):
        prev = (self.bx, self.by, self.asz)
        if bbox is not None:
            x_c, y_c, w, h = bbox
            self.bx = (x_c - self.img_w / 2) / (self.img_w / 2)     # −1 … +1
            self.by = (y_c - self.img_h / 2) / (self.img_h / 2)     # −1 … +1
            self.asz = w / self.img_w                               #  0 … 1
            self.has_seen = True
            visible = 1.0
        else:
            visible = 0.0                                           # values stay frozen
        d = (self.bx - prev[0], self.by - prev[1], self.asz - prev[2])
        return np.array([self.bx, self.by, self.asz, d[0], d[1], d[2], visible],
                        dtype=np.float32)

    def captured(self, threshold):
        """Camera-only victory condition, using the value calibrated in Chapter 3.3."""
        return self.has_seen and self.asz > threshold
```

### Step 2 — Check it against the readings calculated during training

> **Environment:** `env_drone`

Compares the converter's output against what the simulator computed for the same moment. Agreement proves the two definitions match; a constant-factor disagreement means the camera's field of view does not match Chapter 3.1's settings.

This test is easy because you can produce both versions of the same reading. Reuse the frame-capture setup from 5.2, but for each saved frame *also* record what `_camera_readings()` computed for that same moment. Then run the bridge on the image and compare:

```
frame 12   projected (training):  bx=+0.109  by=−0.250  size=0.053
           bridge   (from image):  bx=+0.115  by=−0.244  size=0.049   ✓
```

Acceptance: bearings agree within a few hundredths, angular size within about 20% (the rectangle wobbles with the attacker's attitude — expected and harmless). Disagreement in *sign* means an axis is flipped; disagreement by a large constant factor means your `TiledCameraCfg` field of view doesn't match the `cam_focal_mm` / `cam_aperture_mm` you set in 3.1.

Watch for the second one especially. Nothing errors: the policy receives plausible numbers that mean something different from what it trained on, and the chase degrades in a way that looks like a training problem.

### Step 3 — Check the blind-spot behaviour

> **Environment:** `env_drone`

Feeds the converter frames where the attacker is out of view. It should hold the previous readings and drop the visible flag — the same pattern the policy met during training whenever the attacker got too small to detect.

Feed the bridge a few frames where the attacker is genuinely out of view. It should return `visible = 0`, hold the previous bearings, and report all three deltas as zero. This is the same pattern the policy saw during training whenever `ang_size` fell below the visibility threshold, so it responds with behaviour it has already learned.

> ✅ **Checkpoint 6.1**
> 1. Bridge output matches the training-time projection on ≥15 varied frames
> 2. Field of view confirmed consistent between `TiledCameraCfg` and the cfg constants from 3.1
> 3. Out-of-view frames produce `visible = 0` with frozen values and zero deltas

---

## 6.2 The final demo: the loop closes (≤1.5h)

> **What / Why / How it contributes:** Everything meets: the pursuit env with its onboard camera, your trained policy, the ONNX detector, and the reading converter from 6.1. We run one environment and, each control step, replace observation slots 6–12 with the seven numbers measured from a rendered frame. Everything the defender knows about the attacker now arrives through the camera. If it still captures, the full see-decide-act loop works.

### The demo architecture (one env, one loop)

```
              ┌─────────────────── every control step ───────────────────┐
              │                                                          │
 TiledCamera ─┤→ rgb frame → DroneDetector → rectangle → CameraReadingBridge │
 (on defender)│                    │ (None?)                  │          │
              │                    ▼                          ▼          │
              │        visible = 0, hold last values    7 numbers        │
              │                                               │          │
              │   obs[0:6]  ← simulator (the drone's sensors) │          │
              │   obs[6:13] ← CAMERA ────────────────────────┘          │
              │                        │                                 │
              │                        ▼                                 │
              │                policy.act(obs) → thrust + moments        │
              │                        │                                 │
              │      bridge.captured(capture_ang_size)? → declare win    │
              └──────────────────────────────────────────────────────────┘
```

Two design notes worth stating plainly:

- **Self-state stays from the simulator, and that is not cheating.** A real drone reads its own velocity and attitude from its inertial sensors. Only knowledge *about the attacker* has to be earned through the camera, and that is precisely the part we are replacing.
- **Losing sight is already trained for.** When the detector comes up empty, the bridge holds the previous readings and sets `visible = 0` — the same pattern the policy met thousands of times during training whenever the attacker fell below the detection threshold.

### The demo script (the trickiest file in the tutorial — take it slow)

`C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_pursuit_demo.py`:

```python
import argparse, torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)   # your Ch.3 best_agent.pt
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
bridge = CameraReadingBridge()
CAPTURE_ANG_SIZE = raw_env.cfg.capture_ang_size     # calibrated back in Chapter 3.3

obs, _ = env.reset()
while app.is_running():
    # 1) SEE — this step's rendered frame, (1, H, W, 3) uint8 → numpy HWC
    rgb = raw_env._camera.data.output["rgb"][0].cpu().numpy()
    readings = bridge.update(detector.detect(rgb))     # 7 numbers

    # 2) SPLICE — the camera replaces exactly the slots it owns
    obs[:, 6:13] = torch.tensor(readings, device=obs.device)

    # 3) ACT
    with torch.no_grad():
        actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
    obs, reward, terminated, truncated, info = env.step(actions)

    # 4) COMPARE — the camera's verdict alongside the simulator's true distance
    truth = raw_env._dist[0].item()
    print(f"true dist {truth:5.2f} m | seen {readings[6]:.0f} "
          f"| size {readings[2]:.3f} | camera says captured: "
          f"{bridge.captured(CAPTURE_ANG_SIZE)}")
```

Run it:

*Run from:* `any folder`
```bat
conda activate env_drone
python C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_pursuit_demo.py --checkpoint C:\projects\drone_pursuit\drone_pursuit\logs\skrl\<run-folder>\checkpoints\best_agent.pt --enable_cameras
```

⚠️ Expect to iterate. This script touches every subsystem, and the usual suspects are small: the exact wrapper import path for your Isaac Lab minor version, the camera attribute name, agent-loading details. **Debug method:** comment out the splice first, so the policy runs on the training-time projected readings. That must reproduce Chapter 3 behaviour exactly — which isolates whether a problem lives in checkpoint loading or in the vision path. Only then re-enable the splice.

**What success looks like:** the chase is less smooth than in Chapter 3. The detector's rectangle shifts by a few pixels between frames, so the bearings shift with it and the defender corrects more often. It still closes and captures. The difference between its capture rate on calculated readings and on measured readings is the number worth recording.

**What to watch at the end of a chase:** `true dist` drops below your capture radius, and within a frame or two `camera says captured` flips to True. The simulator and the camera are measuring the same event by completely different means; if they agree within a frame or two, your Chapter 3.3 calibration was correct.

> ✅ **Checkpoint 6.2 — FINAL MILESTONE: Problem 3 (integration) SOLVED**
> The defender captures the attacker with everything it knows about the attacker coming from its camera and your detector — no assumed dimensions, no positions from the simulator, no camera geometry. Record a video.

---

## 6.3 What happens next

**The simulation half is complete.** The defender captures a moving attacker using only what a camera reports, with no privileged information about the target.

**Chapter 7 is the next step, and it is the point of the project**: putting this policy on the Tello you set up in 1.4 and finding out whether it transfers. Nothing needs retraining — the policy already speaks the drone's language.

### Directions beyond that

Each of these builds on what you now have, ordered by effort:

1. **Reactive attacker** (hours): make the scripted path flee — add a velocity component away from the defender, then re-run the curriculum. A harder pursuit problem, entirely in simulation.
2. **Bearing-only range estimation** (a project): add a filter that accumulates bearings across many frames and combines them with your own known motion to recover the attacker's actual position, plus an estimate of how uncertain it is. Standard probabilistic state estimation, and it sits *below* the existing policy — no retraining needed. It would also give you something Chapter 6 deliberately gave up: an inspectable distance estimate.
3. **True multi-agent** (project): 2 defenders against 1 RL-controlled attacker, using skrl's IPPO or MAPPO — the Direct workflow supports multi-agent environments. The reference point is the Tsinghua Multi-UAV pursuit-evasion work: adaptive curriculum, evader prediction network, and sim-to-real on real quadrotors.
4. **OmniDrones** (project): a full drone-RL framework on Isaac Sim with realistic rotor dynamics and controllers, from the same lineage as that paper — the step beyond the simplified force model used here.

*(The noise-hardening that used to sit at the top of this list is no longer optional or deferred — it is built into Chapter 3.1 and 7.4, because hardware needs it.)*

---

# Chapter 7 — Flying It: The Sim-to-Real Test

Chapter 6 proved the system works in simulation with a rendered camera. This chapter puts it on the Tello you set up and measured in 1.4.

**Nothing needs retraining here.** The policy already commands stick channels, already decides at your drone's rate, already expects delayed readings, and already uses only telemetry the Tello can report. That is what 1.4 bought by coming before Chapter 3.

**What you still need:** the attacker drone (a cheap toy quadcopter, C$50–70), spare propellers, a couple of batteries, and bright tape or a coloured shell for the target so the detector can find it at range. Total remaining spend is under C$100.

---

## 7.1 Export the policy (≤1.5h)

> **What / Why / How it contributes:** This is the transfer itself. The trained policy lives inside a skrl checkpoint that only Isaac Lab can open. Here you extract the decision-making network and save it as a standalone file your flight script loads without Isaac Lab installed — exactly what you did to the detector in Chapter 5.2, for the same reason.

### What a checkpoint is, and what you need from it

**A checkpoint is a save file.** During training, skrl periodically writes the state of the learning process to a `.pt` file so a run can be resumed or evaluated later. It is what `play.py` loads and what you resumed from when raising the attacker's speed.

| Inside it | Needed at flight time? |
|---|---|
| **The actor** — maps 17 observations to 4 commands | **Yes. This is the policy.** |
| **The critic** — estimated future reward, used only to compute training updates | No |
| **Optimiser state** — how PPO was adjusting weights mid-run | No |

Exporting means reaching past the training machinery and taking only the actor.

### Step 1 — Export

> **Environment:** `env_drone`

Extracts the decision-making network from your training checkpoint and saves it as `policy.onnx`. Your flight script has no Isaac Lab installed, so this is what makes the trained policy runnable beside a real drone.

`C:\projects\drone_pursuit\drone_pursuit\scripts\demo\export_policy.py`:

```python
"""Extract the actor network from a skrl checkpoint and save it as ONNX."""
import argparse, torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, default=r"C:\projects\drone_pursuit\drone_pursuit\models\policy.onnx")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
import drone_pursuit.tasks  # noqa: F401

TASK = "Template-Drone-Pursuit-Direct-v0"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
env_cfg.scene.num_envs = 1
env = SkrlVecEnvWrapper(gym.make(TASK, cfg=env_cfg, render_mode=None))

runner = Runner(env, load_cfg_from_registry(TASK, "skrl_cfg_entry_point"))
runner.agent.load(args.checkpoint)
runner.agent.set_running_mode("eval")


class DeterministicActor(torch.nn.Module):
    """Wraps the policy so ONNX sees a plain observations → actions function."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        out = self.net.act({"states": obs}, role="policy")
        return out[2]["mean_actions"]        # the mean, not a random sample


wrapper = DeterministicActor(runner.agent.policy)
dummy = torch.zeros(1, 17, device=runner.agent.device)
torch.onnx.export(
    wrapper, dummy, args.out,
    input_names=["obs"], output_names=["action"],
    dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    opset_version=17,
)
print(f"exported → {args.out}")
```

**Why the mean and not a sample.** During training the policy deliberately adds randomness so it explores alternatives. At flight time you want the same observation to produce the same command every time, so you take the centre of the distribution rather than drawing from it.

⚠️ The key holding the mean varies between skrl versions. If that line fails, print `out` and inspect the third element — usually `mean_actions`, sometimes `net_output`.

### Step 2 — Prove the export is faithful

> **Environment:** `env_drone`

Feeds the same numbers to the checkpoint and to the exported file and compares the results. A faulty export flies badly for reasons you would otherwise spend days blaming on the drone.

Do not skip this. A wrong export flies badly for reasons you would spend days blaming on the drone.

```python
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
probe = torch.randn(1, 17, device=runner.agent.device)
from_torch = wrapper(probe).detach().cpu().numpy()
from_onnx = sess.run(None, {"obs": probe.cpu().numpy()})[0]
print("max difference:", np.abs(from_torch - from_onnx).max())
```

Below about 1e-4 means faithful.

### Step 3 — Both models in one place

> **Environment:** none needed — you are checking that two files exist.

Confirms the detector and the policy sit together in `models\`. These two files are the entire sim-to-real transfer; everything else stays behind in the simulator.

```
C:\projects\drone_pursuit\drone_pursuit\models\
    drone_detector.onnx     ← Chapter 5.2
    policy.onnx             ← this subchapter
```

**These two files are the entire transfer.** The environment, the reward, the arena and Isaac Lab itself all stay behind.

> ✅ **Checkpoint 7.1**
> 1. `policy.onnx` exists and loads under onnxruntime
> 2. Its output matches the checkpoint to within 1e-4 on random input
> 3. Both models sit together in `models\`

---

## 7.2 Build the flight script and bench-test it (≤1.5h)

> **What / Why / How it contributes:** This assembles everything into the loop that flies the drone. You then run it with the propellers removed, so every part can be verified while nothing can hurt you — including the one bug most likely to send a drone into a wall.

### The loop

```
  ┌─► newest camera frame
  │        ↓
  │   detector → rectangle, or nothing
  │        ↓
  │   seven readings   (Chapter 6.1's converter, unchanged)
  │        ↓
  │   drone's own speed and tilt
  │        ↓
  │   assemble 17 numbers, in the trained order
  │        ↓
  │   policy → four numbers
  │        ↓
  │   send as stick commands
  │        ↓
  └── wait, so the loop runs at the rate you trained at
```

### The script

First create the folder the script records into. **Do this before flying, not after** — the script opens its log file immediately after takeoff, so a missing folder crashes it with the drone already in the air:

*Run from:* `any folder`
```bat
mkdir C:\projects\drone_pursuit\drone_pursuit\flights
```

Then `C:\projects\drone_pursuit\drone_pursuit\scripts\demo\fly_real.py`, run in `env_drone`. It never imports Isaac Lab:

```python
"""Fly the Tello with the policy trained in simulation."""
import time, threading, csv, datetime
import numpy as np, onnxruntime as ort, cv2
from djitellopy import Tello
from vision_bridge import DroneDetector, CameraReadingBridge

CONTROL_HZ = 20                  # your measurement from 1.4
DT = 1.0 / CONTROL_HZ
MODELS = r"C:\projects\drone_pursuit\drone_pursuit\models"

detector = DroneDetector(rf"{MODELS}\drone_detector.onnx")
bridge = CameraReadingBridge()
policy = ort.InferenceSession(rf"{MODELS}\policy.onnx", providers=["CPUExecutionProvider"])

drone = Tello()
drone.connect()
print(f"battery: {drone.get_battery()}%")
if drone.get_battery() < 30:
    raise SystemExit("charge before flying")
drone.streamon()
reader = drone.get_frame_read()

# --- every flight is recorded, for 7.4 --------------------------------------
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log = open(rf"C:\projects\drone_pursuit\drone_pursuit\flights\flight_{stamp}.csv", "w", newline="")
writer = csv.writer(log)
writer.writerow(["t", "bx", "by", "asz", "dbx", "dby", "dasz", "visible",
                 "vx", "vy", "vz", "roll", "pitch",
                 "act0", "act1", "act2", "act3", "loop_ms"])
video = cv2.VideoWriter(rf"C:\projects\drone_pursuit\drone_pursuit\flights\flight_{stamp}.mp4",
                        cv2.VideoWriter_fourcc(*"mp4v"), CONTROL_HZ, (960, 720))

# --- emergency stop ----------------------------------------------------------
# The drone already HAS one: drone.emergency() cuts the motors. This thread only
# gives you a keyboard trigger for it. It is a convenience, not a new mechanism.
running = True


def watch_for_stop():
    global running
    choice = input(">>> ENTER = land normally   |   'x' + ENTER = cut motors <<<\n")
    running = False
    if choice.strip().lower() == "x":
        drone.emergency()


threading.Thread(target=watch_for_stop, daemon=True).start()

prev_action = np.zeros(4, dtype=np.float32)
t0 = time.time()
drone.takeoff()
time.sleep(2)

try:
    while running:
        loop_start = time.time()

        # 1 — SEE
        frame = reader.frame
        if frame is None:
            continue
        video.write(frame)
        rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 480))

        # 2 — the seven readings
        readings = bridge.update(detector.detect(rgb))

        # 3 — the drone's own motion (speeds are cm/s → m/s)
        roll = np.radians(drone.get_roll())
        pitch = np.radians(drone.get_pitch())
        self_state = np.array([
            drone.get_speed_x() / 100.0,
            drone.get_speed_y() / 100.0,
            drone.get_speed_z() / 100.0,
            np.sin(roll), np.sin(pitch), -np.cos(roll) * np.cos(pitch),
        ], dtype=np.float32)

        # 4 — assemble the 17 numbers, in the trained order
        obs = np.concatenate([self_state, readings, prev_action])[None].astype(np.float32)

        # 5 — DECIDE.  ◄── THIS LINE IS THE TRAINED POLICY RUNNING.
        #     `policy` is the ONNX file from 7.1. Everything Chapters 2 and 3
        #     trained lives inside it.
        action = np.clip(policy.run(None, {"obs": obs})[0][0], -1.0, 1.0)
        prev_action = action.copy()

        # 6 — ACT.  Tello channels: left/right, forward/back, up/down, yaw (−100..100)
        drone.send_rc_control(
            int(action[1] * 100),
            int(action[0] * 100),
            int(action[2] * 100),
            int(action[3] * 100),
        )

        # 7 — record and hold the rate
        loop_ms = (time.time() - loop_start) * 1000
        writer.writerow([f"{time.time() - t0:.3f}", *[f"{v:.4f}" for v in readings],
                         *[f"{v:.3f}" for v in self_state[:5]],
                         *[f"{v:.3f}" for v in action], f"{loop_ms:.1f}"])

        if loop_ms / 1000 < DT:
            time.sleep(DT - loop_ms / 1000)
        else:
            print(f"loop overran: {loop_ms:.0f} ms")
finally:
    drone.send_rc_control(0, 0, 0, 0)
    drone.land()
    drone.streamoff()
    log.close()
    video.release()
    print(f"flight recorded → flight_{stamp}.csv / .mp4")
```

### Three things worth understanding

**The channel mapping in step 6 is a guess until verified.** Which action index drives which stick depends on how you ordered them in 3.1. Get it wrong and the drone goes sideways when it should go forward. Verifying it is the main purpose of the bench test below.

**Every flight is recorded automatically** — a CSV of every observation and command, plus the video. This costs nothing during flight and is the entire input to 7.4. A flight you did not record teaches you only what you happened to notice at the time.

**The Tello lands itself after 15 seconds without a command**, so a crashed script does not leave a drone flying.

### Bench test — propellers removed

Take the propellers off. Comment out `takeoff()` and `land()`. Run it and hold the attacker in front of the camera.

| Check | What you should see |
|---|---|
| Detection works | `visible` is 1 with the attacker in view, 0 when hidden |
| Bearings correct | Move it left; `bx` moves consistently one way — note which |
| Size responds | Move it closer; `asz` grows |
| Commands sensible | Attacker to the left → the left/right command is non-zero and correctly signed |
| Rate holds | No "loop overran" messages |
| Emergency stop | Both ENTER and `x` behave as expected |

**If commands point the wrong way, fix the channel mapping now.**

> ✅ **Checkpoint 7.2**
> 1. Bench run passes all six checks
> 2. A CSV and MP4 appeared in `flights\`
> 3. You know which action index drives which direction

---

## 7.3 Fly it, in stages (≤1.5h per stage)

> **What / Why / How it contributes:** Each stage adds exactly one new thing, so a failure names its own cause. Skipping stages converts a diagnosable problem into a broken drone and an unanswerable question.

### Before every session

Propeller guards on. Eye protection. Clear space. Nobody else present. Battery above 30%. Hand near the keyboard.

### The stages

**Stage 1 — Hover only.** Restore `takeoff()`, but force the policy output to zero so it just hovers. Checks the loop holds rate while flying, video keeps up, landing works. Two minutes.

**Stage 2 — Live policy, no attacker.** Let the policy run with nothing to detect. It should sit roughly still, holding its last reading with `visible` at 0. If it wanders off aggressively, the lost-sight behaviour is wrong — stop and check the converter is holding values rather than zeroing them.

**Stage 3 — Stationary attacker.** On a stand, two metres away. The defender should approach and stop at capture distance. **This is the first real test of the transfer.**

**Stage 4 — Hand-carried attacker.** Walk it slowly across the space.

**Stage 5 — Flying attacker, slow.** Both airborne, attacker near walking pace.

**Stage 6 — Increase attacker speed.** The Chapter 3.3 curriculum again, in hardware.

### Reading a failure

| What you see | Most likely cause | What to check in the log |
|---|---|---|
| Oscillates while hovering | Stand-in stabiliser in 3.1 does not match the Tello's | Commands alternating sign rapidly |
| Overshoots repeatedly | Real delay exceeds what you trained for | Re-measure delay; widen `obs_delay_*` |
| Drifts to one side | Real drift larger than randomised | `vx`/`vy` biased with near-zero commands |
| Stops or wanders when the target moves | Detector losing the target | `visible` dropping to 0 often |
| Works close, fails far | Detector cannot resolve a small target | `asz` small, `visible` flickering |
| Flies the wrong way entirely | Channel mapping wrong | Commands correctly signed but on the wrong axis |

**Read the log before forming a theory.** Almost every row above is answered by the CSV rather than by what you saw.

### On legality

Indoors in a private space is the least regulated situation. Outdoors in Canada, drone operation falls under Transport Canada rules, and deliberately flying one aircraft toward another is not routine. Check current requirements before flying outdoors at all.

> ✅ **Checkpoint 7.3 — SIM-TO-REAL VALIDATED**
> A policy trained entirely in simulation flies a real drone, finds a real target through a real camera, and closes on it. Record it.

---

## 7.4 Improve the system from real flight data (≤1.5h per cycle)

> **What / Why / How it contributes:** Your first flights will be imperfect, and the recordings from 7.2 tell you why. This turns them into improvements — a better detector and a more truthful simulation. This is the loop professional sim-to-real work runs continuously.

### The honest constraint

**You cannot train the policy directly on real flight data.** PPO needs millions of steps; each flight gives a few thousand, and the mistakes it would learn from are crashes. Nobody does this on real hardware for a task like this.

What you do instead loops through the simulator:

```
   real flight ──► what was wrong? ──► fix the SIMULATION ──► retrain ──► fly again
                                            ▲
                                  (the sim becomes more truthful each
                                   cycle, so the policy transfers better)
```

This is **system identification**: using measurements of the real thing to correct your model of it.

**What is automatic, and what is not.** Collection is automatic and complete — every observation, command, loop time and video frame is written to `flights\` on every run without you doing anything. *Interpretation* is manual: you read the logs, decide what was wrong, change a parameter. No code here adjusts the simulator by itself, and building one would be a research project. What you have is complete data and a short checklist, which is what makes the manual step twenty minutes rather than guesswork.

### Loop 1 — Improve the detector (biggest gains, least effort)

> **Environment:** `drone_vision` for the YOLO commands; the frame extraction itself needs none.

Your detector has only seen synthetic renders. The recorded videos are real footage of the real target.

1. **Extract frames** — every tenth; consecutive ones are nearly identical. Put them somewhere of their own:

*Run from:* `any folder`
```bat
mkdir C:\projects\drone_pursuit\drone_pursuit\data\real\images
```


2. **Pre-label them with the detector you already have**, then correct what it got wrong. You never label from a blank slate:

*Run from:* `C:\projects\drone_pursuit\drone_pursuit`
```bat
conda activate drone_vision
yolo detect predict model=C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\weights\best.pt source=C:\projects\drone_pursuit\drone_pursuit\data\real\images save_txt=True save_conf=True conf=0.25
```

`save_txt` writes YOLO-format labels beside each image. Load those into any labelling tool and you are reviewing boxes rather than drawing them — five to ten times faster.

3. **Propagate across nearby frames.** Because these come from continuous video, a box on frame *n* is nearly right for frame *n+1*. CVAT, Label Studio and Roboflow all interpolate between two corrected frames.

4. **Spend effort where the detector failed.** Two automatic ways to find those frames: any pre-labelled box below about 0.4 confidence, and any row in the flight CSV where `visible` flickers between 0 and 1. Pull those timestamps and extract the matching frames. Frames where the attacker is large and obvious teach almost nothing.

**Realistic effort:** reviewing 200–400 pre-labelled frames takes perhaps 45 minutes, versus several hours drawing from scratch. It cannot be fully automated — an auto-labeller good enough to do it would already be the detector you are trying to build.

5. **Fine-tune from your synthetic model**, not from scratch:

*Run from:* `C:\projects\drone_pursuit\drone_pursuit`
```bat
yolo detect train data=C:\projects\drone_pursuit\drone_pursuit\data\real\drone.yaml model=C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\weights\best.pt epochs=40 imgsz=640 name=drone_real
```

6. **Re-export to ONNX** at `imgsz=480,640` and re-measure `capture_ang_size` — a different target at the same distance fills a different share of the frame.

### Loop 2 — Make the simulation more truthful

> **Environment:** none needed — you are reading logs and editing cfg values.

The CSV contains what you commanded and how the drone responded. That is enough to correct the biggest modelling errors.

**Correct the stand-in stabiliser.** Find a segment where you commanded a steady forward value and look at how `vx` rose. Real drone faster than simulation → raise `vel_gain`; slower → lower it. Ten minutes comparing plots beats any amount of guessing.

**Correct the drift range.** Find segments with near-zero commands and look at residual `vx`/`vy`. That is the Tello's true drift. Widen the randomisation range in 3.3 to contain it comfortably.

**Correct the delay range.** Consistent overshoot means the real delay exceeds what you trained for. Widen `obs_delay_max`.

**Correct the speed limits.** If maximum forward produced less speed than `max_speed` assumes, lower it — otherwise the policy expects performance the drone does not have.

Then retrain and fly again.

### What good iteration looks like

| Cycle | Typical change | Typical result |
|---|---|---|
| 1 | Detector fine-tuned on real frames | Far fewer lost-sight events |
| 2 | `vel_gain` and drift corrected from logs | Smoother approach, less weaving |
| 3 | Delay range widened | Overshoot reduced |
| 4 | Attacker speed increased | A harder task, honestly passed |

**Change one thing per cycle.** Two changes and a better result tells you nothing about which helped — the same rule that governed the simulation chapters, now applied to hardware.

> ✅ **Checkpoint 7.4**
> 1. At least one detector fine-tune from real footage, with measurably fewer lost-sight events
> 2. At least one simulation parameter corrected from flight logs
> 3. A second flight session visibly better than the first

---
# Appendix A — Version & compatibility ledger

| Component | Version targeted | Pin reason |
|---|---|---|
| Isaac Sim | 5.1.0 (pip) | Isaac Lab 2.3 is built on it; Python 3.11 required |
| Isaac Lab | 2.3 line — `isaaclab` extension 0.48.0, `isaaclab_tasks` 0.11.8, `isaaclab_assets` 0.2.3 | current API (`isaaclab.*`), TiledCamera, template wizard |
| torch | 2.7.0+cu128 | shipped with this Isaac Lab; do not replace |
| Python (both envs) | 3.11 | dictated by Isaac Sim 5.x |
| torch (env_drone) | 2.7.0+cu128, installed by Isaac Lab — **never upgrade manually** | CUDA build; a re-resolve silently drops GPU support |
| setuptools (env_drone) | 80.10.2 — **must stay below 81** | 82 removed `pkg_resources`, which tensorboard's dependencies still import |
| skrl | as installed by Isaac Lab RL extras | the wizard's yaml targets it |
| ultralytics (drone_vision only) | latest | isolated env → free to float |
| onnxruntime (env_drone) | latest | runs the detector without pulling in torch |

**Symptoms → likely cause quick table**

| Symptom | Check |
|---|---|
| `cuda: False` after any install | something replaced torch — `pip list | findstr torch`, reinstall per Isaac Lab docs |
| Crazyflie USD not found | asset path renamed in 5.x: `Robots/Bitcraze/Crazyflie/cf2x.usd` |
| Camera task OOM | fewer envs, smaller resolution; cameras dominate VRAM |
| Ghost imports after refactor | delete `__pycache__` in the task package |
| `isaaclab.__version__` looks too low (0.x) | that is the extension version, not the release; check `pip list | findstr isaacsim` instead |
| `git describe` shows an old tag on `main` | `main` is untagged between releases; the Isaac Sim version is the reliable indicator |
| ONNX output shape ≠ (1,5,8400) | exported the wrong .pt (pretrained 80-class instead of your best.pt) |
| Chase works in Ch.3, degrades badly in Ch.6 | field of view mismatch — `TiledCameraCfg` focal length must match `cam_focal_mm`/`cam_aperture_mm` in the env cfg |
| Defender flies off when attacker leaves frame | `visible` flag not wired, or previous readings not held on a miss |
| Ch.7: policy fine in sim, oscillates on hardware | real delay exceeds the trained range, or the stand-in stabiliser in 3.1 does not match the drone — re-measure, widen `obs_delay_*`, retrain |
| Ch.7: video latency near one second | buffering, not the drone — set the capture buffer size to 1 and read in a thread that discards stale frames |
| Ch.7: works close, fails far | detector cannot resolve a small target — add far-range training frames or brighter markers |
| Ch.7: drone flies sideways when it should go forward | action-to-stick channel mapping wrong in the bridge — verify on the bench with propellers off |
| Ch.7: ONNX policy disagrees with the checkpoint | the export wrapper extracted a sampled action instead of the distribution mean |
| Ch.7: "loop overran" printing constantly | detector or frame conversion too slow for the control rate — lower the detector input size |

# Appendix B — Time budget recap

| Subchapter | ~Time | Depends on |
|---|---|---|
| 1.0 Isolate the project | 1.5h | — |
| 1.1 Understand the flight code | 1.5h | 1.0 |
| 1.2 External project | 1.5h | 1.0, 1.1 |
| 1.3 Vision env + boundary tests | 1.5h | 1.0 |
| **1.4 Set up and measure the Tello** | **1.5h** | **hardware in hand** |
| 2.1 Attacker in the scene | 1.5h | 1.2 |
| 2.2 Scripted motion | 1.5h | 2.1 |
| 3.1 Commands and observations | 1.5h | 2.2, **1.4** |
| 3.2 Reward design | 1.5h | 3.1 |
| 3.3 Randomise, train, curriculum | 1.5h (+ GPU hours) | 3.2 |
| 4.1 SDG scene | 1.5h | 1.2 (parallel to Ch.3) |
| 4.2 Randomisation | 1.5h | 4.1 |
| 4.3 QA + YOLO convert | 1.5h | 4.2 |
| 5.1 Train YOLO | 1.5h | 4.3, 1.3 |
| 5.2 Stress-test + ONNX | 1.5h | 5.1 |
| 6.1 Reading converter | 1.5h | 5.2 |
| 6.2 Simulated demo | 1.5h+ | 3.3, 6.1 |
| 6.3 What happens next | — | — |
| 7.1 Export the policy | 1.5h | 6.2 |
| 7.2 Flight script + bench test | 1.5h | 7.1 |
| 7.3 Staged flight tests | 1.5h per stage | 7.2 |
| 7.4 Improve from real flight data | 1.5h per cycle | 7.3 |

**Buy the hardware early.** Subchapter 1.4 needs the Tello in hand, and everything from Chapter 3 onward is built on its measurements. Ordering the drone while working through 1.0–1.3 keeps the sequence unbroken.

## Sources

[1] Isaac Lab Project Developers, NVIDIA. "Local Installation — Isaac Lab Documentation" (Isaac Sim 5.1 / Python 3.11 requirements). 2026. https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

[2] Isaac Lab Project Developers, NVIDIA. "Available Environments — Isaac Lab Documentation" (Isaac-Quadcopter-Direct-v0, camera tasks, multi-agent IPPO/MAPPO support). 2026. https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html

[3] Isaac Lab Project Developers, NVIDIA. "Camera — Sensors — Isaac Lab Documentation" (TiledCamera, annotators, --enable_cameras, VRAM guidance). 2026. https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html

[4] Isaac Lab Project Developers, NVIDIA. "Adding sensors on a robot — Isaac Lab Tutorials." 2026. https://isaac-sim.github.io/IsaacLab/main/source/tutorials/04_sensors/add_sensors_on_robot.html

[5] Isaac Lab Project Developers, NVIDIA. "Release Notes — Isaac Lab 2.3.0" (built on Isaac Sim 5.1; Crazyflie asset path migration). 2025–2026. https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html

[6] NVIDIA. "Isaac Sim 5.1 Documentation — Isaac Lab Tutorials." 2025–2026. https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html

[7] Jiayu Chen, Chao Yu, et al., Tsinghua University. "Multi-UAV Pursuit-Evasion with Online Planning in Unknown Environments by Deep Reinforcement Learning." arXiv:2409.15866, 2024. https://arxiv.org/abs/2409.15866 (code: https://github.com/thu-uav/Multi-UAV-pursuit-evasion)

[8] Botian Xu et al. "OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control" (framework docs). https://omnidrones.readthedocs.io/en/latest/

[9] Ultralytics. "YOLOv8 / YOLO Documentation — Train, Predict, Export (ONNX)." https://docs.ultralytics.com/

[10] Y. Zhang et al. "AirSwarm: Enabling Cost-Effective Multi-UAV Research with COTS Drones." arXiv:2503.06890, 2025. — Tello video latency measured at 99.3–218.5 ms (mean 174.5, s.d. 37.0) at 720p/30fps H.264, and command latency at 25.9 ms mean. https://arxiv.org/abs/2503.06890

[11] "UAV Control with Vision-based Hand Gesture Recognition over Edge-Computing." arXiv:2505.17303, 2025. — Independent Tello measurement of 80–120 ms video latency and ~150 ms end-to-end sensing-to-movement. https://arxiv.org/abs/2505.17303

[12] Toni-SM et al. "skrl — Multi-agent API Documentation (IPPO, MAPPO)." https://skrl.readthedocs.io/en/latest/api/multi_agents.html

---

*Built for your learning style: big picture → detail, one hard thing at a time, checkpoints before commitments, and every tool proven compatible before you bet hours on it. Good hunting.* 🛩️

