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

## How this tutorial is organised: seven blocks

The project is seven self-contained pieces of work. Each block below produces one finished artifact, and the last block combines them. The block titles say what you do; the chapters inside them say how.

| Block | What you do in it | Chapters | Drone in hand? | What comes out |
|---|---|---|---|---|
| **A** | Install and isolate the software, then measure the real drone | 0–1 | 🔌 **YES — in 1.4 only** | Two working conda environments, three hardware measurements |
| **B** | Build the two-drone chase scene in Isaac Sim | 2 | 💻 No | A rendering arena with a defender and a moving attacker |
| **C** | Train the reinforcement learning policy that flies the chase | 3 | 💻 No — but needs the 1.4 numbers | `best_agent.pt` — a checkpoint that intercepts |
| **D** | Generate labelled synthetic images of a drone | 4 | 💻 No | ~2500 images with YOLO-format labels |
| **E** | Train the object detection model | 5 | 💻 No | `drone_detector.onnx` |
| **F** | Connect detector to policy and run the whole loop in simulation | 6 | 💻 No | A chase driven entirely by rendered camera frames |
| **G** | Put both models on the real drone and fly | 7 | 🔌 **YES — every subchapter** | `policy.onnx` plus recorded real flights |

**The two hardware markers used throughout this tutorial:**

| Marker | Meaning |
|---|---|
| 🔌 **DRONE HARDWARE REQUIRED** | You cannot complete this part without the Tello (and, from 7.3, the target drone) physically in front of you |
| 💻 **NO HARDWARE — simulation only** | Runs entirely on your laptop; the drone can stay in its box |

Blocks C and D/E are independent of each other — if a training run is cooking overnight, start Block D in a second terminal. Block F is the first point that needs both.

---

# ██ BLOCK A — Install the Software Stack and Measure the Real Drone ██

> 🔌 **DRONE HARDWARE REQUIRED — in subchapter 1.4 only.** Chapters 0, 1.0, 1.1, 1.2 and 1.3 are 💻 simulation and setup only.

<details>
<summary>Expand Block A</summary>

**What this block produces:** two conda environments that provably work together, a pinned Isaac Lab commit you can return to, and three measurements taken from the Tello (control rate, video delay, available telemetry) that Chapter 3 builds the policy around. Nothing here trains anything or flies a chase — it removes the two failure classes that would otherwise surface later as unexplainable bugs: package conflicts, and a policy designed for a drone that behaves differently.

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

# Chapter 1 — Create Virtual Environments and Perform Compatibility Tests Among Libraries

## 1.0 Isolate the project before installing anything (≤1.5h)

<details> 

<summary>Expand: </summary>

> **What / Why / How it contributes:** This chapter installs the necessary libraries and makes sure that they are compatible among themselves, inside the same environments. It also sets up three protections first: a private copy of the conda environment, a pinned Isaac Lab commit, and a frozen package list. So that if something breaks, we can easily restore the initial state. 

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

<details>
     <summary>Expand</summary>

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
</details>

## 1.3 Build the vision env and test every tool boundary (≤1.5h)

> **What / Why / How it contributes:** We create the second conda env (`drone_vision`) that will train YOLOv8 in Chapter 5, and — critically — we run ALL the cross-tool compatibility tests NOW, before investing hours in data generation. We prove: (1) Ultralytics trains on your GPU, (2) a YOLO model exports to ONNX, (3) that ONNX file runs inside `env_drone` via onnxruntime. If the full round trip works with a toy model today, it will work with your real model in Chapter 6.

<details>
     <summary>Expand</summary>

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

This test opens the ONNX file from the *other* environment and runs one frame of random numbers through it. It is required because:

You have two separate Python installations on your machine — `drone_vision` and `env_drone`. Each has its own copy of every library.

They're separate because they need conflicting versions. `env_drone` runs Isaac Lab, which requires torch 2.7. `drone_vision` runs Ultralytics, which installed torch 2.11. If both lived in one environment, pip would install a single torch, and whichever version it picked would break the other tool.

That creates a problem. Chapter 5 trains the detector in `drone_vision`. Chapter 6 needs that detector running *inside the Isaac Lab simulation*, in `env_drone` — it looks at each rendered camera frame and finds the attacker. So the detector has to get from one environment to the other, and you can't just install Ultralytics on the other side.

ONNX is the answer. It's a file format that stores the trained model with nothing framework-specific attached. `onnxruntime` opens it, and `onnxruntime` needs no torch — which is why adding it to `env_drone` is safe.

**This test confirms that path works**, using a throwaway pre-trained model rather than your own. It loads the file `drone_vision` created, feeds it one frame of random numbers, and checks something sensible comes back.


Create a Notepad file with a script to run this test:

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

Basically, what this whole script does it: it takes a dummy input `images [1, 3, 640, 640]` in the expected shape, passes it through the model and verifies that the model produces an output `output: (1, 84, 8400)`  in the expected shape
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
</details>

## 1.4 Set up the Tello and measure it (≤1.5h) - DRONE HARDWARE NEEDED

> **What / Why / How it contributes:** You measure the real drone **before** designing the policy, not after. Three numbers come out of this subchapter — how late its video arrives, how fast it accepts commands, and which telemetry it can report — and all three go straight into Chapter 3. Measuring first means you train once. Measuring afterwards would mean training, discovering a mismatch, and training again.

<details>
     <summary>Expand</summary>

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
</details>

---

</details>

---

# ██ BLOCK B — Build the Two-Drone Chase Scene in Isaac Sim ██

> 💻 **NO HARDWARE — simulation only.** The drone can stay in its box for the whole of this block.

<details>
<summary>Expand Block B</summary>

**What this block produces:** an Isaac Lab environment containing two aircraft per arena — a defender flown by physics forces, and an attacker whose position you write directly along a circular path — cloned across every parallel environment and confirmed by eye in the viewport. No reward function, no observations about the attacker, and no training happens here. The only artifact is a scene that renders correctly, which is what Block C then attaches a reward and a policy to.

**Why this comes before the policy.** A reward function that reads `self._attacker.data.root_pos_w` cannot be debugged until an attacker exists and moves predictably. Building the scene first means that when Chapter 3's training misbehaves, the scene is already known-good and only the reward or the observations are in question.

**Every file edited in this block is the same one:**

```
C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py
```

---

# Chapter 2 — Add a Second Drone to the Isaac Lab Task and Give It a Flight Path

> 💻 **NO HARDWARE — simulation only.**

<details>
<summary>Expand Chapter 2</summary>

> Edits the template files in Isaaclab's custom drone task to include an attacker drone and makes it move in a randomized way. Then, we render the scene on IsaacSim — just the simulation, no Reinforcement Learning runs or rewards calculation yet.

## 2.1 Spawn the Attacker Drone in Every Parallel Environment (≤1.5h)

<details>
<summary>Expand 2.1</summary>

> **What this subchapter does:** your task file currently spawns one Crazyflie, the defender, and gives it a fixed hover goal. A pursuit needs a second aircraft. This subchapter adds an attacker to the config class, creates and registers it in `_setup_scene`, holds it in place by writing its pose every step, and resets it alongside the defender. At the end you render 16 environments and confirm both drones appear in each. Chapter 2.2 replaces the fixed pose with motion, and Chapter 3 reads the attacker's position to compute observations and reward.

### How two robots coexist in one Direct-workflow environment

You already know the pattern for one robot: an `ArticulationCfg` in the env cfg, instantiated in `_setup_scene`, registered in `self.scene.articulations`. Two robots is the same pattern twice, with two different prim paths under each env namespace:

```
/World/envs/env_0/
   ├── Robot     ← defender (physics + external forces from policy)
   └── Attacker  ← attacker (we write its pose every step)
/World/envs/env_1/
   ├── Robot
   └── Attacker
   ...
```

`clone_environments` replicates everything under `env_.*`, so the attacker is copied into all 2048 environments by the same mechanism that already copies the defender and the goal markers. You write the attacker once.

### Step 1 — Add the attacker's spawn recipe and the pursuit constants to the config class

<details>
<summary>Expand Step 1</summary>

> **Environment:** none needed — you are editing files.

The config class currently describes one drone. This step adds a second `ArticulationCfg` that reuses `CRAZYFLIE_CFG` — the same spawn recipe as the defender, since a Crazyflie is the only quadcopter asset Isaac Lab ships — changing only the prim path and the starting position. It also adds three plain numbers the pursuit needs: how close counts as a capture, how far the defender may stray, and how fast the attacker flies. Chapters 3.1 and 3.2 read all three.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: the imports at the very top of the file ────────────────────────

from isaaclab.assets import Articulation, ArticulationCfg     # ← ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg         #   may already be
from isaaclab.utils import configclass                        #   imported; if so,
from isaaclab_assets import CRAZYFLIE_CFG                     #   leave as-is
# ... the rest of the existing imports, unchanged ...


# ── SECTION: the config class, roughly 30 lines down ────────────────────────

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # --- EXISTING CODE (leave everything above and below untouched) ---
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 12
    # ... sim cfg, scene cfg, reward scales, etc. ...

    # defender — EXISTING LINE, unchanged
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ▼▼▼ INSERT HERE! — add these lines directly BELOW the `robot:` line ▼▼▼

    # attacker — same drone asset, different prim path, spawned 4 m away at 1.5 m
    attacker: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Attacker",
        init_state=ArticulationCfg.InitialStateCfg(pos=(4.0, 0.0, 1.5)),
    )

    # pursuit geometry knobs (plain attributes, like the reward scales you know)
    capture_radius = 0.35        # meters — "caught" if closer than this
    arena_radius = 8.0           # meters — episode fails if defender strays this far
    attacker_speed = 0.6         # m/s along its path (tuned in Ch. 3.3)

    # ▲▲▲ END OF INSERT — the existing reward-scale lines continue below ▲▲▲

    lin_vel_reward_scale = -0.05          # ← EXISTING, unchanged
    ang_vel_reward_scale = -0.01          # ← EXISTING, unchanged
```

**Why 0.35 m for the capture radius.** A Crazyflie is about 9 cm rotor to rotor, so 0.35 m means the two airframes roughly overlap. Requiring actual mesh contact would make the capture bonus in 3.2 fire so rarely during early training that the policy would never learn what earns it.

**Also raise `env_spacing`.** In the same config class, find the scene cfg line and change one number:

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: inside QuadcopterEnvCfg, the scene line ────────────────────────

    # BEFORE:
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, ...)

    # AFTER — env_spacing must be at least 2 * arena_radius:
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=16.0,
                                                     replicate_physics=True)
```

With smaller spacing, a neighbouring environment's drones sit inside this environment's arena volume, and the camera you attach in 5.2 would photograph them.

</details>

### Step 2 — Create and register the attacker in `_setup_scene`

<details>
<summary>Expand Step 2</summary>

> **Environment:** none needed — you are editing files.

`_setup_scene` builds the world once before simulation starts. The config from Step 1 is only a description; this step turns it into a live object and adds it to `self.scene.articulations`. Registration is what makes the scene refresh the attacker's position and velocity buffers each step — without it the drone renders but `root_pos_w` never changes, and Chapter 3's reward would read a frozen number with no error.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method _setup_scene ───────────────────────

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)                    # ← EXISTING

        # ▼▼▼ INSERT HERE! — one line, directly below the defender ▼▼▼
        self._attacker = Articulation(self.cfg.attacker)
        # ▲▲▲ END OF INSERT ▲▲▲

        self.scene.articulations["robot"] = self._robot               # ← EXISTING

        # ▼▼▼ INSERT HERE! — one line, directly below the defender's registration ▼▼▼
        self.scene.articulations["attacker"] = self._attacker
        # ▲▲▲ END OF INSERT ▲▲▲

        # --- EXISTING CODE BELOW, unchanged: ground plane, clone_environments, lights ---
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        # ... lights ...
```

⚠️ Both inserted lines must appear **before** `clone_environments`, because cloning copies whatever exists under `env_.*` at the moment it runs.

</details>

### Step 3 — Hold the attacker in place by writing its pose every step

<details>
<summary>Expand Step 3</summary>

> **Environment:** none needed — you are editing files.

The attacker is a physics body with nothing driving it, so PhysX will drop it to the floor. Until 2.2 gives it a path, this step pins it at its spawn point by overwriting its root pose and zeroing its velocity on every step. It is temporary scaffolding, but it lets you confirm spawning and cloning work before motion adds a second thing that could be wrong.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method _apply_action ──────────────────────

    def _apply_action(self):
        # EXISTING LINE — the defender's thrust and moments, unchanged
        self._robot.set_external_force_and_torque(
            self._thrust, self._moment, body_ids=self._body_id
        )

        # ▼▼▼ INSERT HERE! — everything below, at the END of the method ▼▼▼
        # attacker: hold pose. TEMPORARY — replaced by _move_attacker() in 2.2
        hold = self._attacker.data.default_root_state.clone()
        hold[:, :3] += self.scene.env_origins            # local spawn pos → world coords
        self._attacker.write_root_pose_to_sim(hold[:, :7])
        self._attacker.write_root_velocity_to_sim(torch.zeros_like(hold[:, 7:]))
        # ▲▲▲ END OF INSERT — nothing else in this method ▲▲▲
```

**Flag — this is different from the control you know.** The defender is *dynamic*: you apply forces and PhysX integrates them into motion. The attacker is *kinematic* (from Greek *kinema*, "motion" — describing motion without the forces producing it): you write its pose directly, and PhysX never computes gravity or collision response for it. The trade-off is that a kinematic attacker is not pushed around on contact. That costs nothing here, because "capture" in 3.2 is a distance comparison, not a collision event.

</details>

### Step 4 — Return the attacker to its spawn point on every reset

<details>
<summary>Expand Step 4</summary>

> **Environment:** none needed — you are editing files.

`_reset_idx` restarts only the environments whose episode just ended. The defender is already reset there; the attacker needs the same treatment or it will begin the new episode at whatever pose the last one left it in, which makes the starting distance differ unpredictably between episodes.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method _reset_idx ─────────────────────────

    def _reset_idx(self, env_ids):
        # --- EXISTING CODE, unchanged: logging, super()._reset_idx, goal resampling ---
        # ... existing defender reset, which ends with lines like these: ---
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # ▼▼▼ INSERT HERE! — mirror the four lines above, for the attacker ▼▼▼
        a_state = self._attacker.data.default_root_state[env_ids].clone()
        a_state[:, :3] += self.scene.env_origins[env_ids]
        self._attacker.write_root_pose_to_sim(a_state[:, :7], env_ids)
        self._attacker.write_root_velocity_to_sim(a_state[:, 7:], env_ids)
        # ▲▲▲ END OF INSERT ▲▲▲
```

Every line indexes by `env_ids`. Writing the whole table instead would teleport the environments still mid-flight back to their start.

</details>

### Step 5 — Render 16 environments and confirm both drones appear

<details>
<summary>Expand Step 5</summary>

> **Environment:** `env_drone`

Nothing has been trained, so there is nothing to play back. The quickest way to see the scene is to run training for a few iterations *without* `--headless`, which opens the viewport while an untrained policy emits random actions. What you are checking is that the attacker exists in every cloned environment and that the defender's force pipeline still works.

*Run from:* `any folder` — *the script lives in:* `C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\train.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 16
```

You should see 16 arenas, each with a jittering defender (random policy) and an attacker hovering frozen at (4, 0, 1.5).

</details>

> ✅ **Checkpoint 2.1**
> 1. No spawn errors across 16 envs
> 2. Attacker visibly present and motionless at its spawn point in every env
> 3. Defender still jitters under the (untrained) policy — proving you didn't break its force pipeline

</details>

---

## 2.2 Move the Attacker Along a Randomised Circular Path (≤1.5h)

<details>
<summary>Expand 2.2</summary>

> **What this subchapter does:** a frozen attacker turns the pursuit into the hover task with a different goal position — the policy would learn to fly to one point. This subchapter replaces the held pose from 2.1 with a parametric path (a horizontal circle plus a vertical bob), and randomises each environment's starting angle and direction of travel so no two chases are identical. The path's speed, `attacker_speed`, becomes the difficulty dial that Chapter 3.3's curriculum turns from 0.3 to 1.0 m/s.

### Why a scripted path rather than a second RL agent

Three ways to drive an attacker, in order of cost:

1. **Parametric path** (sin/cos loop) — deterministic, tunable by one number, needs no training. ← **what we use**
2. Reactive script (flee from the defender) — a harder target, but early in training a random defender and a fleeing attacker can lock into loops that produce no useful experience.
3. RL attacker (adversarial / MARL) — a research-scale problem. Chen, Yu et al. (Tsinghua, arXiv:2409.15866) do this with MAPPO plus a generated curriculum.

The rule the whole tutorial follows is **change one hard thing at a time**. The hard thing in Chapter 3 is the pursuit reward, so the attacker stays simple until that works.

### Step 1 — Understand the path the attacker will fly

<details>
<summary>Expand Step 1</summary>

> **Environment:** none needed — this step is explanation only, nothing is edited.

The path is a point travelling around a horizontal circle of radius 3 m while rising and falling by 0.4 m. Two lines of arithmetic produce it, and one constant — `attacker_speed` — controls how hard the chase is.

```
      TOP VIEW                        SIDE VIEW
   .──────────.                    z
  /            \                 1.9 ┈╭─╮┈┈┈┈╭─╮┈    ← bobbing ±0.4 m
 │      ●──────│──→ x           1.5 ─┤  ╰────╯  ├─
  \     center /                 1.1 ┈┈┈┈┈┈┈┈┈┈┈┈
   '──────────'
   R = 3 m circle                horizontal loop + vertical wave
```

`sin` and `cos` here are just the x and y coordinates of a point walking around a circle — that is the whole of the maths involved. This is pseudocode for understanding, not code to paste:

```
theta = phase + direction * (attacker_speed / R) * t     # angle grows over time
x = R * cos(theta);  y = R * sin(theta)                  # the circle
z = 1.5 + 0.4 * sin(0.7 * t + phase)                     # the bob
```

`attacker_speed / R` converts metres per second along the path into radians per second, so raising `attacker_speed` from 0.3 to 0.6 in Chapter 3.3 genuinely doubles the target's ground speed rather than changing an abstract rate.

</details>

### Step 2 — Write the trajectory method and randomise it per environment

<details>
<summary>Expand Step 2</summary>

> **Environment:** none needed — you are editing files.

This step replaces the "hold pose" block from 2.1 with a new method, `_move_attacker`, which computes a new position each physics tick and writes it to the simulator. It also allocates three per-environment buffers — starting angle, direction of travel, and an episode clock — and re-randomises the first two on every reset. Without that randomisation, every environment presents the identical chase and the policy can memorise one manoeuvre sequence instead of learning to intercept a moving target.

Three separate edits to the same file follow. Make them in order.

**Edit 1 of 3 — allocate the buffers in `__init__`:**

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method __init__ ───────────────────────────
# ── (at the top of the file, add `import math` if it is not already there)  ──

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- EXISTING CODE, unchanged: action/thrust/moment buffers, logging dict ---
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # ▼▼▼ INSERT HERE! — attacker trajectory state, one row per environment ▼▼▼
        self._atk_phase = torch.zeros(self.num_envs, device=self.device)   # start angle
        self._atk_dir = torch.ones(self.num_envs, device=self.device)      # +1 or -1 (CW/CCW)
        self._atk_t = torch.zeros(self.num_envs, device=self.device)       # per-env clock
        self._atk_radius = 3.0                                             # metres
        # ▲▲▲ END OF INSERT ▲▲▲

        # --- EXISTING CODE BELOW, unchanged: self._body_id, self._robot_mass, etc. ---
```

**Edit 2 of 3 — add the new method and call it from `_apply_action`:**

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method _apply_action ──────────────────────

    def _apply_action(self):
        self._robot.set_external_force_and_torque(          # ← EXISTING, unchanged
            self._thrust, self._moment, body_ids=self._body_id
        )

        # ▼▼▼ DELETE the four "hold pose" lines you added in 2.1 Step 3: ▼▼▼
        #   hold = self._attacker.data.default_root_state.clone()
        #   hold[:, :3] += self.scene.env_origins
        #   self._attacker.write_root_pose_to_sim(hold[:, :7])
        #   self._attacker.write_root_velocity_to_sim(torch.zeros_like(hold[:, 7:]))
        # ▲▲▲ and REPLACE them with this single call: ▲▲▲
        self._move_attacker()


    # ▼▼▼ INSERT HERE! — a brand-new method, directly BELOW _apply_action ▼▼▼
    def _move_attacker(self):
        """Kinematic attacker: circle + vertical bob, written to sim each physics tick."""
        self._atk_t += self.physics_dt
        w = self.cfg.attacker_speed / self._atk_radius            # m/s → rad/s
        theta = self._atk_phase + self._atk_dir * w * self._atk_t

        pos = torch.zeros(self.num_envs, 3, device=self.device)
        pos[:, 0] = self._atk_radius * torch.cos(theta)
        pos[:, 1] = self._atk_radius * torch.sin(theta)
        pos[:, 2] = 1.5 + 0.4 * torch.sin(0.7 * self._atk_t + self._atk_phase)
        pos += self.scene.env_origins                             # local → world

        pose = self._attacker.data.root_pose_w.clone()
        pose[:, :3] = pos                                         # keep orientation as-is
        self._attacker.write_root_pose_to_sim(pose)

        # attacker velocity by finite difference — Chapter 3 uses it
        if not hasattr(self, "_atk_prev_pos"):
            self._atk_prev_pos = pos.clone()
        self._atk_vel = (pos - self._atk_prev_pos) / self.physics_dt
        self._atk_prev_pos = pos.clone()
    # ▲▲▲ END OF INSERT ▲▲▲


    def _get_observations(self) -> dict:      # ← EXISTING method, continues below
        ...
```

**Edit 3 of 3 — re-randomise the chase on reset:**

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method _reset_idx ─────────────────────────

    def _reset_idx(self, env_ids):
        # --- EXISTING CODE, unchanged, including the attacker reset from 2.1 Step 4 ---
        self._attacker.write_root_velocity_to_sim(a_state[:, 7:], env_ids)

        # ▼▼▼ INSERT HERE! — directly below the attacker pose reset ▼▼▼
        n = len(env_ids)
        self._atk_phase[env_ids] = torch.rand(n, device=self.device) * 2 * math.pi
        self._atk_dir[env_ids] = torch.where(
            torch.rand(n, device=self.device) > 0.5, 1.0, -1.0
        )
        self._atk_t[env_ids] = 0.0
        # ▲▲▲ END OF INSERT ▲▲▲
```

Every quantity here is a batched torch tensor covering all environments at once — no Python loop over envs. Every reward and observation function you write in Chapter 3 follows the same rule, because a per-env loop at 2048 environments would dominate step time.

This is domain randomisation applied to behaviour rather than appearance — the same reasoning that made you randomise lighting and textures in the pallet-jack SDG pipeline, moved from what the scene looks like to what the target does.

</details>

### Step 3 — Render the scene and confirm the attackers orbit

<details>
<summary>Expand Step 3</summary>

> **Environment:** `env_drone`

Run 16 environments without `--headless` again. Motion has to be smooth here because Chapter 3.1 computes the attacker's apparent size and bearing from this position each step, and a stuttering path would produce jumpy readings that the policy learns to distrust.

*Run from:* `any folder` — *the script lives in:* `C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\train.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 16
```

You should see attackers sweeping circles at different starting angles and in both directions, while the defenders jitter under the random policy.

</details>

> ✅ **Checkpoint 2.2**
> 1. Attackers orbit smoothly — if the motion stutters, confirm `_move_attacker` is called from `_apply_action` (physics rate) and not from `_get_observations` (env-step rate)
> 2. Different envs show different phases and directions
> 3. After episodes time out, the trajectories re-randomise
> 4. Commit to git: "attacker added and moving — scene complete"

</details>

</details>

</details>

---
# ██ BLOCK C — Train the Reinforcement Learning Policy That Flies the Chase ██

> 💻 **NO HARDWARE — simulation only.** The drone stays in its box, but this block cannot start until the three measurements from 🔌 subchapter 1.4 are written in `project_notes.txt`.

<details>
<summary>Expand Block C</summary>

**What this block produces:** `best_agent.pt`, a skrl checkpoint holding a neural network that takes 17 numbers describing the defender's own motion and what its camera reports, and emits 4 stick commands that close on the attacker. It also produces one measured constant, `capture_ang_size`, which Block F needs to declare a capture without a simulator.

**What it does not use:** rendered images. Every observation in this block is computed arithmetically from ground-truth positions, using the same equations a camera would obey. That is what keeps training as fast as the plain hover task while producing a policy that runs unchanged on real camera data in Blocks F and G.

**Every file edited in this block is the same one:**

```
C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py
```

---

# Chapter 3 — Define the Policy's Inputs and Outputs, Write the Reward, and Train with PPO

> 💻 **NO HARDWARE — simulation only**, but every number in 3.1 comes from the 🔌 1.4 measurements.

<details>
<summary>Expand Chapter 3</summary>

## 3.1 Define the Four Commands the Policy Sends and the Seventeen Numbers It Receives (≤1.5h)

<details>
<summary>Expand 3.1</summary>

> **What this subchapter does:** a policy is defined by two interfaces — what it emits and what it is given. This subchapter sets both, using the three measurements you took from the Tello in 1.4. The four outputs become normalised stick commands instead of forces; the seventeen inputs contain only quantities a Tello can actually supply, arrive delayed by the video latency you measured, and describe the attacker purely in camera terms. Getting these right now is what lets the checkpoint you train in 3.3 fly the real drone in Chapter 7 without a second training run.

> **Environment for Parts A to G:** none needed — they are all edits to `quadcopter_env.py`. You run nothing until the sanity check at the end of 3.1.

### Part A — Replace force outputs with the four stick commands a Tello accepts

<details>
<summary>Expand Part A</summary>

Isaac Lab's hover task outputs one thrust and three torques. **A Tello does not accept forces.** It accepts four normalised channels and runs its own stabiliser underneath:

```
   channel 0   forward / backward
   channel 1   left / right
   channel 2   up / down
   channel 3   turn (yaw rate)
```

That built-in stabiliser is a large amount of balancing work the policy no longer has to learn — and a large amount of behaviour the simulation must now imitate, because the policy will be trained against whatever the simulation does.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method _pre_physics_step ──────────────────

    def _pre_physics_step(self, actions: torch.Tensor):

        # ▼▼▼ DELETE the hover task's force conversion — these three lines: ▼▼▼
        #   self._actions = actions.clone().clamp(-1.0, 1.0)
        #   self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        #   self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        # ▲▲▲ and REPLACE the whole method body with the code below ▲▲▲

        # ▼▼▼ INSERT HERE! ▼▼▼
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
        # ▲▲▲ END OF INSERT ▲▲▲
```

`quat_apply` comes from `isaaclab.utils.math` — add it to the imports at the top of the file if it is not already there.

The four constants it reads go in the config class:

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnvCfg, next to the 2.1 pursuit knobs ──────────

    capture_radius = 0.35        # ← EXISTING, from 2.1
    arena_radius = 8.0           # ← EXISTING, from 2.1
    attacker_speed = 0.6         # ← EXISTING, from 2.1

    # ▼▼▼ INSERT HERE! — the stick-command model ▼▼▼
    max_speed = 2.0          # m/s — conservative; a Tello can do more
    max_yaw_rate = 1.5       # rad/s
    vel_gain = 3.0           # how hard the stand-in stabiliser corrects
    yaw_gain = 0.05
    # ▲▲▲ END OF INSERT ▲▲▲
```

**Why approximate rather than model the real stabiliser.** The Tello's control law is undocumented, so no amount of effort reproduces it exactly. What must match is the *interface* — four normalised numbers in, roughly velocity-like behaviour out. Section 3.3's randomisation of mass, thrust and drift covers the remaining gap, and 7.4 narrows `vel_gain` using real flight logs.

**After this change the simulated airframe stops being a Crazyflie** in any meaningful sense. It is a generic hovering body that responds to velocity commands, which is what a Tello is from your code's point of view.

</details>

### Part B — Set the control rate to the value you measured in 1.4

<details>
<summary>Expand Part B</summary>

From 1.4, Step 6. If you measured 20 commands per second:

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnvCfg, the first few lines of the class ───────

    episode_length_s = 10.0

    # BEFORE:  decimation = 2      ← the hover task's value, gives 50 Hz control
    # AFTER:   sim.dt = 1/100, so decimation 5 gives 20 Hz — YOUR measured rate
    decimation = 5               # ◄── CHANGE THIS ONE NUMBER
```

**This cannot be skipped.** At 20 Hz each command persists two and a half times longer than at 50 Hz, so identical policy output produces much larger movement. A policy trained at 50 Hz and flown at 20 Hz overshoots consistently.

</details>

### Part C — List the seventeen numbers the policy receives

<details>
<summary>Expand Part C</summary>

Every entry below is something a Tello can supply in flight, or something computed from the camera. That constraint is what makes the same seventeen assemblable in Chapter 7.2 from real hardware, with the policy unchanged.

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

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnvCfg, near decimation ────────────────────────

    action_space = 4             # ← EXISTING, unchanged

    # BEFORE:  observation_space = 12     ← the hover task's twelve
    observation_space = 17       # ◄── CHANGE THIS ONE NUMBER
```

**Three things are absent, each for a reason from 1.4:**

**No angular rates.** A Tello reports attitude angles but not how fast they are changing. Training on a number the drone cannot supply would produce a policy that fails on hardware in a way you could not diagnose. The policy can fly without them: rotating the drone sweeps the attacker across the frame, so `d_bearing_x` already reveals rotation.

**No positions in metres, anywhere.** Recovering metres from a photograph requires knowing the attacker's true width. Bearings and frame-share make no such claim, so a target drone of different size changes nothing.

**No instantaneous readings.** They arrive late, on purpose — see Part D.

</details>

### Part D — Delay the camera readings and feed the previous command back

<details>
<summary>Expand Part D</summary>

**The delay.** A Tello's video reaches your laptop 99–219 ms after it was captured. At 20 Hz control that is 2 to 5 decisions of staleness. A policy trained on instant readings has no reason to account for this, and on hardware it oscillates: it steers toward where the target was, finds it has moved, over-corrects, repeats. Training with the delay present teaches the policy to **lead** the target rather than track it.

Three edits to the same file.

**Edit 1 of 3 — the two constants, from your own measurement:**

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnvCfg, below the Part A stick-command block ───

    yaw_gain = 0.05              # ← EXISTING, from Part A

    # ▼▼▼ INSERT HERE! — from project_notes.txt, converted to control steps ▼▼▼
    obs_delay_min = 2
    obs_delay_max = 5
    # ▲▲▲ END OF INSERT ▲▲▲
```

**Edit 2 of 3 — the history buffer and the per-env delay, in `__init__`:**

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method __init__ ───────────────────────────

        self._atk_radius = 3.0                    # ← EXISTING, from 2.2 Step 2

        # ▼▼▼ INSERT HERE! — delayed-reading machinery ▼▼▼
        # one row per env, one column per past step, 7 readings each
        self._reading_history = torch.zeros(
            self.num_envs, self.cfg.obs_delay_max + 1, 7, device=self.device
        )
        # each env draws its own lag, so the policy meets the whole range
        self._delay_steps = torch.randint(
            self.cfg.obs_delay_min, self.cfg.obs_delay_max + 1,
            (self.num_envs,), device=self.device
        )
        # last-seen readings, held while the attacker is out of view
        self._prev_bx = torch.zeros(self.num_envs, device=self.device)
        self._prev_by = torch.zeros(self.num_envs, device=self.device)
        self._prev_asz = torch.zeros(self.num_envs, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 4, device=self.device)
        # ▲▲▲ END OF INSERT ▲▲▲
```

**Edit 3 of 3 — the method that pushes fresh readings in and takes delayed ones out:**

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv — a new method, place it directly ──────────
# ──          ABOVE _get_observations                                ────────

    # ▼▼▼ INSERT HERE! ▼▼▼
    def _delayed_readings(self, fresh):          # fresh: (num_envs, 7)
        """Shift the history one step, store the newest, return each env's delayed row."""
        self._reading_history = torch.roll(self._reading_history, shifts=1, dims=1)
        self._reading_history[:, 0] = fresh
        idx = self._delay_steps.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 7)
        return self._reading_history.gather(1, idx).squeeze(1)
    # ▲▲▲ END OF INSERT ▲▲▲
```

And in `_reset_idx`, alongside the 2.2 randomisation, clear the history and draw a new lag:

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method _reset_idx ─────────────────────────

        self._atk_t[env_ids] = 0.0               # ← EXISTING, from 2.2 Step 2

        # ▼▼▼ INSERT HERE! ▼▼▼
        self._reading_history[env_ids] = 0.0
        self._delay_steps[env_ids] = torch.randint(
            self.cfg.obs_delay_min, self.cfg.obs_delay_max + 1,
            (len(env_ids),), device=self.device
        )
        self._prev_bx[env_ids] = 0.0
        self._prev_by[env_ids] = 0.0
        self._prev_asz[env_ids] = 0.0
        # ▲▲▲ END OF INSERT ▲▲▲
```

**Why the delay is randomised rather than fixed.** A constant lag can be cancelled exactly, and a policy trained on one would be tuned to a number it never sees twice — the measured spread on a Tello is roughly ±37 ms within a single flight. Drawing a fresh delay per episode from 2 to 5 steps produces tolerance across the range instead.

**The previous action.** With delayed readings, commands issued in the last few steps have not yet appeared in anything the policy can see. Feeding back what it just commanded lets it account for those instead of issuing them again. Four numbers, standard practice for delayed control.

</details>

### Part E — Match the simulated camera to the Tello's lens

<details>
<summary>Expand Part E</summary>

Bearings are measured as a fraction of the frame, so what `bearing_x = +0.5` means in degrees depends entirely on the lens. This step sets the simulated camera's focal length and frame shape to the Tello's, so that fraction means the same angle in training and in flight.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnvCfg, below the Part D delay constants ───────

    obs_delay_max = 5            # ← EXISTING, from Part D

    # ▼▼▼ INSERT HERE! — camera model. MUST match the TiledCameraCfg you add ▼▼▼
    # in 5.2 AND the real drone's lens.
    cam_width = 640
    cam_height = 480                  # 4:3, matching the Tello's 960x720
    cam_focal_mm = 12.0               # gives ~83 deg horizontal field of view
    cam_aperture_mm = 20.955          # PinholeCameraCfg default horizontal aperture
    attacker_span_m = 0.13            # target's real width — ONLY to simulate the camera
    # ▲▲▲ END OF INSERT ▲▲▲
```

**Why 12 mm and not Isaac Lab's default 24 mm.** Isaac Lab's default gives roughly 47° of horizontal view; a Tello sees about 83°. Train on 47° and fly on 83° and every bearing corresponds to nearly twice the angle it did in training, so the defender under-steers on every correction — and nothing errors.

For a different camera, solve:

```
focal_mm = aperture_mm / (2 x tan(FOV / 2))
         = 20.955 / (2 x tan(41.3 deg)) = 12.0     for an 83 deg lens
```

**Aspect ratio too.** A Tello outputs 4:3. Rendering square and then squashing a 4:3 photograph into it would stretch every horizontal bearing by 4/3. Render 640×480, resize real frames to 640×480, and the geometry stays consistent.

**This is the most likely silent failure in the whole project.** Chapter 6.1 Step 2 exists specifically to catch it, by comparing these numbers against ones measured from a rendered frame.

</details>

### Part F — Compute the seven readings from ground truth during training

<details>
<summary>Expand Part F</summary>

During training the seven readings are computed from the true relative position, run *forward* through the camera model: given where the attacker really is, how large would it appear and where in frame? This is legitimate — the simulated camera really is photographing a target of known size, and rendering the frame would produce the same rectangle far more slowly. What never happens anywhere in this project is the *backward* step: taking a rectangle and dividing by an assumed width to produce metres.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: the imports at the top of the file ─────────────────────────────

# ▼▼▼ INSERT HERE! (if not already imported) ▼▼▼
from isaaclab.utils.math import subtract_frame_transforms, quat_apply
# ▲▲▲ END OF INSERT ▲▲▲


# ── SECTION: class QuadcopterEnv — a new method, place it directly BELOW ────
# ──          _delayed_readings and ABOVE _get_observations             ─────

    # ▼▼▼ INSERT HERE! ▼▼▼
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

        # the detector's blind spot: a box under ~8 px wide usually returns nothing
        visible = (fwd > 0.05) & (bearing_x.abs() < 1.0) & (bearing_y.abs() < 1.0) & (ang_size > 0.012)
        return bearing_x, bearing_y, ang_size, visible.float()
    # ▲▲▲ END OF INSERT ▲▲▲
```

Simulating that blind spot means the policy meets the "too small to detect" case thousands of times in training and learns a response, instead of meeting it for the first time on a real flight.

</details>

### Part G — Assemble the observation vector in the order the policy will always see

<details>
<summary>Expand Part G</summary>

This step concatenates the seventeen numbers, holds the last camera reading whenever the attacker is not visible, and computes the three rate-of-change terms. Chapter 7.2's flight script builds the same seventeen in the same order from real telemetry, so any change to this ordering must be mirrored there.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method _get_observations ──────────────────

    def _get_observations(self) -> dict:

        # ▼▼▼ DELETE the hover task's whole body — it looked like this: ▼▼▼
        #   desired_pos_b, _ = subtract_frame_transforms(
        #       self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        #   )
        #   obs = torch.cat([self._robot.data.root_lin_vel_b,
        #                    self._robot.data.root_ang_vel_b,
        #                    self._robot.data.projected_gravity_b,
        #                    desired_pos_b], dim=-1)
        #   return {"policy": obs}
        # ▲▲▲ and REPLACE it with everything below ▲▲▲

        # ▼▼▼ INSERT HERE! ▼▼▼
        # cache the true distance — the reward (3.2) and _get_dones both read it
        self._dist = torch.linalg.norm(
            self._attacker.data.root_pos_w - self._robot.data.root_pos_w, dim=1
        )

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
            self._robot.data.root_lin_vel_b,          # 3   slots 0:3
            self._robot.data.projected_gravity_b,     # 3   slots 3:6
            delayed,                                  # 7   slots 6:13  ← camera
            self._actions,                            # 4   slots 13:17
        ], dim=-1)
        return {"policy": obs}
        # ▲▲▲ END OF INSERT ▲▲▲
```

`_prev_bx / _prev_by / _prev_asz` were allocated in Part D's Edit 2 and zeroed on reset in its final block. Without that zeroing, a fresh episode inherits the previous episode's last sighting and its first `d_ang_size` is a large meaningless jump.

</details>

### Sanity check before moving on

<details>
<summary>Expand the sanity check</summary>

> **Environment:** `env_drone`

Run a handful of non-headless iterations printing `bearing_x`, `ang_size`, `_dist` and `visible` for env 0. Three things must hold: **ang_size rises as `_dist` falls**, **bearing_x flips sign** as the attacker crosses the frame, and **the delayed readings lag the fresh ones** by the expected number of steps.

*Run from:* `any folder` — *the script lives in:* `C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\train.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 4 --max_iterations 5
```

</details>

> ✅ **Checkpoint 3.1**
> 1. Env steps with no shape errors at `observation_space = 17` and `decimation` matching your measured rate
> 2. Printed readings behave sensibly, and delayed values visibly lag
> 3. `visible` drops to 0 when the attacker leaves the frame, with held values staying frozen
> 4. Your cfg contains the three numbers from `project_notes.txt`, not the defaults printed here

</details>

---

## 3.2 Write the Reward Function and the Episode-Ending Conditions (≤1.5h)

<details>
<summary>Expand 3.2</summary>

> **What this subchapter does:** the policy becomes whatever the reward pays for, including any loophole in it. This subchapter builds a dense reward from three payments (closing speed, proximity, capture bonus) and two penalties (jerky commands, plus the hover task's existing stability terms), then defines the four events that end an episode. It is mostly reading and deciding rather than typing, because a reward bug is only visible after a full training run — the most expensive kind of mistake in this project.

### Why the reward may use the true distance when the observations may not

The observations changed in 3.1, but the reward does not have to. The reward is read only by the PPO update inside skrl; once training ends it is never called again — neither `play.py` nor the Chapter 6 demo evaluates it. So it may use the true distance between the drones even though the policy never receives that number.

```
 OBSERVATIONS ──► must be obtainable from a camera at deployment  (strict)
 REWARDS      ──► may use anything the simulator knows            (free)
```

Keeping the true metric is what lets the learning signal stay smooth while the policy still learns to act on camera readings alone.

### Step 1 — Add the reward dials to the config class

<details>
<summary>Expand Step 1</summary>

> **Environment:** none needed — you are editing files.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnvCfg, where the hover reward scales live ─────

    # --- EXISTING: keep these two, they fight jitter ---
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01

    # --- EXISTING: the hover task's distance reward is now unused ---
    # distance_to_goal_reward_scale = 15.0     ← comment out or delete

    # ▼▼▼ INSERT HERE! — the pursuit reward dials ▼▼▼
    closing_reward_scale = 2.0      # per m/s of speed TOWARD the target
    proximity_reward_scale = 1.5    # smooth "warmth" signal as distance shrinks
    capture_bonus = 200.0           # paid once, at capture
    crash_penalty = -50.0           # hit the floor / left the arena
    action_rate_penalty = -0.02     # sim-to-real: penalise jerky command changes
    # ▲▲▲ END OF INSERT ▲▲▲
```

</details>

### Step 2 — Rewrite `_get_rewards` with three payments and two penalties

<details>
<summary>Expand Step 2</summary>

> **Environment:** none needed — you are editing files.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method _get_rewards ───────────────────────

    def _get_rewards(self) -> torch.Tensor:

        # ▼▼▼ DELETE the hover task's body — it computed distance_to_goal ▼▼▼
        #   distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        #   distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        #   rewards = { ... }   ; reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # ▲▲▲ and REPLACE the whole body with the code below ▲▲▲

        # ▼▼▼ INSERT HERE! ▼▼▼
        to_target = self._attacker.data.root_pos_w - self._robot.data.root_pos_w
        dir_to_target = to_target / self._dist.unsqueeze(1).clamp(min=1e-6)

        # 1) CLOSING SPEED: my velocity, projected onto the target direction.
        #    +1.0 means "approaching at 1 m/s"; negative means fleeing. Paid EVERY step.
        closing = (self._robot.data.root_lin_vel_w * dir_to_target).sum(dim=1)

        # 2) PROXIMITY: 1 - tanh(dist/4) — a smooth 0..1 value that rises as you approach.
        proximity = 1.0 - torch.tanh(self._dist / 4.0)

        # 3) CAPTURE: the one-off bonus
        captured = self._dist < self.cfg.capture_radius

        # 4) SMOOTHNESS: how much the command changed since last step
        action_rate = torch.sum(torch.square(self._actions - self._prev_actions), dim=1)

        reward = (
            self.cfg.closing_reward_scale * closing
            + self.cfg.proximity_reward_scale * proximity
            + self.cfg.capture_bonus * captured.float()
            + self.cfg.action_rate_penalty * action_rate
        ) * self.step_dt
        self._prev_actions = self._actions.clone()
        return reward
        # ▲▲▲ END OF INSERT ▲▲▲
```

(Scaling by `step_dt`, as the built-in tasks do, keeps reward magnitudes comparable if you later change the control frequency.)

**Why three payments instead of only the capture bonus.** With the bonus alone, an untrained policy would have to stumble within 0.35 m of a moving attacker before receiving anything but zero. Across thousands of episodes that may never happen, and PPO cannot improve a policy whose returns are identical everywhere — that is a sparse reward. `closing` and `proximity` pay on every step, so even a bad policy gets told which direction was better. The bonus then supplies the push from "nearby" to "in contact".

**Why `tanh` for proximity.** It maps distance onto a bounded 0–1 curve: a steep gradient near the target, flat far away. An unbounded `1/dist` grows without limit at small distances and produces advantage estimates large enough to destabilise the PPO update. The hover task uses the same bounded squash for its position term.

**Why penalise the action rate.** In simulation, changing the command from +1 to −1 between steps costs nothing — the force model responds instantly. On a Tello, motors have inertia and the WiFi link drops packets, so a command stream that swings wildly produces a drone that shakes instead of flying. Published sim-to-real work identifies command smoothness as one of the decisive factors in whether a policy transfers. Keep the weight small: raise it too far and the defender becomes sluggish and stops manoeuvring.

**The reward-hacking pattern to watch for**, the same failure mode you saw in Ant: the policy learns to orbit the attacker at about 1 m. Proximity pays well there, closing averages zero over a lap, and capture never happens. If you see it, raise `closing_reward_scale` or shrink the `4.0` inside the tanh to steepen the near-field gradient.

</details>

### Step 3 — Define when an episode ends

<details>
<summary>Expand Step 3</summary>

> **Environment:** none needed — you are editing files.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method _get_dones ─────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        # ▼▼▼ DELETE the hover task's body — it looked like this: ▼▼▼
        #   time_out = self.episode_length_buf >= self.max_episode_length - 1
        #   died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1,
        #                           self._robot.data.root_pos_w[:, 2] > 2.0)
        #   return died, time_out
        # ▲▲▲ and REPLACE it with the code below ▲▲▲

        # ▼▼▼ INSERT HERE! ▼▼▼
        captured = self._dist < self.cfg.capture_radius                       # success
        crashed = self._robot.data.root_pos_w[:, 2] < 0.1                     # floor
        escaped = self._dist > self.cfg.arena_radius                          # lost it
        died = crashed | escaped | captured        # all three END the episode now
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return died, time_out
        # ▲▲▲ END OF INSERT ▲▲▲
```

Ending the episode on capture matters: if it continued, the defender would sit just inside the 0.35 m radius and collect the 200-point bonus again on every step — a reward exploit you would only notice after a confusing training run in which reward climbed steeply and capture rate did not.

</details>

> ✅ **Checkpoint 3.2** — the code compiles and the env steps, and you can answer: *"If I set `closing_reward_scale` to 0, what degenerate behaviour appears?"* (Answer: hovering at the distance where the tanh curve is steepest — proximity pays without ever closing.)

</details>

---

## 3.3 Randomise the Drone's Physics, Train with PPO, and Read the Curves (≤1.5h hands-on + background compute)

<details>
<summary>Expand 3.3</summary>

> **What this subchapter does:** launches the real training run and manages it. First it randomises the physical properties of the simulated drone that you cannot measure on a C$115 aircraft, then it trains against a slow attacker and raises the speed in stages, then it reads the five metrics that diagnose a pursuit task specifically. It ends by measuring `capture_ang_size` — the share of frame the attacker fills at capture — which Chapter 6.2 needs because it has no simulator to ask for distance. Deliverable: a checkpoint that reliably intercepts, and one calibrated constant.

### Step 0 — Randomise the physical properties you cannot measure

<details>
<summary>Expand Step 0</summary>

> **Environment:** none needed — you are editing files.

A Tello's mass changes with battery and wear, its available thrust sags as the battery drains, and it drifts because its trim is imperfect. You cannot measure any of these precisely, so instead of guessing one value, each episode trains against a slightly different drone drawn from a range that contains the real one.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method __init__ ───────────────────────────

        self._prev_actions = torch.zeros(self.num_envs, 4, device=self.device)  # ← from 3.1 D

        # ▼▼▼ INSERT HERE! — allocate the randomised-drone buffers ▼▼▼
        self._mass_scale = torch.ones(self.num_envs, device=self.device)
        self._thrust_scale = torch.ones(self.num_envs, device=self.device)
        self._drift = torch.zeros(self.num_envs, 3, device=self.device)
        # ▲▲▲ END OF INSERT ▲▲▲


# ── SECTION: class QuadcopterEnv, method _reset_idx ─────────────────────────

        self._prev_asz[env_ids] = 0.0            # ← EXISTING, from 3.1 Part D

        # ▼▼▼ INSERT HERE! — draw a slightly different drone for each new episode ▼▼▼
        n = len(env_ids)
        # mass varies with battery charge and wear
        self._mass_scale[env_ids] = 1.0 + (torch.rand(n, device=self.device) - 0.5) * 0.2
        # available thrust falls as the battery drains
        self._thrust_scale[env_ids] = 0.85 + torch.rand(n, device=self.device) * 0.3
        # cheap drones drift — a small constant push in a random direction
        self._drift[env_ids] = (torch.rand(n, 3, device=self.device) - 0.5) * 0.15
        # ▲▲▲ END OF INSERT ▲▲▲
```

Then apply them where the force is built, in `_pre_physics_step` (Part A), and add small Gaussian noise to the readings before they enter the delay buffer in `_get_observations` (Part G). The detector's rectangle jitters by a few pixels between frames, and a policy that only ever saw perfectly smooth bearings will chase that jitter.

**Randomisation matters more than accuracy here.** A policy that works across a wide band of possible drones works on the actual one; a policy tuned to your single best guess fails wherever that guess was wrong, and you cannot tell which parameter was wrong from the flight.

**A note on wind.** `_drift` is a constant push per episode, which represents imperfect trim well and wind poorly — real wind gusts and changes direction. This tutorial assumes calm conditions, and that assumption is load-bearing: on a breezy day the disturbance exceeds anything the policy trained against. Modelling wind would mean varying `_drift` *during* an episode, which is a harder task needing its own training run.

**Expect a lower capture rate than an unrandomised run.** You have made the task harder in the ways reality is harder. A policy capturing 60–70% under randomisation is more likely to fly than one capturing 95% under ideal conditions, because the second was never solving the real problem.

</details>

### Step 1 — Start training against a slow attacker

<details>
<summary>Expand Step 1</summary>

> **Environment:** `env_drone`

Set `attacker_speed = 0.3` before launching. At the full 0.6 m/s an untrained policy almost never comes within 0.35 m of the attacker, so the capture bonus is never collected and PPO converges on whatever the proximity term alone pays for — usually stationary hovering. Starting slow gets captures happening early, which is what makes the bonus informative.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnvCfg, the 2.1 pursuit knobs ──────────────────

    # BEFORE: attacker_speed = 0.6
    attacker_speed = 0.3       # ◄── CHANGE for the first curriculum stage
```

*Run from:* `any folder` — *the script lives in:* `C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\train.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 2048 --headless --max_iterations 1500
tensorboard --logdir C:\projects\drone_pursuit\drone_pursuit\logs\skrl
```

Expect roughly 30–90 min depending on GPU. Block D is fully independent of this run — open a second terminal and start Chapter 4 while it trains.

</details>

### Step 2 — Log and interpret the pursuit-specific metrics

<details>
<summary>Expand Step 2</summary>

> **Environment:** none needed — you are editing files.

Total reward can climb steadily while the defender never actually catches anything, because proximity and closing pay continuously. This step adds five task-specific numbers to the logging dictionary the built-in tasks already use, and gives the failure signature for each.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnv, method _reset_idx — the logging block ─────
# ──          that already exists near the top of the method            ─────

        extras = dict()
        # --- EXISTING episode-sum logging, unchanged ---

        # ▼▼▼ INSERT HERE! — pursuit-specific metrics ▼▼▼
        extras["Metrics/final_distance"] = self._dist[env_ids].mean().item()
        extras["Metrics/capture_rate"] = (
            self._dist[env_ids] < self.cfg.capture_radius
        ).float().mean().item()
        extras["Metrics/visible_fraction"] = self._prev_asz[env_ids].gt(0).float().mean().item()
        # ▲▲▲ END OF INSERT ▲▲▲

        self.extras["log"] = dict(extras)      # ← EXISTING line, keep it LAST
```

| Metric | Healthy | Sick pattern → diagnosis |
|---|---|---|
| **capture rate** (fraction of episodes ending in capture) | climbs past 50–80% | stuck at 0% while reward climbs → orbiting exploit (see 3.2) |
| **mean final distance** | falls toward capture_radius | plateaus at a fixed radius → orbiting again, or the attacker is simply faster than `max_speed` |
| **episode length** | *falls* as captures come sooner | pinned at max → nobody is catching anybody |
| **crash rate** | < 10% after the early phase | high forever → stability penalties too weak against the closing reward (diving into the floor) |
| **lost-sight fraction** (mean of `visible`) | rises toward ~0.9 as the policy learns to keep the attacker in frame | falling → the defender is flying blind and the reward is not making that costly |

</details>

### Step 3 — Raise the attacker's speed and resume from the checkpoint

<details>
<summary>Expand Step 3</summary>

> **Environment:** `env_drone`

Once capture rate passes about 80% at `attacker_speed = 0.3`, stop the run, set `attacker_speed = 0.6` in the cfg, and resume from the saved checkpoint rather than restarting. Restarting would discard a policy that already knows how to intercept and re-learn it against a harder target, which takes longer and often fails.

*Run from:* `any folder` — *checkpoints live in:* `C:\projects\drone_pursuit\drone_pursuit\logs\skrl\<run-folder>\checkpoints\`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\train.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 2048 --headless --max_iterations 1500 --checkpoint C:\projects\drone_pursuit\drone_pursuit\logs\skrl\<run-folder>\checkpoints\best_agent.pt
```

Repeat toward 1.0 m/s if you want a harder chase. This staged difficulty is curriculum learning in its simplest form; the adaptive environment generator in the Tsinghua paper is the same instinct built as a research system.

</details>

### Step 4 — Measure the frame share at capture (5 minutes, saves an hour in Chapter 6)

<details>
<summary>Expand Step 4</summary>

> **Environment:** `env_drone`

Chapter 6.2 has to declare a capture using only the camera, because in the real world no simulator reports distance. What it needs is a threshold on `ang_size` — the share of frame width the attacker fills at the moment `_dist` crosses `capture_radius`. Right now is the only point in the project where both quantities exist at the same time, which is why this measurement happens here rather than in Chapter 6.

Add logging so that on every step where `_dist` first drops below `capture_radius`, the corresponding `ang_size` is recorded, then run `play.py` for a few dozen episodes and look at the distribution.

*Run from:* `any folder` — *the script lives in:* `C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\play.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 16
```

You get a spread rather than a single value, because the attacker's rectangle is wider seen face-on than edge-on. Pick from that spread according to the error you prefer: the **low end** declares capture eagerly and occasionally claims one it did not earn; the **median** is balanced; the **high end** only confirms certain captures and silently misses real ones.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnvCfg, below the Part E camera model ──────────

    attacker_span_m = 0.13       # ← EXISTING, from 3.1 Part E

    # ▼▼▼ INSERT HERE! — YOUR measured value, not this placeholder ▼▼▼
    capture_ang_size = 0.19      # frame share at capture — read off in this step
    # ▲▲▲ END OF INSERT ▲▲▲
```

Record it in `C:\projects\drone_pursuit\drone_pursuit\project_notes.txt` alongside the 1.4 measurements — values you derive by measurement are not in the lockfile and are not recoverable from it.

You are not choosing this number freely. You set `capture_radius` in metres in 2.1, and the camera geometry from 3.1 Part E determines what that distance looks like in pixels. This step reads off the answer.

</details>

### Step 5 — Play the trained policy and check for interception

<details>
<summary>Expand Step 5</summary>

> **Environment:** `env_drone`

*Run from:* `any folder` — *the script lives in:* `C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\play.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 16
```

Watch for the *lead*: a well-trained defender cuts the corner toward where the attacker is going rather than following its path from behind. Tail-chasing means the rate-of-change readings (`d_bearing_x`, `d_ang_size`) are not influencing the policy, which usually points back to a delay buffer that is returning fresh values instead of delayed ones.

</details>

> ✅ **Checkpoint 3.3 — MILESTONE: Problem 1 (control) SOLVED**
> 1. Capture rate > 80% at `attacker_speed` ≥ 0.6
> 2. Play shows visible interception (corner-cutting), not just tail-chasing
> 3. Best checkpoint path recorded in `project_notes.txt` — Chapters 6 and 7 both need it
> 4. `capture_ang_size` measured and written into the cfg

</details>

</details>

</details>

---
# ██ BLOCK D — Generate Labelled Synthetic Images of a Drone ██

> 💻 **NO HARDWARE — simulation only.**

<details>
<summary>Expand Block D</summary>

**What this block produces:** roughly 2500 rendered images of a Crazyflie under randomised lighting, backgrounds and viewpoints, each with a bounding box label, converted into the folder layout Ultralytics reads. No detector is trained here and no policy is touched — the output is a dataset on disk, which Block E consumes.

**Why the images are generated rather than photographed.** Labelling real photographs means a person drawing a rectangle on every one. In Isaac Sim the renderer already knows which pixels belong to the drone prim, so the `bounding_box_2d_tight` annotator writes the label at the same moment it writes the image, with no drawing and no pixel error. Your work reduces to deciding what varies between frames.

**This block is independent of Block C.** Run it in a second terminal while Chapter 3.3 trains.

**The files created in this block:**

```
C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\generate_drone_data.py    ← 4.1, 4.2
C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\convert_to_yolo.py        ← 4.3
C:\projects\drone_pursuit\drone_pursuit\data\yolo\drone.yaml                  ← 4.3
```

---

# Chapter 4 — Photograph a Drone in Isaac Sim with Replicator and Label It Automatically

> 💻 **NO HARDWARE — simulation only.**

<details>
<summary>Expand Chapter 4</summary>

## 4.1 Build a Replicator Script That Photographs and Auto-Labels a Drone (≤1.5h)

<details>
<summary>Expand 4.1</summary>

> **What this subchapter does:** creates a standalone Replicator script — separate from the RL environment — that spawns a Crazyflie tagged with the semantic class "drone", points a camera at it from randomly chosen positions, and writes each frame together with its bounding box. It runs outside the pursuit env because data generation wants high-quality rendering and the RL env wants step speed, and mixing the two makes both harder to tune. The 20 trial frames it produces are checked by hand before 4.2 scales it to 2500.

### Why synthetic images arrive pre-labelled

The renderer knows every pixel's source prim. Tagging the Crazyflie prim with `semantic_tags=[("class", "drone")]` tells Replicator which prim matters, and the `bounding_box_2d_tight` annotator then emits the pixel rectangle enclosing its visible pixels, per frame, automatically. Your remaining job is only: tag the right prims, point the camera from varied poses, and randomise everything else so the network keys on drone shape rather than on the scenery it happened to be rendered against.

### Step 1 — Create the folders and the script's opening section

<details>
<summary>Expand Step 1</summary>

> **Environment:** `env_drone` (the `mkdir` needs no environment; the script does)

This step writes the top half of the generator: the `AppLauncher` boilerplate that starts Isaac Sim with rendering enabled, a ground plane, two lights, and the Crazyflie carrying its semantic tag. The tag is the one line that makes labels possible; without it the annotator returns empty boxes and every frame is dropped in 4.3.

*Run from:* `C:\projects\drone_pursuit\drone_pursuit`
```bat
cd C:\projects\drone_pursuit\drone_pursuit
mkdir C:\projects\drone_pursuit\drone_pursuit\scripts\sdg C:\projects\drone_pursuit\drone_pursuit\data\raw
```

*File to CREATE (new, empty file):* `C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\generate_drone_data.py`

```python
# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\generate_drone_data.py
# ── This is a NEW file. Everything below is section 1 of 3; sections 2 and 3
# ──          are appended in Step 2 and Step 3, in this order, at the END.

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

# ── SECTION 2 (Step 2) GOES HERE — camera and writer ────────────────────────
# ── SECTION 3 (Step 3) GOES BELOW THAT — the capture trigger ────────────────
```

Note the asset path: Isaac Sim 5.x moved it to `Robots/Bitcraze/Crazyflie/cf2x.usd`. Older tutorials say `Robots/Crazyflie/`, and the rename is listed in the Isaac Lab release notes — a concrete example of why 1.0 pinned a commit.

</details>

### Step 2 — Add the Replicator camera and the writer that saves the labels

<details>
<summary>Expand Step 2</summary>

> **Environment:** none needed — you are editing a file.

This step creates the camera that takes the photograph and the `BasicWriter` that saves each frame's RGB image plus its tight bounding box into `data\raw`. The camera settings are not free choices: focal length 12 mm and a 640×480 render match the Tello's 83° 4:3 lens, for the same reason as Chapter 3.1 Part E — a detector trained on a different field of view sees a differently-shaped drone at the same distance.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\generate_drone_data.py`

```python
# ── FILE: ...\scripts\sdg\generate_drone_data.py ────────────────────────────
# ── SECTION: append directly BELOW the drone_cfg.func(...) line from Step 1 ─

drone_cfg.func("/World/Drone", drone_cfg, translation=(0.0, 0.0, 1.5))   # ← from Step 1

# ▼▼▼ INSERT HERE! ▼▼▼
# a Replicator camera and a render product (the surface it exposes onto)
camera = rep.create.camera(focal_length=12.0)   # ~83 deg FOV, matching a Tello
render_product = rep.create.render_product(camera, (640, 480))   # 4:3, like the real drone

# BasicWriter: saves RGB + tight 2D boxes for every captured frame
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir=args.out_dir,
    rgb=True,
    bounding_box_2d_tight=True,      # "tight" = shrink-wrapped to visible pixels
)
writer.attach([render_product])
# ▲▲▲ END OF INSERT ▲▲▲
```

**Tight versus loose boxes:** a *loose* box encloses the object's full extent even where another object hides part of it; a *tight* box encloses only the pixels actually visible. The detector is trained on what is visible, so tight is the matching choice.

</details>

### Step 3 — Add the capture trigger and generate 20 trial frames

<details>
<summary>Expand Step 3</summary>

> **Environment:** `env_drone`

Twenty frames take under a minute and are enough to reveal a broken camera pose, a missing tag or an empty writer directory. The production run in 4.2 takes far longer, so anything wrong is worth finding here.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\generate_drone_data.py`

```python
# ── FILE: ...\scripts\sdg\generate_drone_data.py ────────────────────────────
# ── SECTION: append at the very END of the file, below writer.attach(...) ───

writer.attach([render_product])          # ← last line from Step 2

# ▼▼▼ INSERT HERE! — 4.2 adds more randomisers INSIDE this same `with` block ▼▼▼
with rep.trigger.on_frame(max_execs=args.num_frames):
    with camera:
        rep.modify.pose(
            position=rep.distribution.uniform((-3, -3, 0.5), (3, 3, 3.0)),
            look_at="/World/Drone",
        )
rep.orchestrator.run_until_complete()
simulation_app.close()
# ▲▲▲ END OF INSERT — this is the end of the file ▲▲▲
```

*Run from:* `any folder` — *the script lives in:* `C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\generate_drone_data.py --num_frames 20 --headless
```

</details>

### Step 4 — Inspect the frames and the label files by hand

<details>
<summary>Expand Step 4</summary>

> **Environment:** none needed — you are looking at files in Explorer.

*Folder to open:* `C:\projects\drone_pursuit\drone_pursuit\data\raw\`

You should find `rgb_0000.png`…, plus `bounding_box_2d_tight_0000.npy` and a matching `..._labels.json` per frame. Open a few PNGs and confirm the drone is visible from varied angles and distances. Load one `.npy`: it is a structured array with `x_min, y_min, x_max, y_max` and a `semanticId` that maps through the labels JSON to `"drone"`.

A labelling problem found here costs a few minutes. The same problem found in Chapter 5 looks like a detector whose mAP will not rise no matter how long it trains, and takes a training run to notice.

</details>

> ✅ **Checkpoint 4.1** — 20 frames exist; the box coordinates in the `.npy` match where the drone appears in the PNG; the labels JSON contains the `drone` class.

</details>

---

## 4.2 Randomise Lighting, Background and Pose, Then Generate 2500 Frames (≤1.5h)

<details>
<summary>Expand 4.2</summary>

> **What this subchapter does:** twenty photographs of one drone in one grey world would train a detector that only works in that grey world. This subchapter varies camera distance and elevation, both light sources, the drone's own orientation, and a set of distractor shapes, so the only constant across thousands of frames is the drone itself. It then runs the production batch that Chapter 4.3 converts and Chapter 5.1 trains on.

### Step 1 — Decide what to randomise, and what each dial buys at demo time

<details>
<summary>Expand Step 1</summary>

> **Environment:** none needed — this step is explanation only.

| Randomise | Range | Because at demo time… |
|---|---|---|
| Camera distance | 0.5–6 m from drone | the attacker's apparent size changes continuously as the chase closes |
| Camera elevation | below AND above the drone | a defender below the attacker sees it against **sky**; above it, against **ground** — two backgrounds with completely different brightness and texture statistics |
| Dome light intensity/colour | 500–6000, warm↔cool | arena and room lighting are not fixed across sessions |
| Drone yaw/pitch | full yaw, ±20° pitch | a banking drone presents a different silhouette from a level one |
| Distractor objects | 3–8 shapes, random colour and scale | teaches "this is NOT a drone"; without negative examples, any small dark object becomes a detection |
| **Sun direction and intensity** | full azimuth, 10–80° elevation, wide intensity range | outdoors, one strong directional source produces harsh shadows, silhouettes and blown highlights that a dome light never creates |

### Why the sun needs its own light, separate from the dome

A **dome light** illuminates evenly from every direction at once — the look of an overcast sky or an evenly lit room. A **distant light** is a single source infinitely far away with parallel rays, which is what the sun is. It produces three things a dome light cannot:

- **A lit side and a dark side.** With the sun behind the attacker, the drone becomes a near-black silhouette against bright sky. This is the hardest case the detector will face outdoors, and without a directional light it never sees one.
- **Cast shadows**, which give the detector a drone-shaped dark region it must learn *not* to box.
- **Blown-out highlights**, where bright sky saturates the sensor and detail disappears.

**What rendering still does not reproduce** is lens flare — the streaks and rings from light scattering inside a real lens, which a Tello camera produces heavily when pointed near the sun. If the detector fails specifically when flying toward the sun, this is the cause, and Chapter 7.4's retraining on real footage is the fix. Flying with the sun behind you avoids it entirely.

</details>

### Step 2 — Add the randomisers to the capture trigger

<details>
<summary>Expand Step 2</summary>

> **Environment:** none needed — you are editing a file.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\generate_drone_data.py`

```python
# ── FILE: ...\scripts\sdg\generate_drone_data.py ────────────────────────────
# ── SECTION: the `with rep.trigger.on_frame(...)` block from 4.1 Step 3 ─────

writer.attach([render_product])          # ← EXISTING, from 4.1 Step 2

# ▼▼▼ INSERT HERE! — the distractor pool, ABOVE the trigger block ▼▼▼
distractors = rep.create.group([
    rep.create.cube(count=4, semantics=[("class", "distractor")]),
    rep.create.sphere(count=4, semantics=[("class", "distractor")]),
])
# ▲▲▲ END OF INSERT ▲▲▲

with rep.trigger.on_frame(max_execs=args.num_frames):     # ← EXISTING line
    with camera:                                          # ← EXISTING block, but
        rep.modify.pose(                                  #   WIDEN the ranges:
            # BEFORE: uniform((-3, -3, 0.5), (3, 3, 3.0))
            position=rep.distribution.uniform((-6, -6, 0.2), (6, 6, 4.0)),
            look_at="/World/Drone",
        )

    # ▼▼▼ INSERT HERE! — everything below, still INSIDE the `with` block, ▼▼▼
    # ▼▼▼ i.e. indented to the same level as `with camera:`                ▼▼▼
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
    # ▲▲▲ END OF INSERT ▲▲▲

rep.orchestrator.run_until_complete()      # ← EXISTING, unchanged
simulation_app.close()                     # ← EXISTING, unchanged
```

The distractors carry their own semantic class, and Chapter 4.3's conversion script filters those boxes out. They exist only as visual noise in the image, never as a class the detector is taught.

The rotation range `(-80, 0, 0)` to `(-10, 0, 360)` sweeps the sun through every compass direction at elevations from 10° (low sun, long shadows, frequent backlighting) to 80° (near overhead). The colour range spans golden low sun through neutral midday. **The intensity ceiling of 12000 is deliberately high**, so that a meaningful share of frames are genuinely difficult — overexposed, with the drone reduced to a dark shape. Those are the frames that teach the detector to survive the conditions Chapter 7 flies in.

</details>

### Step 3 — Run the production batch of 2500 frames

<details>
<summary>Expand Step 3</summary>

> **Environment:** `env_drone`

*Run from:* `any folder` — *the script lives in:* `C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\` — *output goes to:* `C:\projects\drone_pursuit\drone_pursuit\data\raw\`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\generate_drone_data.py --num_frames 2500 --headless
```

2000–3000 frames is a solid single-class dataset. Variety across the dials above matters more than raw count. This takes a while — start it, then work through 4.3's conversion script while it runs.

</details>

> ✅ **Checkpoint 4.2** — flipping through 30 random production frames you see: near and far drones, sky and ground backgrounds, bright and dark scenes, distractors present, **some frames with the drone strongly backlit and nearly a silhouette**, and the drone always findable by you (if a human cannot find it, the network will not).
>
> If no frame looks harshly lit, the sun is not being randomised — check that `/World/Sun` exists and that the `rep.get.prims` pattern matches it.

</details>

---

## 4.3 Convert Replicator Output to YOLO Format and Verify the Labels (≤1.5h)

<details>
<summary>Expand 4.3</summary>

> **What this subchapter does:** Ultralytics does not read Replicator's `.npy` files. This subchapter converts the pixel-corner boxes into YOLO's normalised centre-and-size text format, discards frames that are empty, corrupt, or contain a drone smaller than 10 pixels, filters out distractor boxes, and splits the result 85/15 into train and val folders. It ends with a visual check, because a conversion bug — a swapped width and height, for instance — shows up in Chapter 5 as a low mAP that no amount of extra training fixes.

### The format translation

```
REPLICATOR (absolute pixel corners)        YOLO (normalized center + size)
(x_min,y_min)                              one line per object in a .txt:
   ┌─────────┐                             class  x_c     y_c     w      h
   │  drone  │                             0      0.512   0.430   0.146  0.118
   └─────────┘(x_max,y_max)                        └── all divided by image size (0..1) ──┘
```

### Step 1 — Write the conversion and filtering script

<details>
<summary>Expand Step 1</summary>

> **Environment:** none needed to write it; `env_drone` (or any env with numpy) to run it.

*File to CREATE (new, empty file):* `C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\convert_to_yolo.py`

```python
# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\convert_to_yolo.py
# ── This is a NEW file — the whole content, nothing to insert into.
# ── Reads : C:\projects\drone_pursuit\drone_pursuit\data\raw
# ── Writes: C:\projects\drone_pursuit\drone_pursuit\data\yolo

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
    if lines:                                                # keep only frames with a drone
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

*Run from:* `any folder`
```bat
conda activate env_drone
python C:\projects\drone_pursuit\drone_pursuit\scripts\sdg\convert_to_yolo.py
```

</details>

### Step 2 — Write the dataset descriptor

<details>
<summary>Expand Step 2</summary>

> **Environment:** none needed — you are creating a text file.

`drone.yaml` is the file the `yolo train` command in 5.1 is pointed at. It names the two split folders and declares that this dataset has exactly one class, which is what makes the exported ONNX output shape `(1, 5, 6300)` rather than the 80-class `(1, 84, 8400)` you saw in the 1.3 smoke test.

*File to CREATE (new, empty file):* `C:\projects\drone_pursuit\drone_pursuit\data\yolo\drone.yaml`

```yaml
# ── FILE: C:\projects\drone_pursuit\drone_pursuit\data\yolo\drone.yaml ──────
# ── NEW file — the whole content.

path: C:\projects\drone_pursuit\drone_pursuit\data\yolo
train: images\train
val: images\val
names:
  0: drone
```

</details>

### Step 3 — Draw ten boxes onto their images and look at them

<details>
<summary>Expand Step 3</summary>

> **Environment:** any env with PIL installed.

*Folders involved:* images in `C:\projects\drone_pursuit\drone_pursuit\data\yolo\images\train\`, labels in `...\data\yolo\labels\train\`

Write a short PIL script (or use Ultralytics' dataset visualiser once you are in Chapter 5) that draws each label rectangle onto its image for ten random samples. A swapped width and height, or a centre computed from the wrong corner, is immediately obvious in a picture and completely invisible in the numbers — the drop count and file counts all look correct while every box is wrong.

</details>

> ✅ **Checkpoint 4.3 — MILESTONE: the detector's training data is ready**
> 1. ~85/15 train/val split on disk in the YOLO folder layout
> 2. Drop rate under about 20% — much higher means the camera pose randomisation in 4.2 frames the drone out too often; tighten the position ranges
> 3. 10 out of 10 spot-checked boxes hug the drone

</details>

</details>

</details>

---

# ██ BLOCK E — Train the Object Detection Model That Finds the Drone ██

> 💻 **NO HARDWARE — simulation only.**

<details>
<summary>Expand Block E</summary>

**What this block produces:** `drone_detector.onnx`, a YOLOv8-nano network that takes a 480×640 RGB frame and returns a rectangle around the drone in it. Training happens in `drone_vision`; the ONNX file is the only thing that crosses into `env_drone`, for the reason established in 1.3 — the two environments hold incompatible torch versions and ONNX carries no framework with it.

**Folders used in this block:**

```
C:\projects\drone_pursuit\drone_pursuit\data\yolo\            ← input, from 4.3
C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\ ← YOLO's own output
C:\projects\drone_pursuit\drone_pursuit\data\arena_frames\    ← 5.2 test frames
C:\projects\drone_pursuit\drone_pursuit\models\               ← the exported .onnx
```

---

# Chapter 5 — Train YOLOv8 on the Synthetic Images and Export It for Isaac Lab

> 💻 **NO HARDWARE — simulation only.**

<details>
<summary>Expand Chapter 5</summary>

## 5.1 Fine-Tune YOLOv8-nano on Your Synthetic Drone Images (≤1.5h)

<details>
<summary>Expand 5.1</summary>

> **What this subchapter does:** takes YOLOv8n, a network already trained on millions of everyday photographs, and continues its training on your 2500 drone renders so it specialises in one class. Fine-tuning rather than training from scratch is why a few thousand images suffice: the network already encodes edges, textures and lighting, and only has to learn what a drone looks like. Deliverable: `best.pt`, which 5.2 stress-tests and exports.

### Why the nano size

YOLOv8 ships in five sizes, n/s/m/l/x. Chapter 6.2 runs the detector inside the simulation loop and Chapter 7.2 runs it inside a 20 Hz flight loop, where a detector taking longer than 50 ms starves the control rate. Nano is about 3 million parameters and infers in a few milliseconds. Larger models earn their cost on cluttered many-class scenes, not on one distinctive object.

### Step 1 — Run the training

<details>
<summary>Expand Step 1</summary>

> **Environment:** `drone_vision`

*Run from:* `C:\projects\drone_pursuit\drone_pursuit` — *output lands in:* `C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\`
```bat
conda activate drone_vision
cd C:\projects\drone_pursuit\drone_pursuit
yolo detect train data=C:\projects\drone_pursuit\drone_pursuit\data\yolo\drone.yaml model=yolov8n.pt epochs=60 imgsz=640 batch=16 name=drone_v1
```

20–40 min typically. While it runs, open the output folder. Two files matter: `train_batch0.jpg` shows augmented training samples with their boxes drawn — your last chance to catch a label bug from 4.3 — and `results.png` plots the loss and metric curves.

</details>

### Step 2 — Read the metrics

<details>
<summary>Expand Step 2</summary>

> **Environment:** none needed — you are reading the output files.

*Folder to open:* `C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\`

- **box_loss / cls_loss** — localisation error and classification error. Both should fall steadily.
- **mAP50** — *mean Average Precision at 50% IoU*. **IoU** (Intersection over Union) scores how much a predicted box overlaps the true one, 0 to 1; "at 50" counts a detection as correct when that overlap is at least 0.5; **AP** integrates precision across confidence thresholds; the **m** averages over classes, of which we have one. For a single distinctive object with matching backgrounds, expect **mAP50 above 0.9**.
- **mAP50-95** — the same measure averaged over stricter overlap thresholds up to 0.95, so it is always lower. Above 0.6 is fine here, because the policy needs roughly where the drone is, not a surgically exact corner.

If mAP50 comes in below 0.8, the cause is almost always the data rather than the hyperparameters: check 4.3's drop rate, re-run the box spot-check, and confirm the training set actually contains hard cases such as a distant drone against ground clutter.

</details>

> ✅ **Checkpoint 5.1** — mAP50 ≥ 0.9 on val; `val_batch0_pred.jpg` shows tight, confident boxes.

</details>

---

## 5.2 Test the Detector on Arena Frames and Export It to ONNX (≤1.5h)

<details>
<summary>Expand 5.2</summary>

> **What this subchapter does:** a detector can score above 0.9 mAP on its own validation split and still miss the attacker in the pursuit arena, because the SDG scene and the arena differ in lighting, background and typical viewing distance. This subchapter captures frames from the defender's own camera during a real chase, tests the detector on those, and then exports it to ONNX at 480×640 so it can run inside `env_drone` in Chapter 6.

### Step 1 — Attach a camera to the defender and capture chase frames

<details>
<summary>Expand Step 1</summary>

> **Environment:** `env_drone` — this runs Isaac Lab to capture the frames.

This step attaches a `TiledCameraCfg` to the defender's body and saves what it sees while your Chapter 3 policy chases. Those images are the ones the detector will really face, and the same camera stays in place for Chapter 6.2's demo — so this is also a dry run of that wiring.

*Run from:* `any folder`
```bat
mkdir C:\projects\drone_pursuit\drone_pursuit\data\arena_frames
```

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: the imports at the top of the file ─────────────────────────────

# ▼▼▼ INSERT HERE! ▼▼▼
from isaaclab.sensors import TiledCamera, TiledCameraCfg
# ▲▲▲ END OF INSERT ▲▲▲


# ── SECTION: class QuadcopterEnvCfg, below the 3.1 Part E camera constants ──

    capture_ang_size = 0.19      # ← EXISTING, from 3.3 Step 4

    # ▼▼▼ INSERT HERE! — the camera that RENDERS (3.1 Part E only described it) ▼▼▼
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/front_cam",       # rides on the defender
        offset=TiledCameraCfg.OffsetCfg(pos=(0.04, 0.0, 0.01), rot=(1.0, 0.0, 0.0, 0.0),
                                        convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0,        # = cam_focal_mm!
                                         clipping_range=(0.05, 30.0)),
        width=640, height=480,                                     # = cam_width/height!
    )
    # ▲▲▲ END OF INSERT ▲▲▲


# ── SECTION: class QuadcopterEnv, method _setup_scene ───────────────────────

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)            # ← EXISTING
        self._attacker = Articulation(self.cfg.attacker)      # ← EXISTING, from 2.1

        # ▼▼▼ INSERT HERE! — BEFORE clone_environments ▼▼▼
        self._camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["camera"] = self._camera
        # ▲▲▲ END OF INSERT ▲▲▲

        self.scene.articulations["robot"] = self._robot       # ← EXISTING
        # ... rest unchanged ...
```

The focal length here must equal `cam_focal_mm` from 3.1 Part E. If they differ, the rendered frames obey a different geometry from the readings the policy trained on, and Chapter 6.1 Step 2 will report a constant-factor disagreement.

Then run `play.py` with your Chapter-3 checkpoint and dump `self._camera.data.output["rgb"]` to PNGs. Save about 20 frames spread across the approach, not 20 from the final second.

*Run from:* `any folder` — *frames go to:* `C:\projects\drone_pursuit\drone_pursuit\data\arena_frames\`
```bat
python C:\projects\drone_pursuit\drone_pursuit\scripts\skrl\play.py --task Template-Drone-Pursuit-Direct-v0 --num_envs 1 --enable_cameras
```

</details>

### Step 2 — Run the detector on those frames and study the misses

<details>
<summary>Expand Step 2</summary>

> **Environment:** `drone_vision`

*Run from:* `C:\projects\drone_pursuit\drone_pursuit` — *predictions are saved under:* `C:\projects\drone_pursuit\drone_pursuit\runs\detect\predict\`
```bat
conda activate drone_vision
yolo predict model=C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\weights\best.pt source=C:\projects\drone_pursuit\drone_pursuit\data\arena_frames save=True conf=0.4
```

The typical finding is that detection is reliable when the attacker is near and fails when it is a 12-pixel speck at range. If misses are frequent at the distances that matter — under about 5 m — go back to 4.2, add camera positions further from the drone, generate another batch, and fine-tune. Testing on frames from the environment where the detector will actually run is the check most often skipped, and the one that catches this before Chapter 6 turns it into an unexplained drop in capture rate.

</details>

### Step 3 — Export the detector to ONNX at 480×640

<details>
<summary>Expand Step 3</summary>

> **Environment:** `drone_vision`

*Run from:* `C:\projects\drone_pursuit\drone_pursuit` — *the finished model belongs in:* `C:\projects\drone_pursuit\drone_pursuit\models\`
```bat
yolo export model=C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\weights\best.pt format=onnx imgsz=480,640
copy C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\weights\best.onnx C:\projects\drone_pursuit\drone_pursuit\models\drone_detector.onnx
```

**Why `imgsz=480,640` and not plain `640`.** Training letterboxes images internally, which is harmless. At flight time you want the detector's input to be exactly the frame shape you feed it — 640 wide by 480 high — so the rectangle it returns is already in that coordinate space. Exporting square would mean padding every frame and subtracting the padding back out of every box, which is arithmetic with no benefit and one more place for a sign error.

Re-run the 1.3 onnxruntime smoke test in `env_drone` against this file. The input shape should now read `[1, 3, 480, 640]` and the output `(1, 5, 6300)` — four box numbers plus one class score for each of 6300 candidate boxes.

</details>

> ✅ **Checkpoint 5.2 — MILESTONE: Problem 2 (perception) SOLVED**
> 1. Detector finds the attacker in real arena frames at chase-relevant distances
> 2. `drone_detector.onnx` loads and runs inside `env_drone`, input `[1, 3, 480, 640]`, output `(1, 5, 6300)`

</details>

</details>

</details>

---
# ██ BLOCK F — Connect the Detector to the Policy and Run the Whole Loop in Simulation ██

> 💻 **NO HARDWARE — simulation only.** This is the rehearsal for Block G: the same converter code and the same seven readings run unchanged on the Tello, so any error found here is found without a drone in the air.

<details>
<summary>Expand Block F</summary>

**What this block produces:** a chase in which everything the defender knows about the attacker comes from a rendered camera frame passed through your detector — no positions read from the simulator.

**The one thing this block adds** is the converter between the detector's pixel rectangle and the policy's seven readings. Everything else already exists: the policy from Block C, the detector from Block E, and the camera from 5.2.

**The files created in this block:**

```
C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_bridge.py         ← 6.1
C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_pursuit_demo.py   ← 6.2
```

`vision_bridge.py` is imported again, unchanged, by Chapter 7.2's flight script.

---

# Chapter 6 — Feed the Policy Camera Readings Instead of Simulator Truth

> 💻 **NO HARDWARE — simulation only.**

<details>
<summary>Expand Chapter 6</summary>

## 6.1 Write the Converter That Turns a Bounding Box into the Seven Policy Inputs (≤1.5h)

<details>
<summary>Expand 6.1</summary>

> **What this subchapter does:** builds the piece of code that sits between the detector and the policy and translates one's output into the other's input. Because Chapter 3.1 defined the policy's inputs in image terms already, the translation is only a rescale by the image dimensions — no focal length, no assumed target width. This subchapter then verifies the converter's output against the numbers `_camera_readings()` computed for the same moment, so that any strange behaviour in 6.2 can be attributed to the detector or the policy rather than to this file.

### What the converter has to do

The detector hands you a rectangle: a centre point and a size, both in pixels. The policy wants the seven readings from §0.4. The whole translation is division by the image dimensions:

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

There is still one thing to get wrong here, and Step 2 exists to catch it: the image dimensions you construct the converter with must match the frames you feed it. Construct it for 640×480 and hand it a 960×720 Tello frame without resizing, and every bearing is scaled by 2/3 with no error raised.

### Step 1 — Write the converter file

<details>
<summary>Expand Step 1</summary>

> **Environment:** none needed — you are creating a folder and a file.

Two classes go in one file. `DroneDetector` loads the ONNX model under onnxruntime and returns the single highest-scoring rectangle, or `None` when nothing clears the confidence threshold. `CameraReadingBridge` rescales that rectangle into the seven numbers, remembers the previous values so it can compute the three rates of change, and holds those values frozen on a miss — the exact pattern the policy met in training whenever `ang_size` fell below the visibility threshold. Chapter 7.2's flight script imports both classes unchanged.

*Run from:* `any folder`
```bat
mkdir C:\projects\drone_pursuit\drone_pursuit\scripts\demo
```

*File to CREATE (new, empty file):* `C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_bridge.py`

```python
# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_bridge.py
# ── This is a NEW file — the whole content, nothing to insert into.
# ── Reads : C:\projects\drone_pursuit\drone_pursuit\models\drone_detector.onnx
# ── Imported by: vision_pursuit_demo.py (6.2) and fly_real.py (7.2)

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
        out = self.sess.run(None, {self.input_name: x})[0][0]   # (5, 6300)
        boxes, scores = out[:4, :], out[4, :]
        best = int(np.argmax(scores))
        if scores[best] < self.conf:
            return None                                     # attacker not seen this frame
        return boxes[:, best]


class CameraReadingBridge:
    """Rectangle → the seven numbers the policy was trained on. Holds last value on a miss."""

    def __init__(self, img_w=640, img_h=480):   # MUST match the frame you feed it
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

</details>

### Step 2 — Compare the converter's output against the training-time projection

<details>
<summary>Expand Step 2</summary>

> **Environment:** `env_drone`

For any given moment you can produce the same seven readings twice: once by projecting ground truth through the camera model (`_camera_readings()` from 3.1 Part F) and once by running the detector on the rendered frame and passing the rectangle through the converter. Agreement proves the two definitions match. This is the check that catches the field-of-view mismatch flagged in 3.1 Part E, which produces no error and degrades the chase in a way that looks like a training problem.

*Folders involved:* frames in `C:\projects\drone_pursuit\drone_pursuit\data\arena_frames\`, converter in `C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_bridge.py`

Reuse the frame-capture setup from 5.2, but for each saved frame also record what `_camera_readings()` computed for that moment. Then run the converter on the image and compare:

```
frame 12   projected (training):  bx=+0.109  by=−0.250  size=0.053
           bridge   (from image):  bx=+0.115  by=−0.244  size=0.049   ✓
```

Acceptance: bearings agree within a few hundredths, angular size within about 20% — the rectangle genuinely widens and narrows as the attacker banks, so some spread is expected. A disagreement in *sign* means an axis is flipped. A disagreement by a large constant factor means the `TiledCameraCfg` focal length does not match the `cam_focal_mm` / `cam_aperture_mm` in the env cfg.

</details>

### Step 3 — Test the out-of-view behaviour

<details>
<summary>Expand Step 3</summary>

> **Environment:** `env_drone`

Feed the converter frames in which the attacker is genuinely out of view. It must return `visible = 0`, hold the previous bearings unchanged, and report all three deltas as zero. Zeroing the bearings instead of holding them would send the policy a reading meaning "the attacker is dead ahead", which is the opposite of what losing sight means, and the defender would fly straight on.

</details>

> ✅ **Checkpoint 6.1**
> 1. Converter output matches the training-time projection on ≥15 varied frames
> 2. Field of view confirmed identical between `TiledCameraCfg` and the cfg constants from 3.1
> 3. Out-of-view frames produce `visible = 0` with frozen values and zero deltas

</details>

---

## 6.2 Run the Full See-Decide-Act Loop in Simulation (≤1.5h)

<details>
<summary>Expand 6.2</summary>

> **What this subchapter does:** connects the pursuit env with its onboard camera, your trained policy, the ONNX detector and the converter from 6.1 into one script. It runs a single environment and, on every control step, overwrites observation slots 6 to 12 with the seven numbers measured from a rendered frame. If the defender still captures, the complete see-decide-act loop works and only the hardware remains.

### The demo architecture

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

- **Self-state stays from the simulator, and that is not cheating.** A real drone reads its own velocity and attitude from its inertial sensors, and Chapter 7.2 gets those same six numbers from `get_speed_x` and `get_roll`. Only knowledge about the attacker has to be earned through the camera, and that is exactly what slots 6 to 12 hold.
- **Losing sight is already trained for.** When the detector returns nothing, the converter holds the previous readings and sets `visible = 0` — the pattern the policy met thousands of times during training whenever the attacker fell below the 0.012 detection threshold.

### Step 1 — Write the demo script

<details>
<summary>Expand Step 1</summary>

> **Environment:** none needed to write it; `env_drone` to run it.

This is the file that touches every subsystem, so expect to iterate on it.

*File to CREATE (new, empty file):* `C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_pursuit_demo.py`

```python
# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_pursuit_demo.py
# ── NEW file — the whole content, nothing to insert into.
# ── Imports  : vision_bridge.py, sitting in this SAME folder (scripts\demo\)
# ── Loads    : C:\projects\drone_pursuit\drone_pursuit\models\drone_detector.onnx
# ── Loads    : your Ch.3 checkpoint, passed on the command line

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

</details>

### Step 2 — Run it, and debug in two halves if it misbehaves

<details>
<summary>Expand Step 2</summary>

> **Environment:** `env_drone`

*Run from:* `any folder` — *the script lives in:* `C:\projects\drone_pursuit\drone_pursuit\scripts\demo\` — *checkpoints live in:* `C:\projects\drone_pursuit\drone_pursuit\logs\skrl\<run-folder>\checkpoints\`
```bat
conda activate env_drone
python C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_pursuit_demo.py --checkpoint C:\projects\drone_pursuit\drone_pursuit\logs\skrl\<run-folder>\checkpoints\best_agent.pt --enable_cameras
```

⚠️ The failures here are usually small and mechanical: the exact wrapper import path for your Isaac Lab minor version, the camera attribute name, or the details of loading the agent.

**Debug method — comment out one line first:**

```python
# ── FILE: ...\scripts\demo\vision_pursuit_demo.py, inside the while loop ────

    # 2) SPLICE — TEMPORARILY DISABLE this line for the first debug pass:
    # obs[:, 6:13] = torch.tensor(readings, device=obs.device)
```

With that line commented out the policy runs on the training-time projected readings, which must reproduce Chapter 3 behaviour exactly. That separates a checkpoint-loading problem from a vision-path problem. Re-enable the splice only after that passes.

**What success looks like.** The chase is less smooth than in Chapter 3. The detector's rectangle shifts by a few pixels between frames, the bearings shift with it, and the defender corrects more often. It still closes and captures. The gap between its capture rate on calculated readings and on measured readings is the number worth writing in `project_notes.txt`, because Chapter 7 will produce a third value for the same measurement on hardware.

**What to watch at the end of a chase:** `true dist` drops below your capture radius, and within a frame or two `camera says captured` flips to True. Two independent measurements of the same event agreeing within a frame or two means the `capture_ang_size` you calibrated in 3.3 is correct.

</details>

> ✅ **Checkpoint 6.2 — MILESTONE: Problem 3 (integration) SOLVED**
> The defender captures the attacker with everything it knows about the attacker coming from its camera and your detector — no assumed dimensions, no positions from the simulator, no camera geometry in the converter. Record a video.

</details>

---

## 6.3 Review What Is Complete and Choose the Next Extension

<details>
<summary>Expand 6.3</summary>

**The simulation half is complete.** The defender captures a moving attacker using only what a camera reports, with no privileged information about the target.

**Chapter 7 is the next step**: putting this policy on the Tello you set up in 1.4 and finding out whether it transfers. Nothing needs retraining — the policy already commands stick channels at your measured rate, expects delayed readings, and uses only telemetry a Tello reports.

### Directions beyond that

Each of these builds on what you now have, ordered by effort:

1. **Reactive attacker** (hours): add a velocity component that flees from the defender to `_move_attacker` in 2.2, then re-run the 3.3 curriculum. A harder pursuit problem, entirely in simulation.
2. **Bearing-only range estimation** (a project): add a filter that accumulates bearings across many frames and combines them with the defender's own known motion to recover the attacker's actual position plus an uncertainty estimate. Standard probabilistic state estimation, and it sits *below* the existing policy, so no retraining is needed. It would also restore something Chapter 6 deliberately gave up: an inspectable distance estimate.
3. **True multi-agent** (project): two defenders against one RL-controlled attacker, using skrl's IPPO or MAPPO — the Direct workflow supports multi-agent environments. The reference point is the Tsinghua Multi-UAV pursuit-evasion work: adaptive curriculum, evader prediction network, and sim-to-real on real quadrotors.
4. **OmniDrones** (project): a full drone-RL framework on Isaac Sim with realistic rotor dynamics and controllers, from the same lineage as that paper — the step beyond the simplified force model used here.

</details>

</details>

</details>

---

# ██ BLOCK G — Transfer the Trained Models to the Real Drone and Fly It ██

> 🔌 **DRONE HARDWARE REQUIRED — every subchapter.** 7.1 is the only one you can complete with the drone still in its box; 7.2 needs the Tello with its propellers off; 7.3 and 7.4 need the Tello flying, plus the target drone, guards and spare batteries.

<details>
<summary>Expand Block G</summary>

**What this block produces:** `policy.onnx` — the decision-making network extracted from your skrl checkpoint — plus a flight script that runs it beside the detector on your laptop and commands a real Tello, and a set of recorded flights you use to improve both models.

**Nothing is retrained to get here.** The policy already commands stick channels, already decides at your drone's measured rate, already expects delayed readings, and already uses only telemetry a Tello reports. That is what 1.4 bought by coming before Chapter 3.

**What you still need to buy:** the attacker drone (a cheap toy quadcopter, C$50–70), spare propellers, a couple of batteries, and bright tape or a coloured shell for the target so the detector can find it at range. Total remaining spend is under C$100.

**The files created in this block:**

```
C:\projects\drone_pursuit\drone_pursuit\scripts\demo\export_policy.py   ← 7.1
C:\projects\drone_pursuit\drone_pursuit\scripts\demo\fly_real.py        ← 7.2
C:\projects\drone_pursuit\drone_pursuit\models\policy.onnx              ← output of 7.1
C:\projects\drone_pursuit\drone_pursuit\flights\                        ← flight recordings
```

---

# Chapter 7 — Export the Trained Models to the Real Drone and Fly the Chase

> 🔌 **DRONE HARDWARE REQUIRED** from 7.2 onward.

<details>
<summary>Expand Chapter 7</summary>

## 7.1 Export the Trained Policy from the Checkpoint to an ONNX File (≤1.5h)

> 💻 **No hardware needed for this subchapter** — it runs entirely in `env_drone`.

<details>
<summary>Expand 7.1</summary>

> **What this subchapter does:** the trained policy currently lives inside a skrl checkpoint that only Isaac Lab can open, and the flight script has no Isaac Lab installed. This subchapter extracts the decision-making network from that checkpoint and saves it as `policy.onnx`, then proves the exported file produces the same output as the checkpoint on identical input. It is the same operation you performed on the detector in 5.2, for the same reason.

### What a checkpoint contains, and which part you need

A checkpoint is a save file. During training, skrl periodically writes the state of the learning process to a `.pt` file so a run can be resumed or evaluated later — it is what `play.py` loads and what you resumed from in 3.3 Step 3.

| Inside it | Needed at flight time? |
|---|---|
| **The actor** — maps 17 observations to 4 commands | **Yes. This is the policy.** |
| **The critic** — estimated future reward, used only to compute PPO updates | No |
| **Optimiser state** — how PPO was adjusting weights mid-run | No |

Exporting means reaching past the training machinery and taking only the actor.

### Step 1 — Write the export script

<details>
<summary>Expand Step 1</summary>

> **Environment:** `env_drone`

*File to CREATE (new, empty file):* `C:\projects\drone_pursuit\drone_pursuit\scripts\demo\export_policy.py`

```python
# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\export_policy.py
# ── NEW file — the whole content, nothing to insert into.
# ── Reads : your Ch.3 checkpoint (command line)
# ── Writes: C:\projects\drone_pursuit\drone_pursuit\models\policy.onnx

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
dummy = torch.zeros(1, 17, device=runner.agent.device)      # 17 = your obs size
torch.onnx.export(
    wrapper, dummy, args.out,
    input_names=["obs"], output_names=["action"],
    dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    opset_version=17,
)
print(f"exported → {args.out}")

# ── SECTION 2 (Step 2) IS APPENDED BELOW THIS LINE ──────────────────────────
```

*Run from:* `any folder` — *the script lives in:* `C:\projects\drone_pursuit\drone_pursuit\scripts\demo\`
```bat
conda activate env_drone
python C:\projects\drone_pursuit\drone_pursuit\scripts\demo\export_policy.py --checkpoint C:\projects\drone_pursuit\drone_pursuit\logs\skrl\<run-folder>\checkpoints\best_agent.pt
```

**Why the mean and not a sample.** During training the policy draws its action from a distribution so it explores alternatives. In flight you want the same observation to produce the same command every time, so the export takes the centre of that distribution instead of drawing from it. Exporting the sampling version gives a drone that behaves slightly differently each run, which is almost impossible to debug.

⚠️ The dictionary key holding the mean varies between skrl versions. If that line fails, print `out` and inspect the third element — usually `mean_actions`, sometimes `net_output`.

</details>

### Step 2 — Prove the exported file matches the checkpoint

<details>
<summary>Expand Step 2</summary>

> **Environment:** `env_drone`

Feed the same random 17 numbers to both the wrapped checkpoint and the ONNX file and compare the outputs. A faulty export produces a drone that flies badly for reasons you would otherwise spend days attributing to the hardware, the detector or the delay model.

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\scripts\demo\export_policy.py`

```python
# ── FILE: ...\scripts\demo\export_policy.py ─────────────────────────────────
# ── SECTION: append at the very END of the file ─────────────────────────────

print(f"exported → {args.out}")          # ← last line from Step 1

# ▼▼▼ INSERT HERE! — the faithfulness check ▼▼▼
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
probe = torch.randn(1, 17, device=runner.agent.device)
from_torch = wrapper(probe).detach().cpu().numpy()
from_onnx = sess.run(None, {"obs": probe.cpu().numpy()})[0]
print("max difference:", np.abs(from_torch - from_onnx).max())
# ▲▲▲ END OF INSERT ▲▲▲
```

Below about 1e-4 means faithful.

</details>

### Step 3 — Confirm both models sit together in `models\`

<details>
<summary>Expand Step 3</summary>

> **Environment:** none needed — you are checking that two files exist.

*Folder to open:* `C:\projects\drone_pursuit\drone_pursuit\models\`

```
C:\projects\drone_pursuit\drone_pursuit\models\
    drone_detector.onnx     ← Chapter 5.2
    policy.onnx             ← this subchapter
```

These two files are the entire transfer. The environment, the reward function, the arena and Isaac Lab itself all stay behind.

</details>

> ✅ **Checkpoint 7.1**
> 1. `policy.onnx` exists and loads under onnxruntime
> 2. Its output matches the checkpoint to within 1e-4 on random input
> 3. Both models sit together in `models\`

</details>

---

## 7.2 Write the Flight Script and Bench-Test It with the Propellers Removed (≤1.5h)

> 🔌 **DRONE HARDWARE REQUIRED** — the Tello, powered on, with its propellers removed for the bench test.

<details>
<summary>Expand 7.2</summary>

> **What this subchapter does:** assembles the detector, the converter, the exported policy and the Tello SDK into the loop that flies the drone, and records every observation and command to CSV plus the video to MP4. It then runs that loop with the propellers off, so every part can be verified while nothing can hurt you — including the channel-mapping error that otherwise sends the drone sideways into a wall.

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

### Step 1 — Create the flight-recording folder

<details>
<summary>Expand Step 1</summary>

> **Environment:** none needed — this is a folder command.

*Run from:* `any folder`
```bat
mkdir C:\projects\drone_pursuit\drone_pursuit\flights
```

**Do this before flying, not after.** The script opens its log file immediately after takeoff, and Python's `open()` fails if the parent folder is missing — which would crash the script with the drone already in the air.

</details>

### Step 2 — Write the flight script

<details>
<summary>Expand Step 2</summary>

> **Environment:** `env_drone` to run it; none needed to write it.

*File to CREATE (new, empty file):* `C:\projects\drone_pursuit\drone_pursuit\scripts\demo\fly_real.py`

```python
# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\fly_real.py
# ── NEW file — the whole content, nothing to insert into.
# ── Imports: vision_bridge.py, in this SAME folder (scripts\demo\)
# ── Loads  : C:\projects\drone_pursuit\drone_pursuit\models\drone_detector.onnx
# ──          C:\projects\drone_pursuit\drone_pursuit\models\policy.onnx
# ── Writes : C:\projects\drone_pursuit\drone_pursuit\flights\flight_<stamp>.csv / .mp4
# ── NOTE   : this script NEVER imports Isaac Lab.

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
drone.takeoff()                  # ◄── COMMENT OUT for the Step 3 bench test
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

        # 4 — assemble the 17 numbers, in the SAME order as 3.1 Part G
        obs = np.concatenate([self_state, readings, prev_action])[None].astype(np.float32)

        # 5 — DECIDE.  ◄── THIS LINE IS THE TRAINED POLICY RUNNING.
        #     `policy` is the ONNX file from 7.1. Everything Chapters 2 and 3
        #     trained lives inside it.
        action = np.clip(policy.run(None, {"obs": obs})[0][0], -1.0, 1.0)
        prev_action = action.copy()

        # 6 — ACT.  Tello channels: left/right, forward/back, up/down, yaw (−100..100)
        #     ◄── THIS MAPPING IS A GUESS until Step 3 verifies it.
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
    drone.land()                 # ◄── COMMENT OUT for the Step 3 bench test
    drone.streamoff()
    log.close()
    video.release()
    print(f"flight recorded → flight_{stamp}.csv / .mp4")
```

**Three things worth understanding in that script:**

**The channel mapping in step 6 is a guess until verified.** Which action index drives which stick depends on the order you assigned in 3.1 Part A. Get it wrong and the drone translates sideways when the policy meant forward.

**Every flight is recorded automatically** — a CSV of every observation and command plus the raw video. This costs nothing during flight and is the entire input to 7.4. A flight you did not record teaches you only what you happened to notice while it was happening.

**The Tello lands itself after 15 seconds without a command**, so a crashed script does not leave a drone flying.

</details>

### Step 3 — Bench-test with the propellers removed

<details>
<summary>Expand Step 3</summary>

> **Environment:** `env_drone` — 🔌 **Tello powered on, propellers OFF.**

Take the propellers off, then disable takeoff and landing:

*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\scripts\demo\fly_real.py`

```python
# ── FILE: ...\scripts\demo\fly_real.py ──────────────────────────────────────
# ── SECTION: two lines, one before the loop and one in the finally block ────

# BEFORE the loop:
# drone.takeoff()          ◄── COMMENT OUT for the bench test
time.sleep(2)

# ... and inside `finally:` ...
    drone.send_rc_control(0, 0, 0, 0)
    # drone.land()         ◄── COMMENT OUT for the bench test
    drone.streamoff()
```

*Run from:* `any folder` — *the script lives in:* `C:\projects\drone_pursuit\drone_pursuit\scripts\demo\`
```bat
conda activate env_drone
python C:\projects\drone_pursuit\drone_pursuit\scripts\demo\fly_real.py
```

Hold the attacker drone in front of the camera by hand and check all six rows:

| Check | What you should see |
|---|---|
| Detection works | `visible` is 1 with the attacker in view, 0 when hidden |
| Bearings correct | Move it left; `bx` moves consistently one way — note which |
| Size responds | Move it closer; `asz` grows |
| Commands sensible | Attacker to the left → the left/right command is non-zero and correctly signed |
| Rate holds | No "loop overran" messages |
| Emergency stop | Both ENTER and `x` behave as expected |

**If commands point the wrong way, fix the channel mapping now**, before anything spins. Restore `takeoff()` and `land()` afterwards.

</details>

> ✅ **Checkpoint 7.2**
> 1. Bench run passes all six checks
> 2. A CSV and MP4 appeared in `C:\projects\drone_pursuit\drone_pursuit\flights\`
> 3. You know which action index drives which direction

</details>

---

## 7.3 Fly the Chase in Six Staged Tests (≤1.5h per stage)

> 🔌 **DRONE HARDWARE REQUIRED** — Tello flying with propellers and guards, plus the target drone from Stage 3 onward.

<details>
<summary>Expand 7.3</summary>

> **What this subchapter does:** runs the flight script for real, in six stages that each add exactly one new element. Staging matters because a failure at stage 3 names its own cause — the one thing stage 3 added — while jumping straight to a flying attacker turns any failure into an unanswerable question and often a broken drone.

### Before every session

Propeller guards on. Eye protection. Clear space. Nobody else present. Battery above 30%. Hand near the keyboard.

### Step 1 — Work through the six stages in order

<details>
<summary>Expand Step 1</summary>

> **Environment:** `env_drone` — 🔌 drone flying.

*Run from:* `any folder` — *the script lives in:* `C:\projects\drone_pursuit\drone_pursuit\scripts\demo\`
```bat
conda activate env_drone
python C:\projects\drone_pursuit\drone_pursuit\scripts\demo\fly_real.py
```

**Stage 1 — Hover only.** Restore `takeoff()`, but force the policy output to zero so the drone just hovers. In `fly_real.py`, immediately after the `action = np.clip(...)` line, temporarily add `action[:] = 0.0`. Checks that the loop holds 20 Hz while flying, that video keeps up, and that landing works. Two minutes.

**Stage 2 — Live policy, no attacker.** Remove the `action[:] = 0.0` line and let the policy run with nothing to detect. It should sit roughly still, holding its last reading with `visible` at 0. If it wanders off aggressively, the lost-sight behaviour is wrong — stop and confirm the converter is holding values rather than zeroing them.

**Stage 3 — Stationary attacker on a stand, two metres away.** The defender should approach and stop at capture distance. This is the first real test of the transfer.

**Stage 4 — Hand-carried attacker.** Walk it slowly across the space.

**Stage 5 — Flying attacker, slow.** Both airborne, attacker near walking pace.

**Stage 6 — Increase attacker speed.** The Chapter 3.3 curriculum again, this time in hardware.

</details>

### Step 2 — Diagnose failures from the log, not from memory

<details>
<summary>Expand Step 2</summary>

> **Environment:** none needed — you are reading CSV files.

*Folder to open:* `C:\projects\drone_pursuit\drone_pursuit\flights\`

| What you see | Most likely cause | What to check in the CSV |
|---|---|---|
| Oscillates while hovering | Stand-in stabiliser from 3.1 does not match the Tello's | Commands alternating sign rapidly |
| Overshoots repeatedly | Real delay exceeds the range you trained for | Re-measure delay; widen `obs_delay_*` |
| Drifts to one side | Real drift larger than the randomised range | `vx`/`vy` biased while commands are near zero |
| Stops or wanders when the target moves | Detector losing the target | `visible` dropping to 0 often |
| Works close, fails far | Detector cannot resolve a small target | `asz` small, `visible` flickering |
| Flies the wrong way entirely | Channel mapping wrong | Commands correctly signed but on the wrong axis |

Almost every row above is answered by the CSV rather than by what you saw from across the room. Read the log before forming a theory.

</details>

### On legality

Indoors in a private space is the least regulated situation. Outdoors in Canada, drone operation falls under Transport Canada rules, and deliberately flying one aircraft toward another is not routine. Check current requirements before flying outdoors at all.

> ✅ **Checkpoint 7.3 — SIM-TO-REAL VALIDATED**
> A policy trained entirely in simulation flies a real drone, finds a real target through a real camera, and closes on it. Record it.

</details>

---

## 7.4 Improve the Detector and the Simulation Using Recorded Flight Data (≤1.5h per cycle)

> 🔌 **DRONE HARDWARE REQUIRED** — you need the recordings from 7.3, and each cycle ends with another flight.

<details>
<summary>Expand 7.4</summary>

> **What this subchapter does:** turns the CSVs and videos recorded in 7.2 into two concrete improvements — a detector fine-tuned on real footage of your real target, and simulator constants corrected to match how your drone actually responded. Each cycle makes the simulation more truthful, so the next policy trained in it transfers better. This is the loop professional sim-to-real work runs continuously.

### Why the policy is not trained on flight data directly

PPO needs millions of environment steps; one flight gives a few thousand, and the informative ones are crashes. Nobody trains a task like this on real hardware. The improvements loop through the simulator instead:

```
   real flight ──► what was wrong? ──► fix the SIMULATION ──► retrain ──► fly again
                                            ▲
                                  (the sim becomes more truthful each
                                   cycle, so the policy transfers better)
```

This is **system identification**: using measurements of the real system to correct your model of it.

**What is automatic and what is not.** Collection is automatic and complete — every observation, command, loop time and video frame lands in `flights\` on every run. Interpretation is manual: you read the logs, decide what was wrong, and change a parameter. No code here adjusts the simulator on its own, and writing one would be a research project. What you have is complete data plus the checklist below, which is what makes the manual step twenty minutes rather than guesswork.

### Step 1 — Fine-tune the detector on real footage (largest gain, least effort)

<details>
<summary>Expand Step 1</summary>

> **Environment:** `drone_vision` for the YOLO commands; the frame extraction itself needs none.

Your detector has only ever seen synthetic Crazyflie renders. The recorded videos are real footage of the real target drone, which is the single biggest difference between Chapter 6's conditions and Chapter 7's.

**1 — Extract frames** — every tenth, since consecutive frames at 20 Hz are nearly identical:

*Run from:* `any folder` — *source videos in:* `C:\projects\drone_pursuit\drone_pursuit\flights\`
```bat
mkdir C:\projects\drone_pursuit\drone_pursuit\data\real\images
```

**2 — Pre-label them with the detector you already have**, then correct what it got wrong. You never label from a blank slate:

*Run from:* `C:\projects\drone_pursuit\drone_pursuit`
```bat
conda activate drone_vision
yolo detect predict model=C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\weights\best.pt source=C:\projects\drone_pursuit\drone_pursuit\data\real\images save_txt=True save_conf=True conf=0.25
```

`save_txt` writes YOLO-format labels beside each image. Load those into any labelling tool and you are reviewing boxes rather than drawing them — five to ten times faster.

**3 — Propagate across nearby frames.** Because these come from continuous video, a box on frame *n* is nearly right for frame *n+1*. CVAT, Label Studio and Roboflow all interpolate between two corrected frames.

**4 — Spend the effort where the detector failed.** Two automatic ways to find those frames: any pre-labelled box below about 0.4 confidence, and any row in the flight CSV where `visible` flickers between 0 and 1. Pull those timestamps and extract the matching frames. Frames where the attacker is large and obvious teach the network almost nothing.

Reviewing 200–400 pre-labelled frames takes roughly 45 minutes, against several hours drawing from scratch. It cannot be fully automated — an auto-labeller good enough to do it unsupervised would already be the detector you are trying to build.

**5 — Fine-tune from your synthetic model**, not from scratch. You also need a `drone.yaml` for this new folder, identical to 4.3's but with `path:` pointing at `data\real`:

*Run from:* `C:\projects\drone_pursuit\drone_pursuit`
```bat
yolo detect train data=C:\projects\drone_pursuit\drone_pursuit\data\real\drone.yaml model=C:\projects\drone_pursuit\drone_pursuit\runs\detect\drone_v1\weights\best.pt epochs=40 imgsz=640 name=drone_real
```

**6 — Re-export to ONNX** at `imgsz=480,640`, copy over `models\drone_detector.onnx`, and re-measure `capture_ang_size`. A target of different width at the same distance fills a different share of the frame, so the old threshold would declare capture at the wrong distance.

</details>

### Step 2 — Correct the simulation constants from the flight logs

<details>
<summary>Expand Step 2</summary>

> **Environment:** none needed — you are reading logs and editing cfg values.

*Logs to read:* `C:\projects\drone_pursuit\drone_pursuit\flights\flight_<stamp>.csv`
*File to edit:* `C:\projects\drone_pursuit\drone_pursuit\source\drone_pursuit\drone_pursuit\tasks\direct\quadcopter\quadcopter_env.py`

```python
# ── FILE: ...\tasks\direct\quadcopter\quadcopter_env.py ─────────────────────
# ── SECTION: class QuadcopterEnvCfg — four numbers you already added ────────
# ── Change them IN PLACE; nothing new is inserted here.

    max_speed = 2.0        # ◄── LOWER if max forward command produced less speed
    vel_gain = 3.0         # ◄── RAISE if the real drone accelerated faster than sim
    obs_delay_max = 5      # ◄── RAISE if the drone overshoots consistently
    # and in _reset_idx, widen the 0.15 multiplier on self._drift if the real
    # residual velocity at zero command exceeds the randomised range
```

**The stand-in stabiliser.** Find a segment where you commanded a steady forward value and plot how `vx` rose. If the real drone accelerated faster than the simulation does, raise `vel_gain`; slower, lower it. Ten minutes comparing two plots beats any amount of guessing at 3.1's approximation.

**The drift range.** Find segments with near-zero commands and read the residual `vx`/`vy`. That is your Tello's true drift. Widen the randomisation range in 3.3 Step 0 so it comfortably contains that value.

**The delay range.** Consistent overshoot means the real delay exceeds what you trained for. Widen `obs_delay_max`.

**The speed limits.** If maximum forward command produced less speed than `max_speed` assumes, lower `max_speed` — otherwise the policy plans manoeuvres around performance the drone does not have.

Then retrain (3.3 Step 1) and fly again (7.3).

</details>

### What good iteration looks like

| Cycle | Typical change | Typical result |
|---|---|---|
| 1 | Detector fine-tuned on real frames | Far fewer lost-sight events |
| 2 | `vel_gain` and drift corrected from logs | Smoother approach, less weaving |
| 3 | Delay range widened | Overshoot reduced |
| 4 | Attacker speed increased | A harder task, honestly passed |

**Change one thing per cycle.** Two changes and a better result tells you nothing about which helped — the same rule that governed Chapters 2 and 3, now applied to hardware.

> ✅ **Checkpoint 7.4**
> 1. At least one detector fine-tune from real footage, with measurably fewer lost-sight events
> 2. At least one simulation parameter corrected from flight logs
> 3. A second flight session visibly better than the first

</details>

</details>

</details>

---

# Appendix A — Check Versions and Diagnose Compatibility Symptoms

<details>
<summary>Expand Appendix A</summary>

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
| ONNX output shape ≠ (1,5,6300) | exported the wrong .pt (pretrained 80-class instead of your best.pt) |
| Chase works in Ch.3, degrades badly in Ch.6 | field of view mismatch — `TiledCameraCfg` focal length must match `cam_focal_mm`/`cam_aperture_mm` in the env cfg |
| Defender flies off when attacker leaves frame | `visible` flag not wired, or previous readings not held on a miss |
| Ch.7: policy fine in sim, oscillates on hardware | real delay exceeds the trained range, or the stand-in stabiliser in 3.1 does not match the drone — re-measure, widen `obs_delay_*`, retrain |
| Ch.7: video latency near one second | buffering, not the drone — set the capture buffer size to 1 and read in a thread that discards stale frames |
| Ch.7: works close, fails far | detector cannot resolve a small target — add far-range training frames or brighter markers |
| Ch.7: drone flies sideways when it should go forward | action-to-stick channel mapping wrong in the flight script — verify on the bench with propellers off |
| Ch.7: ONNX policy disagrees with the checkpoint | the export wrapper extracted a sampled action instead of the distribution mean |
| Ch.7: "loop overran" printing constantly | detector or frame conversion too slow for the control rate — lower the detector input size |

</details>

# Appendix B — Plan Your Time, Hardware and Dependencies

<details>
<summary>Expand Appendix B</summary>

| Block | Subchapter | Hardware | ~Time | Depends on |
|---|---|---|---|---|
| A | 1.0 Isolate the project | 💻 | 1.5h | — |
| A | 1.1 Understand the flight code | 💻 | 1.5h | 1.0 |
| A | 1.2 External project | 💻 | 1.5h | 1.0, 1.1 |
| A | 1.3 Vision env + boundary tests | 💻 | 1.5h | 1.0 |
| A | **1.4 Set up and measure the Tello** | 🔌 | **1.5h** | **drone in hand** |
| B | 2.1 Spawn the attacker in every env | 💻 | 1.5h | 1.2 |
| B | 2.2 Move the attacker along a randomised path | 💻 | 1.5h | 2.1 |
| C | 3.1 Define commands and observations | 💻 | 1.5h | 2.2, **1.4** |
| C | 3.2 Write the reward and episode endings | 💻 | 1.5h | 3.1 |
| C | 3.3 Randomise, train, read the curves | 💻 | 1.5h (+ GPU hours) | 3.2 |
| D | 4.1 Build the Replicator SDG script | 💻 | 1.5h | 1.2 (parallel to Block C) |
| D | 4.2 Randomise and generate 2500 frames | 💻 | 1.5h | 4.1 |
| D | 4.3 Convert to YOLO format and verify | 💻 | 1.5h | 4.2 |
| E | 5.1 Fine-tune YOLOv8-nano | 💻 | 1.5h | 4.3, 1.3 |
| E | 5.2 Test on arena frames and export ONNX | 💻 | 1.5h | 5.1 |
| F | 6.1 Write the bounding-box converter | 💻 | 1.5h | 5.2 |
| F | 6.2 Run the full loop in simulation | 💻 | 1.5h+ | 3.3, 6.1 |
| F | 6.3 Review and choose the next extension | 💻 | — | — |
| G | 7.1 Export the policy to ONNX | 💻 | 1.5h | 6.2 |
| G | 7.2 Write the flight script and bench-test | 🔌 propellers off | 1.5h | 7.1 |
| G | 7.3 Fly in six staged tests | 🔌 flying | 1.5h per stage | 7.2 |
| G | 7.4 Improve from real flight data | 🔌 flying | 1.5h per cycle | 7.3 |

**Buy the hardware early.** Subchapter 1.4 needs the Tello in hand, and everything from Chapter 3 onward is built on its measurements. Ordering the drone while working through 1.0–1.3 keeps the sequence unbroken. The target drone is not needed until 7.3.

</details>

## Sources

<details>
<summary>Expand sources</summary>

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

</details>

---

*Built for your learning style: big picture → detail, one hard thing at a time, checkpoints before commitments, and every tool proven compatible before you bet hours on it. Good hunting.* 🛩️
