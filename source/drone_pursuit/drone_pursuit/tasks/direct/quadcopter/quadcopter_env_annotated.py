# =============================================================================
#  quadcopter_env.py — ANNOTATED READING COPY
# =============================================================================
#  This is Isaac Lab's built-in Crazyflie hover task, with explanatory comments
#  added. The code is unchanged; only comments were added.
#
#  DO NOT run this file or import it. It is for reading. The real file lives at:
#     %ISAACLAB%\source\isaaclab_tasks\isaaclab_tasks\direct\quadcopter\
#
#  ---------------------------------------------------------------------------
#  WHAT THIS FILE IS
#  ---------------------------------------------------------------------------
#  One complete reinforcement-learning task, in the "Direct" workflow: a
#  Crazyflie quadcopter learns to fly to a randomly placed point and hover
#  there. Everything the task needs is here — the drone, the ground, the
#  lighting, what the policy senses, what it controls, how it is scored, and
#  when an attempt ends.
#
#  ---------------------------------------------------------------------------
#  WHY YOU ARE READING IT
#  ---------------------------------------------------------------------------
#  The drone-pursuit project copies this file and changes one idea: the fixed
#  goal point becomes a second drone that moves. Almost every method below gets
#  edited in Chapter 2, so reading it now means those edits are modifications
#  to something you understand rather than instructions you follow blindly.
#
#  ---------------------------------------------------------------------------
#  HOW THE PIECES FIT — the loop that runs thousands of times per second
#  ---------------------------------------------------------------------------
#
#     ONE ENVIRONMENT STEP (the policy makes one decision)
#     ┌──────────────────────────────────────────────────────────────┐
#     │                                                              │
#     │  _pre_physics_step(actions)   convert 4 policy numbers into  │
#     │        │                      forces — runs ONCE             │
#     │        ▼                                                     │
#     │  _apply_action()  ─► physics tick  ┐                         │
#     │  _apply_action()  ─► physics tick  ├─ runs `decimation`      │
#     │                                    ┘  times (here: 2)        │
#     │        │                                                     │
#     │        ▼                                                     │
#     │  _get_observations()   what the policy senses next step      │
#     │  _get_rewards()        how well it just did (training only)  │
#     │  _get_dones()          did this attempt end?                 │
#     │        │                                                     │
#     │        ▼                                                     │
#     │  _reset_idx(finished_envs)   restart only those that ended   │
#     └──────────────────────────────────────────────────────────────┘
#
#  Called once at startup, before any of that: _setup_scene() and __init__().
#
#  ---------------------------------------------------------------------------
#  THE ONE THING TO CARRY AWAY
#  ---------------------------------------------------------------------------
#  Every method below operates on ALL environments simultaneously, as tensors.
#  With 4096 drones training in parallel there are no Python loops over drones —
#  `self._robot.data.root_pos_w` is a (4096, 3) table, and one line of tensor
#  arithmetic updates every drone at once. If you ever find yourself writing
#  `for env in range(num_envs)`, you have left the intended path and training
#  will slow by orders of magnitude.
# =============================================================================

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


# =============================================================================
#  CLASS 1 of 3 — QuadcopterEnvWindow
# =============================================================================
#  WHAT IT IS
#     The on-screen debug panel you see when running without --headless.
#
#  WHY IT IS HERE
#     It adds a checkbox that toggles the little cubes marking each drone's
#     goal position. Purely a viewing aid.
#
#  WHAT IF IT WERE ABSENT
#     Nothing would break. Training is identical; you would simply lose the
#     toggle and always see (or never see) the goal markers.
#
#  RELEVANCE TO THE PURSUIT PROJECT
#     Low. You can leave it untouched. In Chapter 2 the goal marker becomes
#     unnecessary anyway, because the target is a visible second drone.
# =============================================================================
class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        # Add one custom UI element on top of the standard debug panel.
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


# =============================================================================
#  CLASS 2 of 3 — QuadcopterEnvCfg          ◄── THE SETTINGS
# =============================================================================
#  WHAT IT IS
#     Every tunable value for this task, in one place: how long an attempt
#     lasts, how many drones run in parallel, how strong the rotors are, and
#     how behaviour is scored.
#
#  WHY IT IS SEPARATE FROM THE ENVIRONMENT CLASS
#     Two reasons that matter in practice.
#     1. You can change behaviour without touching logic. Doubling the reward
#        for reaching the goal is a number edit, not a code edit.
#     2. Isaac Lab can create variants by copying this object and overriding a
#        field — which is exactly how Chapter 2 adds a second drone using the
#        same Crazyflie recipe with a different spawn path.
#
#  WHAT @configclass DOES
#     It is Isaac Lab's decorator built on Python dataclasses. It turns the
#     plain attributes below into a proper configuration object that can be
#     copied, nested, and overridden from the command line. Without it these
#     would be ordinary class attributes shared between every instance — so
#     changing one environment's settings would change them all.
#
#  WHAT IF THIS CLASS DID NOT EXIST
#     The numbers would be scattered as literals through the environment code.
#     Tuning would mean hunting through methods, and every variant would be a
#     copy-paste of the whole file.
# =============================================================================
@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):

    # -------------------------------------------------------------------------
    # EPISODE AND CONTROL TIMING
    # -------------------------------------------------------------------------
    episode_length_s = 10.0
    #   How long one attempt lasts before it is cut off, in simulated seconds.
    #   WHY 10: long enough to fly a few metres and settle; short enough that a
    #   hopeless attempt is not wasting compute. Too short and the drone never
    #   experiences success; too long and each attempt costs more to learn from.

    decimation = 2
    #   How many physics ticks pass per policy decision.
    #   Physics runs at 100 Hz (see sim.dt below), so the policy decides at
    #   100/2 = 50 Hz.
    #   WHY NOT 1: a policy deciding at every physics tick would be twice as
    #   expensive to run for no benefit — real flight controllers do not need
    #   100 Hz high-level decisions, and low-level stability is handled by the
    #   physics itself. This mirrors real drones, where an outer control loop
    #   runs slower than the inner one.

    action_space = 4
    #   The policy outputs 4 numbers. See _pre_physics_step for what they mean.
    #   This number must match, or the network and the environment disagree
    #   about tensor shapes and the run fails immediately.

    observation_space = 12
    #   The policy receives 12 numbers. See _get_observations for the contents.
    #   IMPORTANT FOR CHAPTER 3: change what you observe and you MUST change
    #   this number. A mismatch is one of the most common errors when editing
    #   a Direct task, and it surfaces as a confusing shape error at startup.

    state_space = 0
    #   Used only for algorithms that give a critic extra information the actor
    #   does not see. PPO here uses none, so it is zero.

    debug_vis = True
    ui_window_class_type = QuadcopterEnvWindow
    #   Visual aids only. No effect on learning.

    # -------------------------------------------------------------------------
    # THE PHYSICS SIMULATION
    # -------------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        #   Physics timestep: 100 updates per simulated second.
        #   WHY IT MATTERS: too large and fast-moving objects tunnel through
        #   each other or oscillate; too small and simulation crawls. 100 Hz is
        #   a standard compromise for a light, agile vehicle like a Crazyflie.
        render_interval=decimation,
        #   Draw a frame only as often as the policy acts. Rendering more often
        #   would cost time and show nothing new.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            #   restitution 0 = no bounce. A drone that hit the floor and
            #   bounced would produce strange training data; here a crash is a
            #   crash. _get_dones ends the attempt anyway.
        ),
    )

    # -------------------------------------------------------------------------
    # THE GROUND
    # -------------------------------------------------------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    #   WHY A GROUND AT ALL: the drone needs a floor to crash into, so that
    #   "too low" is a real failure the policy learns to avoid. It also gives
    #   the viewport a visual reference.
    #
    #   THE HIDDEN IMPORTANT PART: the terrain importer also computes
    #   `env_origins` — the world position of each parallel environment. With
    #   4096 environments spaced 2.5 m apart, env 0 might sit at (0, 0) and env
    #   1 at (2.5, 0). Any world-frame position must have its environment's
    #   origin added, or every drone would fly toward the same physical spot.
    #   You will see `+= self._terrain.env_origins[...]` several times below.

    # -------------------------------------------------------------------------
    # HOW MANY COPIES, AND HOW FAR APART
    # -------------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )
    #   num_envs: 4096 independent drones learning simultaneously.
    #   WHY SO MANY: PPO learns from experience, and 4096 drones generate 4096
    #   times the experience per wall-clock second. This is the single biggest
    #   reason GPU-based simulators exist. Reduce it if you run out of VRAM;
    #   training still works, just slower.
    #
    #   env_spacing: metres between neighbouring environments. Must exceed how
    #   far a drone can travel, or drones from adjacent environments become
    #   visible to each other — which matters enormously once cameras are added
    #   in the pursuit project.
    #
    #   replicate_physics / clone_in_fabric: performance settings that let the
    #   engine share one physics description across all copies instead of
    #   building 4096 separate ones.

    # -------------------------------------------------------------------------
    # THE ROBOT
    # -------------------------------------------------------------------------
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    #   CRAZYFLIE_CFG is a ready-made recipe living in isaaclab_assets. It says
    #   which USD file to load (a model of the real Bitcraze Crazyflie, hosted
    #   on NVIDIA's asset server), what its mass and physics properties are, and
    #   where it starts.
    #
    #   WHY A SHARED RECIPE: without it, every task using a Crazyflie would
    #   hand-write the USD path and physics properties, and they would drift
    #   apart. One definition means one place to fix.
    #
    #   `.replace(...)` copies the recipe and overrides one field. The original
    #   is untouched, so other tasks are unaffected — this is how the pursuit
    #   project adds a second drone in Chapter 2:
    #       attacker = CRAZYFLIE_CFG.replace(prim_path=".../Attacker", ...)
    #
    #   THE `env_.*` PART is a regular expression. It matches /World/envs/env_0,
    #   env_1, env_2 and so on, so one line spawns a drone into all 4096
    #   environments. Nothing is written per-environment by hand.

    thrust_to_weight = 1.9
    #   Maximum thrust as a multiple of the drone's own weight. At 1.9 the
    #   drone can climb briskly; at exactly 1.0 it could only just hover and
    #   never accelerate upward; below 1.0 it could never leave the ground.
    #   Real Crazyflies sit near this figure.

    moment_scale = 0.01
    #   Converts the policy's unitless turning outputs into Newton-metres.
    #   Larger values make the drone twitchier and harder to stabilise; smaller
    #   values make it sluggish and unable to correct quickly.

    # -------------------------------------------------------------------------
    # REWARD WEIGHTS — how behaviour is scored
    # -------------------------------------------------------------------------
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    #   NEGATIVE values are penalties, POSITIVE are rewards. Read the three
    #   together as a sentence: "get close to the goal (big reward), but do not
    #   move fast (small penalty) and do not spin (smaller penalty)."
    #
    #   WHY THE PENALTIES EXIST: with only the distance reward, the fastest way
    #   to gain points is to hurl yourself toward the goal at maximum speed and
    #   overshoot repeatedly. The velocity penalties buy stability — the drone
    #   learns to arrive and stay rather than to oscillate.
    #
    #   WHY THEIR RELATIVE SIZES MATTER MORE THAN THEIR ABSOLUTE SIZES: 15.0
    #   against 0.05 means reaching the goal is worth roughly 300 times more
    #   than holding still. Make the penalties too large and the drone learns
    #   that the safest behaviour is not to move at all.


# =============================================================================
#  CLASS 3 of 3 — QuadcopterEnv             ◄── THE TASK ITSELF
# =============================================================================
#  WHAT IT IS
#     The environment: it builds the scene, converts policy output into forces,
#     produces observations, computes rewards, decides when attempts end, and
#     resets them.
#
#  WHAT DirectRLEnv GIVES IT
#     The parent class runs the loop, manages the simulation, counts episode
#     steps, and talks to the RL library. This class only fills in the
#     task-specific methods. The alternative workflow — Manager-based — splits
#     these same responsibilities into configurable term lists, which is better
#     for reusing components across many task variants but adds indirection.
#     Direct keeps everything in one readable file, which suits a task with
#     unusual actuation and custom logic.
# =============================================================================
class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    # -------------------------------------------------------------------------
    #  __init__ — runs ONCE at startup
    # -------------------------------------------------------------------------
    #  WHAT IT DOES
    #     Two things, both done once so they never have to be done again.
    #     First, it reserves the memory the task will write into for the rest of the run. A "tensor" here just means a table of numbers held on the GPU — one row per drone, a few columns per row. For example, _thrust is a table with 4096 rows (one drone each) and 3 columns (force sideways, force forward, force up). This method creates those tables filled with zeros. It is not putting meaningful values in them — it is claiming the space, the way you might set out 4096 empty forms before an event rather than fetching a blank one each time someone walks in.
    #     Second, it looks up facts that will never change: how much the drone weighs, and which part of the drone the thrust should push against. Both require asking the simulation, which is slow, and both give the same answer every time.

    # WHY DO THIS HERE

    # Because everything else in this file runs constantly — fifty times per simulated second, for every one of thousands of drones. Reserving memory and looking things up are slow operations. Done inside those methods, they would take more time than the actual flying. Done once here, the later methods only overwrite numbers in tables that already exist, which is fast.
    # This is why you see torch.zeros(...) in this method and nowhere else in the file.
    #
    #  WHAT IF IT WERE ABSENT
    #     Every method would have to allocate its own scratch memory on every
    #     call, and training would slow dramatically for no behavioural gain.
    # -------------------------------------------------------------------------
    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        #   The parent constructor calls _setup_scene() internally, so by the
        #   line below, self._robot already exists. Order matters here.

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        #   Shape (num_envs, 4). The most recent policy output for each drone.

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        #   Shape (num_envs, 1, 3): for each environment, for 1 body, a 3-axis
        #   vector. The middle dimension is 1 because force is applied to a
        #   single body — the drone's centre — not to each rotor.
        #   `device=self.device` puts them on the GPU. A tensor left on the CPU
        #   would force a copy every step, which is a common and silent cause
        #   of poor performance.

        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        #   THIS DOES NOT DEFINE THE GOAL. It reserves a table of num_envs rows
        #   by 3 columns, filled with placeholder zeros. The real goal values
        #   are written in _reset_idx, once per attempt, per environment.
        #   The `_w` suffix means world frame — absolute coordinates, which is
        #   why environment origins get added when the values are written.
        #
        #   THIS IS THE LINE THE PURSUIT PROJECT REPLACES. In Chapter 2 the
        #   target stops being a point written once per attempt and becomes a
        #   second drone whose position is read fresh every step.

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_goal"]
        }
        #   Running totals of each reward component, per environment, for the
        #   current attempt. Purely for reporting: _reset_idx averages them and
        #   sends them to TensorBoard.
        #   WHY BOTHER: when total reward stalls, these tell you which
        #   component is responsible. Without them you would see one number
        #   going nowhere and have no idea why.

        self._body_id = self._robot.find_bodies("body")[0]
        #   Looks up the index of the part named "body" in the Crazyflie USD.
        #   Forces are applied to this index. Looked up once because the answer
        #   never changes, and the lookup involves string matching.

        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        #   Weight in Newtons = mass x gravity. Read from the simulation rather
        #   than hardcoded, so swapping in a heavier drone automatically scales
        #   the thrust — you would not have to remember to update a constant.

        self.set_debug_vis(self.cfg.debug_vis)

    # -------------------------------------------------------------------------
    #  _setup_scene — runs ONCE, before the simulation starts
    # -------------------------------------------------------------------------
    #  WHAT IT DOES
    #     Builds the world: creates the drone, the ground, the lighting, then
    #     duplicates the whole arrangement into thousands of copies.
    #
    #  WHY IT IS A SEPARATE METHOD
    #     Scene construction must happen before physics begins. The parent
    #     class calls it at exactly the right moment; you only supply contents.
    #
    #  WHAT IF IT WERE ABSENT
    #     There would be no drone and no ground — an empty stage.
    # -------------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        #   Turns the recipe into an actual object. An "articulation" is a body
        #   made of linked parts — from Latin articulus, a joint.

        self.scene.articulations["robot"] = self._robot
        #   Registers it with the scene so Isaac Lab keeps its data buffers
        #   updated each step. WITHOUT THIS LINE the drone would exist visually
        #   but `self._robot.data.root_pos_w` would never refresh — a subtle
        #   failure where nothing errors and nothing moves sensibly.
        #   Chapter 2 adds a second registration here for the attacker.

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        #   The terrain needs to know how many environments to lay out and how
        #   far apart, because it is what computes env_origins.

        self.scene.clone_environments(copy_from_source=False)
        #   THE LINE THAT MAKES PARALLEL TRAINING POSSIBLE. Everything created
        #   under the /World/envs/env_.* pattern is replicated num_envs times.
        #   You describe one environment; you get thousands.
        #   copy_from_source=False means the copies share the source geometry
        #   rather than duplicating it in memory — much lighter.

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        #   On CPU, neighbouring environments would otherwise collide with each
        #   other. On GPU this is handled automatically.

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        #   A dome light — even illumination from all directions, like an
        #   overcast sky. Irrelevant to physics, but essential once cameras
        #   enter the picture: in the pursuit project, lighting becomes
        #   something you deliberately randomise so the detector does not learn
        #   one specific lighting condition.

    # -------------------------------------------------------------------------
    #  _pre_physics_step — runs ONCE per policy decision
    # -------------------------------------------------------------------------
    #  WHAT IT DOES
    #     Translates the policy's four unitless numbers into physical forces.
    #
    #  WHY IT IS NEEDED
    #     A neural network outputs numbers with no units. PhysX needs Newtons
    #     and Newton-metres. Someone must define the mapping, and this is it.
    #     The numbers mean thrust and torque *because this method says so* —
    #     nothing in the network knows or cares.
    #
    #  WHY IT IS SEPARATE FROM _apply_action
    #     This runs once per decision; _apply_action runs on every physics tick
    #     (twice as often here). Converting once and applying repeatedly avoids
    #     redundant arithmetic and keeps a clean line between "what did the
    #     policy decide" and "how often was that decision applied".
    #
    #  WHAT IF IT WERE ABSENT
    #     Raw network outputs would reach PhysX as forces: no scaling to the
    #     drone's weight, no clamping of extreme early-training values, and
    #     negative thrust pulling the drone through the floor.
    # -------------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        #   clamp: force every value into [-1, 1]. Early in training the network
        #   is essentially random and can emit large values; without clamping
        #   those become enormous forces, the drone leaves the arena in one
        #   step, and the resulting experience teaches nothing.
        #   clone: work on a copy, so later code cannot corrupt what the RL
        #   library stored for its own update.

        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        #   ACTION 0 BECOMES UPWARD THRUST.
        #
        #   Indexing: [:, 0, 2] means all environments, body 0, axis 2 (= z =
        #   up). Axes 0 and 1 stay zero: a quadcopter cannot push itself
        #   sideways. It moves horizontally by TILTING, so that its upward
        #   thrust points partly sideways.
        #
        #   The (+1)/2 remap: converts the range [-1, 1] into [0, 1].
        #   WHY: rotors can spin or stop; they cannot pull downward. Without
        #   the remap, an action of -1 would produce a negative force sucking
        #   the drone into the ground — physically impossible, and the policy
        #   would happily learn to exploit it.
        #
        #   Worked example with a 27 g Crazyflie (weight about 0.265 N):
        #       action  0.5  ->  (0.5+1)/2 = 0.75  ->  0.75 x 1.9 x 0.265
        #                                            =  0.378 N upward
        #   That exceeds its 0.265 N weight, so it climbs.
        #       action -1.0  ->  0.0 N   rotors off, falls
        #       action  0.05 ->  0.265 N exactly cancels gravity, hovers
        #       action  1.0  ->  0.503 N full power
        #
        #   Note the mapping is monotonic: higher action always means more
        #   thrust, with no branch that could reverse it. That is what
        #   guarantees the policy cannot "mean" strong thrust while emitting a
        #   negative number — the option does not exist.

        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        #   ACTIONS 1, 2, 3 BECOME TURNING TORQUES about the three body axes.
        #   No remap here: torque genuinely goes both ways, since the drone can
        #   twist left or right. Only thrust needed the one-sided treatment.

    # -------------------------------------------------------------------------
    #  _apply_action — runs on EVERY physics tick
    # -------------------------------------------------------------------------
    #  WHAT IT DOES
    #     Hands the force and torque computed above to the physics engine.
    #
    #  WHY IT RUNS REPEATEDLY
    #     External forces in PhysX do not persist between ticks. Applied once,
    #     the drone would receive a single nudge and then coast. Reapplying
    #     each tick is what makes thrust continuous.
    #
    #  THE IMPORTANT THING TO NOTICE
    #     `body_ids=self._body_id` targets the drone's CENTRAL BODY. The four
    #     rotors are not simulated aerodynamically at all. They spin visually,
    #     but no blade forces or airflow are computed — the simulation jumps
    #     straight to the *result* of the rotor physics: one net force and
    #     three net torques.
    #
    #         REAL QUADCOPTER              THIS SIMULATION
    #         4 rotor speeds               1 force + 3 torques
    #             |                            |
    #         airflow, blade               applied directly
    #         aerodynamics                 at the body centre
    #             |
    #         net force + torque
    #
    #     WHY SIMPLIFY: aerodynamics are expensive and add little to a task
    #     about *where* to fly. The cost is realism — a policy trained this way
    #     needs a controller translating force-and-moment commands into rotor
    #     speeds before it could fly real hardware.
    #
    #  VERSION NOTE
    #     Recent Isaac Lab uses `permanent_wrench_composer.set_forces_and_torques`
    #     (shown below). Slightly older versions use:
    #         self._robot.set_external_force_and_torque(
    #             self._thrust, self._moment, body_ids=self._body_id)
    #     Both do the same thing. Check which your installed version uses
    #     before copying code between them.
    # -------------------------------------------------------------------------
    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    # -------------------------------------------------------------------------
    #  _get_observations — what the policy is allowed to know
    # -------------------------------------------------------------------------
    #  WHAT IT DOES
    #     Assembles the 12 numbers the network receives as input.
    #
    #  WHY THE CONTENTS MATTER MORE THAN ANY OTHER CHOICE IN THIS FILE
    #     The policy can only react to what appears here. Omit something it
    #     needs and no amount of training will help; include something it could
    #     never have in reality and the policy will depend on it and fail when
    #     deployed.
    #
    #  THE 12 NUMBERS
    #     [0:3]  own linear velocity   (body frame)
    #     [3:6]  own angular velocity  (body frame)
    #     [6:9]  gravity direction     (body frame) — "which way is down"
    #     [9:12] goal position relative to the drone (body frame)
    #
    #  THE BODY-FRAME IDEA, WHICH IS THE KEY TO THE WHOLE DESIGN
    #     "Body frame" means expressed from the drone's own point of view:
    #     x forward through its nose, y left, z up. So the goal arrives as
    #     "2 m ahead and slightly left of me" rather than "at world coordinates
    #     (14.2, -3.7, 1.5)".
    #     WHY THIS MATTERS: the first is learnable once and applies everywhere
    #     in the arena. The second would have to be memorised per location, and
    #     a drone that learned to hover at one world coordinate would be
    #     useless anywhere else.
    #     It is also why a camera fits so naturally later: a camera mounted on
    #     the drone already sees the world in exactly this frame.
    #
    #  WHAT IF IT WERE ABSENT
    #     The policy would receive nothing and could only act on noise.
    # -------------------------------------------------------------------------
    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        #   Converts the world-frame goal into the drone's body frame, given
        #   where the drone is (root_pos_w) and how it is oriented
        #   (root_quat_w, a quaternion). The `_b` suffix marks body frame.
        #   CHAPTER 2 CHANGES ONE ARGUMENT HERE: the static goal becomes the
        #   attacker's live position. That single substitution is the core of
        #   the pursuit project.

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,      # 3 — how fast am I moving
                self._robot.data.root_ang_vel_b,      # 3 — how fast am I spinning
                self._robot.data.projected_gravity_b, # 3 — which way is down
                desired_pos_b,                        # 3 — where is the target
            ],
            dim=-1,
        )
        #   The first nine are self-knowledge, which a real drone reads from its
        #   onboard inertial sensors. Only the last three concern the outside
        #   world — and in the pursuit project, only those get replaced by
        #   camera-derived values.

        observations = {"policy": obs}
        #   A dictionary because some algorithms feed the critic a different
        #   set. Here both use "policy".
        return observations

    # -------------------------------------------------------------------------
    #  _get_rewards — the score, used ONLY during training
    # -------------------------------------------------------------------------
    #  WHAT IT DOES
    #     Produces one number per environment saying how good the last step was.
    #
    #  WHO READS IT
    #     Only the training algorithm, which uses it to adjust network weights.
    #     Once training ends this method is never called again — play.py does
    #     not evaluate it. That is why a reward may legitimately use
    #     information the policy itself never receives.
    #
    #  WHAT IF IT WERE ABSENT
    #     Every action would score identically and PPO would have no basis for
    #     preferring one behaviour over another. The policy would never improve.
    # -------------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        #   Squared speeds. Squaring makes the penalty grow disproportionately
        #   with speed, so gentle drift is barely punished while violent motion
        #   is punished hard. It also removes sign — direction is irrelevant.

        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        #   Straight-line distance in metres, computed for all environments at
        #   once. `dim=1` collapses the three coordinate columns into one
        #   distance per environment.

        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        #   Converts distance into a 0-to-1 score where closer is better:
        #       0.0 m -> 1.00      0.8 m -> 0.24      3.0 m -> 0.02
        #   WHY tanh: it is bounded, so the reward cannot explode. The obvious
        #   alternative, 1/distance, goes to infinity as the drone approaches
        #   the goal, which destabilises PPO.
        #   THE 0.8 IS THE SENSITIVITY DIAL: smaller makes the reward rise
        #   steeply only very near the goal; larger spreads it over a wider
        #   region. Tuning this changes how precisely the drone learns to
        #   arrive.

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        #   Kept as a dictionary rather than a running total so each component
        #   can be logged separately — essential when diagnosing why learning
        #   stalled.
        #   `* self.step_dt` scales by the timestep, so reward magnitudes stay
        #   comparable if the control frequency changes.

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        #   Add the components into one number per environment. PPO needs a
        #   single scalar; the breakdown exists only for humans.

        for key, value in rewards.items():
            self._episode_sums[key] += value
        #   Accumulate for reporting at reset.

        return reward

    # -------------------------------------------------------------------------
    #  _get_dones — when does an attempt end
    # -------------------------------------------------------------------------
    #  WHAT IT DOES
    #     Returns two boolean tensors: which environments FAILED, and which
    #     simply RAN OUT OF TIME.
    #
    #  WHY THE TWO ARE SEPARATE — this is not bookkeeping pedantry
    #     Failure means the future genuinely ended: crashed, no more reward
    #     possible ever. Timeout means the attempt was cut off arbitrarily
    #     while the drone was doing fine. PPO estimates future reward, and it
    #     must treat these differently: after a crash, expected future reward
    #     is zero; after a timeout, it is whatever the drone would have kept
    #     earning. Conflating them teaches the policy that surviving to the
    #     time limit is a kind of death, and it learns to avoid it.
    #
    #  WHAT IF IT WERE ABSENT
    #     Attempts would never end. Crashed drones would lie on the floor
    #     generating useless experience forever, and the policy would never
    #     encounter a fresh start.
    # -------------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        #   episode_length_buf counts steps taken in the current attempt, per
        #   environment. The parent class maintains it.

        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        #   Column 2 is height. Below 0.1 m means it hit the ground; above
        #   2.0 m means it escaped upward. Both count as failure.
        #   WHY CAP THE CEILING: without it, a policy could discover that
        #   flying straight up avoids crashing, and drift away forever while
        #   collecting the small survival value of not having failed.

        return died, time_out

    # -------------------------------------------------------------------------
    #  _reset_idx — restart the environments that just ended
    # -------------------------------------------------------------------------
    #  WHAT IT DOES
    #     For the given environments only: reports their statistics, picks a
    #     new random goal, and returns the drone to its starting state.
    #
    #  THE CRITICAL DETAIL — it resets SOME environments, not all
    #     `env_ids` lists exactly which environments finished. Out of 4096,
    #     perhaps 37 crashed on this step while the rest are mid-flight. Only
    #     those 37 rows are touched. This is why every line indexes with
    #     [env_ids]. Writing the whole table would teleport 4000 healthy drones
    #     back to the start, destroying their in-progress attempts.
    #
    #  WHAT IF IT WERE ABSENT
    #     A crashed drone would stay crashed and never try again. Training
    #     would collapse to whatever happened in the first ten seconds.
    # -------------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        #   None means "reset everything" — used at startup.

        # --- reporting -------------------------------------------------------
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        #   How close did these attempts finish? One of the most informative
        #   single numbers for judging whether learning is working.

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
            #   Average each reward component over the finished attempts, send
            #   it to TensorBoard, then clear the counters for the next attempt.

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)
        #   The ratio of crashes to timeouts is the fastest read on training
        #   health. Early on, almost everything is a crash; as the policy
        #   improves, timeouts dominate because drones survive the full ten
        #   seconds.

        # --- the actual reset ------------------------------------------------
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        #   At startup, give every environment a RANDOM starting age instead of
        #   zero. Otherwise all 4096 would finish on the same step, causing a
        #   periodic stall and a jagged reward curve. Staggering them spreads
        #   the cost evenly.

        self._actions[env_ids] = 0.0

        # --- pick a new goal --------------------------------------------------
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        #   THIS IS WHERE THE GOAL IS ACTUALLY DEFINED — the __init__ line only
        #   reserved space.
        #   Read it as: pick random x and y within +/-2 m, shift them by this
        #   environment's origin so each parallel arena gets its goal in its own
        #   patch of world, then pick a random height between 0.5 and 1.5 m.
        #
        #   WHY RANDOM RATHER THAN FIXED: a fixed goal would let the policy
        #   memorise one flight path rather than learning to fly to wherever
        #   the target is. The same reasoning drives the pursuit project's
        #   randomised attacker trajectories in Chapter 2.

        # --- return the drone to its starting state ---------------------------
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        #   default_root_state holds position and orientation relative to the
        #   environment; adding env_origins converts it to world coordinates.
        #   FORGETTING THIS LINE is a classic error: every drone spawns at the
        #   same world point, all 4096 stacked inside each other.

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        #   Columns 0-6 are position and orientation (3 + a 4-number
        #   quaternion); 7-12 are linear and angular velocity.
        #   RESETTING VELOCITY MATTERS: without it, a drone that crashed at
        #   speed would respawn still carrying that speed and immediately fail
        #   again, producing a stream of useless experience.
        #
        #   NOTE FOR CHAPTER 2: these same write_root_* methods are how the
        #   pursuit project moves the scripted attacker — but there they are
        #   called every step rather than only at reset, which is what makes
        #   that drone follow a path instead of obeying physics.

    # -------------------------------------------------------------------------
    #  _set_debug_vis_impl and _debug_vis_callback — the goal markers
    # -------------------------------------------------------------------------
    #  WHAT THEY DO
    #     Create small cubes at each goal position and move them each frame.
    #     Viewing aid only; no effect on physics or learning.
    #
    #  WHY THEY ARE WORTH KNOWING ABOUT
    #     Seeing where the target is makes debugging enormously easier — a
    #     drone flying confidently toward the wrong place looks identical to a
    #     drone flying badly, until you can see the goal.
    # -------------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        #   Markers live under /Visuals/, outside /World/envs/, so they are not
        #   duplicated by clone_environments and carry no physics.


# =============================================================================
#  READING SUMMARY — the five lines that become the pursuit project
# =============================================================================
#
#  1. cfg.robot = CRAZYFLIE_CFG.replace(prim_path=...)
#        Chapter 2 adds a SECOND line like this for the attacker drone.
#
#  2. _setup_scene: self.scene.articulations["robot"] = self._robot
#        Chapter 2 registers the attacker the same way.
#
#  3. __init__: self._desired_pos_w = torch.zeros(...)
#        Chapter 2 stops using a stored goal and reads the attacker's live
#        position instead.
#
#  4. _get_observations: subtract_frame_transforms(..., self._desired_pos_w)
#        Chapter 3 replaces this with camera-style readings — where the target
#        appears in view and how large it looks.
#
#  5. _get_rewards: distance_to_goal
#        Chapter 3 replaces "be near the goal" with "close the distance to a
#        moving target, and capture it".
#
#  Everything else in this file — the timing, the cloning, the reset
#  mechanics, the force conversion — carries over essentially unchanged.
# =============================================================================
