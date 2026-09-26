# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab_assets import CRAZYFLIE_CFG 
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
# ▼▼▼ NEW! ▼▼▼
from isaaclab.utils.math import subtract_frame_transforms, quat_apply
# ▲▲▲ END OF INSERT ▲▲▲

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 5              # <<< INSERT THE TELLO NUMBER HERE !!! (using dummy for now)
                                # Decimation is how many physics steps pass between policy decisions
                                # Convertion of hertz in decimations: 
                                    # Physic engine updates / physic steps = game tick
                                    # Policy decision: every time the neural network ingests the inputs and produces the outputs
                                    # Hz/hertz: x times anything happens per sec
                                    # The physics engine updates policy 100 times per second (100 physics steps / sec)
                                    # Policy decision happens once every 5 physics steps (decimation = 5).
                                    # How many policy decision happen every second? = 100 steps / 5 decimations = 20 policy decisions per second = 20 hertz
    action_space = 4
    observation_space = 17      # The number of parameters the policy receives
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
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

    # scene
    # NEW: bumped env_spacing in the scene cfg to 2 * arena_radius (16.0) so neighboring envs' drones never visually overlap into each other's future camera views.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=16, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    # robot (defender) — blue paint so it's easy to tell apart from the attacker
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=CRAZYFLIE_CFG.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 1.0)),
        ),
    )

    # NEW: attacker — same drone, different prim path, spawned 4 m away at 1.5 m altitude
    attacker: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Attacker",
        init_state=ArticulationCfg.InitialStateCfg(pos=(4.0, 0.0, 1.5)),
    )

    thrust_to_weight = 1.9
    moment_scale = 0.01
    # NEW: pursuit geometry knobs (plain attributes, like the reward scales you know)
    capture_radius = 0.35        # meters — "caught" if closer than this
    arena_radius = 8.0           # meters — episode fails if defender strays this far
    attacker_speed = 0.6         # m/s along its path (we'll tune this in Ch. 3)

    # ▼▼▼ NEW — the stick-command model ▼▼▼
    max_speed = 2.0          # m/s — conservative; a Tello can do more
    max_yaw_rate = 1.5       # rad/s
    vel_gain = 3.0           # how hard the stand-in stabiliser corrects
    yaw_gain = 10.0           # <<< CHANGED TO 10.0. BEFORE = 0.05. Fixes ~10% of the turning error per physics tick
    # ▲▲▲ END OF INSERT ▲▲▲
    # ▼▼▼ NEW! — from project_notes.txt, converted to control steps ▼▼▼
    # Manually ads a delay to the input reading on the simulation so it mimics the expected behavior when running on hardware
    obs_delay_min = 2
    obs_delay_max = 5
    # ▲▲▲ END OF INSERT ▲▲▲

    # ▼▼▼ NEW! — camera model. MUST match the TiledCameraCfg you add ▼▼▼
    # in 5.2 AND the real drone's lens.
    cam_width = 640
    cam_height = 480                  # 4:3, matching the Tello's 960x720
    cam_focal_mm = 12.0               # <<< ASSUMED VALUES: VERIFY AND REPLACE WITH REAL HARDWARE MEASUREMENTS!!!
    cam_aperture_mm = 20.955          # PinholeCameraCfg default horizontal aperture
    attacker_span_m = 0.13            # <<< ASSUMED VALUES: VERIFY AND REPLACE WITH REAL HARDWARE MEASUREMENTS!!!
    # ▲▲▲ END OF INSERT ▲▲▲

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    # distance_to_goal_reward_scale = 15.0          <<< commented out because it was only used for the initial test task (hoverring over a fixed point) In the pursuit task, the target is the moving attacker, not a fixed point

    # ---------------------------- REWARD FUNCTION PARAMETERS -----------------------------------------------------------
    # ▼▼▼ NEW! — the pursuit reward dials ▼▼▼
    closing_reward_scale = 2.0      # per m/s of speed TOWARD the target
    proximity_reward_scale = 1.5    # smooth "warmth" signal as distance shrinks
    capture_bonus = 200.0           # paid once, at capture
    crash_penalty = -50.0           # hit the floor / left the arena
    action_rate_penalty = -0.02     # sim-to-real: penalise jerky command changes
    # ▲▲▲ END OF INSERT ▲▲▲
    # -------------------------------------------------------------------------------------------------------------------

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        # # This line creates that table that represents the position of the defender drone target in space, in coordinates
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # ▼▼▼ NEW — attacker trajectory state, one row per environment ▼▼▼
        self._atk_phase = torch.zeros(self.num_envs, device=self.device)   # start angle
        self._atk_dir = torch.ones(self.num_envs, device=self.device)      # +1 or -1 (CW/CCW)
        self._atk_t = torch.zeros(self.num_envs, device=self.device)       # per-env clock
        self._atk_radius = 3.0                                             # metres

        # ▼▼▼ NEW! — delayed-reading machinery ▼▼▼
        # Creates a buffer to stores the policy readin from 2-5 steps ago (obs_delay)
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
        self._captured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # ▲▲▲ END OF INSERT ▲▲▲

        # Logging
        # self._episode_sums = {
        #     key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        #     for key in [
        #         "lin_vel",
        #         "ang_vel",
        #         "distance_to_goal",
        #     ]
        # }

        # Logging                                                                  
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "closing", "proximity", "capture", "crash", "action_rate", "ang_vel",   # ◄── CHANGED: was "lin_vel", "ang_vel", "distance_to_goal"
            ]
        }


        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # ▼▼▼ NEW — resistance to turning (yaw inertia), used to scale the yaw twist ▼▼▼
        inertias = self._robot.root_physx_view.get_inertias()
        self._robot_izz = inertias[0, self._body_id[0], 8].item()
        print(f"[3.1] defender yaw inertia = {self._robot_izz:.2e} kg*m^2")
        # ▲▲▲ END OF INSERT ▲▲▲

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    # Builds the world once, before the simulation starts. It creates the drone from the recipe, 
    # adds a floor and a light, and then duplicates the whole arrangement thousands of times
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._attacker = Articulation(self.cfg.attacker)               # NEW: register the attacker on the scene
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["attacker"] = self._attacker          # NEW: register the attacker on the scene

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------POLICY IN ACTION---------------------------------------------------------
    #   Translates the neural network's output/commands into physical forces.
    #   It turns the policy's 4 numbers into forces, which _apply_action then applies.
    def _pre_physics_step(self, actions: torch.Tensor):

        # ▼▼▼ DELETE the hover task's force conversion — these three lines: ▼▼▼
        #   self._actions = actions.clone().clamp(-1.0, 1.0)
        #   self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        #   self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        # ▲▲▲ and REPLACE the whole method body with the code below ▲▲▲

        # ▼▼▼ NEW ▼▼▼
        self._actions = actions.clone().clamp(-1.0, 1.0)

        desired_vel_b = self._actions[:, :3] * self.cfg.max_speed        # m/s
        desired_yaw_rate = self._actions[:, 3] * self.cfg.max_yaw_rate   # rad/s

        # stands in for the Tello's own stabiliser
        vel_error = desired_vel_b - self._robot.data.root_lin_vel_b
        force_b = self.cfg.vel_gain * vel_error * self._robot_mass
        force_b -= self._robot_weight * self._robot.data.projected_gravity_b   # hold altitude ◄── CHANGED: added * self._robot.data.projected_gravity_b. "up" stays up even when tilted

        self._thrust[:, 0, :] = force_b                    # ◄── CHANGED: before: self._thrust[:, 0, :] = quat_apply(self._robot.data.root_quat_w, force_b). already in the drone's own frame
        yaw_error = desired_yaw_rate - self._robot.data.root_ang_vel_b[:, 2]
        self._moment[:, 0, 2] = self.cfg.yaw_gain * yaw_error * self._robot_izz   # ◄── CHANGED: added * self._robot_izz
        # ▲▲▲ END OF INSERT ▲▲▲

    # Hands those forces to the physics engine, repeatedly. It runs on every physics tick, 
    # which happens more often than the network decides — so one decision gets applied several times.
    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        self._move_attacker()

    # --------------------------------------------------------------------------------------------------------------------------------

    # Only moves the attacker. Since the attacker movement doesn't rely on the RL training or policy (this function just directly teleports it to its next position on every tick)
    # ▼▼▼ NEW  ▼▼▼
    def _move_attacker(self):
        """Kinematic attacker: circle + vertical bob, written to sim each physics tick."""

        # Advance the stopwatch by one physics tick (1/100 s). This is the only
        # line that makes anything move — everything below just reads the clock.
        self._atk_t += self.physics_dt

        # Convert "metres per second along the path" into "radians of angle per
        # second". Dividing by the radius is what makes attacker_speed mean m/s:
        # on a bigger circle the same angle covers more ground, so it needs less angle.
        # w = angular velocity = radians per second
        w = self.cfg.attacker_speed / self._atk_radius            # m/s → rad/s

        # Where the clock hand points RIGHT NOW = where it started (phase)
        # + how far it has swept since (w × t), run forwards or backwards (direction).
        # One number per environment, because phase and direction differ per env.
        # theta = angle on a circle = phase (starting angle) + direction (-1 or 1) * radians per second * time
        theta = self._atk_phase + self._atk_dir * w * self._atk_t

        # An empty table: one row per environment, three columns (x, y, z).
        # Filled in over the next three lines.
        pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Turn the angle into a position. cos gives the offset along x for a
        # circle of radius 1; multiplying by 3 stretches it to a 3-metre circle.
        pos[:, 0] = self._atk_radius * torch.cos(theta)

        # Same for the y offset, using sin. Together, cos and sin place the point
        # exactly on the circle, whatever angle theta happens to be.
        pos[:, 1] = self._atk_radius * torch.sin(theta)

        # Altitude. This sin is NOT a circle coordinate — it is being used as a
        # wave over time: hover at 1.5 m, rise and fall 0.4 m either side.
        # 0.7 sets how fast it bobs (radians per second); adding phase means each
        # environment's bob is out of step with the others, like the circle is.
        pos[:, 2] = 1.5 + 0.4 * torch.sin(0.7 * self._atk_t + self._atk_phase)

        # Everything above was measured from the environment's own centre. Adding
        # env_origins shifts each row to where that arena actually sits in the world.
        pos += self.scene.env_origins                             # local → world

        # Read the attacker's current pose (position + which way it is facing),
        # so we can change only the part we care about.
        pose = self._attacker.data.root_pose_w.clone()

        # Overwrite the position columns with the new point. Columns 3 onward hold
        # the facing direction and are left untouched — the attacker slides around
        # the circle without turning.
        pose[:, :3] = pos                                         # keep orientation as-is

        # Hand the new pose to the physics engine. This is a teleport, not a push:
        # PhysX places the body exactly here rather than working out how it got there.
        self._attacker.write_root_pose_to_sim(pose)

        # On the very first tick there is no previous position to compare against,
        # so seed it with the current one. This makes the first velocity come out
        # as zero instead of a meaningless huge jump.
        if not hasattr(self, "_atk_prev_pos"):
            self._atk_prev_pos = pos.clone()

        # Speed = distance moved ÷ time taken. Because the attacker is teleported
        # rather than pushed, PhysX has no velocity for it — so we work it out
        # ourselves from where it was one tick ago.
        self._atk_vel = (pos - self._atk_prev_pos) / self.physics_dt

        # Remember today's position so the next tick can do the same subtraction.
        self._atk_prev_pos = pos.clone()
    # ▲▲▲ END OF INSERT ▲▲▲

    # ▼▼▼ New!: pushes fresh readings in and takes delayed ones out ▼▼▼
    def _delayed_readings(self, fresh):          # fresh: (num_envs, 7)
        """Shift the history one step, store the newest, return each env's delayed row."""
        self._reading_history = torch.roll(self._reading_history, shifts=1, dims=1)
        self._reading_history[:, 0] = fresh
        idx = self._delay_steps.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 7)
        return self._reading_history.gather(1, idx).squeeze(1)
    # ▲▲▲ END OF INSERT ▲▲▲

    # ▼▼▼ NEW!: Converts position coordinates in a physical space into image positions on a camera frame ▼▼▼
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

        # ▼▼▼ New! - integrating all 17 model inputs into one list ▼▼▼
        # measure the true distance again for the camera readings below — _reset_idx  
        # has just moved the drones in the envs that restarted. (_get_dones measures it separately each step.)
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

    # --------------------------------REWARD CALCULATION------------------------------------------------------------
    # Keeps the score of the training rewards
    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        # ▼▼▼ DELETE : this was the reward structure of the first hover task - to hover over a fixed point. We are replacing this with the new pursuit task ▼▼▼
        # distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        # distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        # rewards = {
        #     "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
        #     "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
        #     "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        # }
        # reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # ▼▼▼ NEW! - Pursuit task reward structure ▼▼▼
        # ── BLOCK A: direction from defender to attacker ─────────────────────
        # The closing-speed term needs to know which way "toward the attacker" is.
        # Dividing the offset vector by its length turns it into a pure direction
        # (length 1). clamp() stops a division by zero if the drones ever coincide.
        # self._dist is updated at the top of _get_dones (Step 3), which Isaac Lab      
        # runs just before this method on every step.  
        to_target = self._attacker.data.root_pos_w - self._robot.data.root_pos_w
        dir_to_target = to_target / self._dist.unsqueeze(1).clamp(min=1e-6)

        # ── BLOCK B: the three PAYMENTS — what the defender is rewarded for ──
        # 1) CLOSING SPEED: the part of the defender's velocity that points at
        #    the attacker. +1.0 = approaching at 1 m/s, negative = moving away.
        #    Paid every step, so even an untrained policy learns which way is better.
        closing = (self._robot.data.root_lin_vel_w * dir_to_target).sum(dim=1)

        # 2) PROXIMITY: 1 - tanh(dist/4) — a smooth 0..1 value that rises as you approach.
        proximity = 1.0 - torch.tanh(self._dist / 4.0)

        # 3) CAPTURE: the one-off bonus
        captured = self._dist < self.cfg.capture_radius

        # ── BLOCK C: the three PENALTIES — what the defender is charged for ──
        # 4) SMOOTHNESS: how much the 4 commands changed since the last step.
        #    Large swings cost nothing in sim but make the real Tello shake.
        action_rate = torch.sum(torch.square(self._actions - self._prev_actions), dim=1)

        # 5) STABILITY: penalise spinning/wobbling (same formula as the hover task)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        # 6) CRASH: same two conditions _get_dones uses for failure
        crashed = (self._robot.data.root_pos_w[:, 2] < 0.1) | (self._dist > self.cfg.arena_radius)

        # ── BLOCK D: weight each term and store it under a name ──────────────
        # Each raw value above is multiplied by its dial from QuadcopterEnvCfg
        # (Step 1). step_dt keeps totals comparable if the control rate changes.
        # Names matter: they must match the keys of self._episode_sums in
        # __init__, and each becomes a TensorBoard curve Episode_Reward/<name>.
        rewards = {
            "closing":     self.cfg.closing_reward_scale   * closing          * self.step_dt,
            "proximity":   self.cfg.proximity_reward_scale * proximity        * self.step_dt,
            "capture":     self.cfg.capture_bonus          * captured.float() * self.step_dt,
            "action_rate": self.cfg.action_rate_penalty    * action_rate      * self.step_dt,
            "ang_vel":     self.cfg.ang_vel_reward_scale   * ang_vel          * self.step_dt,
            "crash":       self.cfg.crash_penalty          * crashed.float()  * self.step_dt,
        }
        # the single number PPO sees: the sum of all six terms, one value per env
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # ── BLOCK E: remember this step's commands ───────────────────────────
        # Next step's smoothness penalty compares against these.
        # Must come AFTER action_rate was computed above.
        self._prev_actions = self._actions.clone()

        # ── BLOCK F: logging (kept from the hover task) ──────────────────────
        # Adds each term to a running total per env. _reset_idx writes the totals
        # to TensorBoard when an episode ends, which lets you see WHICH term is
        # driving the reward — e.g. proximity high + capture 0 = orbiting exploit.
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
        # ▲▲▲ END OF INSERT ▲▲▲

    # Ends Episodes
    # is the method that decides which episodes have terminated (drone failed) or truncated (ran out of time)
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        # ▼▼▼ DELETE the hover task's body — it looked like this: ▼▼▼
        #   time_out = self.episode_length_buf >= self.max_episode_length - 1
        #   died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1,
        #                           self._robot.data.root_pos_w[:, 2] > 2.0)
        #   return died, time_out
        # ▲▲▲ and REPLACE it with the code below ▲▲▲

        # ▼▼▼ NEW! adding capture (success to the ending conditions (as terminated)) ▼▼▼
        # measure the distance AFTER this step's physics — _get_rewards reads it next    
        self._dist = torch.linalg.norm(                                                  
            self._attacker.data.root_pos_w - self._robot.data.root_pos_w, dim=1          
        )                                                                                
        captured = self._dist < self.cfg.capture_radius                       # success
        self._captured = captured                                             # makes captured a global variable so it can be used by _reset_idx to tell if an episode terminaded from a crash or from a capture
        crashed = self._robot.data.root_pos_w[:, 2] < 0.1                     # floor
        escaped = self._dist > self.cfg.arena_radius                          # lost it: attacker got away

        # 1st return value: any outcome that ends the attempt, win or loss
        terminated = crashed | escaped | captured        # all three END the episode now
        # 2nd return value: ran out of time
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out
        # ▲▲▲ END OF INSERT ▲▲▲

    #  restarts only the environments whose episode just ended.
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        # final_distance_to_goal = torch.linalg.norm(
        #     self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        # ).mean()

        # ▼▼▼ NEW! — replaces final_distance_to_goal (leftover from the hover task) ▼▼▼
        # distance between the drones when the episode ended; measured here, before
        # the reset below moves them. Works on the very first reset too.
        final_dist = torch.linalg.norm(
            self._attacker.data.root_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        )
        # ▲▲▲ END OF CHANGE ▲▲▲
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()

        # ▼▼▼ Replace
        # extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()

        # ▼▼▼ NEW — count captures separately from deaths ▼▼▼
        cap = self._captured[env_ids]                                                                        # ◄── NEW
        extras["Episode_Termination/captured"] = torch.count_nonzero(self.reset_terminated[env_ids] & cap)   # ◄── NEW
        extras["Episode_Termination/died"]     = torch.count_nonzero(self.reset_terminated[env_ids] & ~cap)  # ◄── NEW
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids])          # ◄── NEW

        # ▼▼▼ NEW 3.3 Step 2 — pursuit-specific metrics ▼▼▼
        extras["Metrics/final_distance"]   = final_dist.mean()
        extras["Metrics/capture_rate"]     = cap.float().mean()
        extras["Metrics/visible_fraction"] = self._prev_asz[env_ids].gt(0).float().mean()
        # ▲▲▲ END OF INSERT ▲▲▲

        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        
        # Reset robot state: Defender
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # NEW: Reset robot state: Attacker
        a_state = self._attacker.data.default_root_state[env_ids].clone()
        a_state[:, :3] += self.scene.env_origins[env_ids]
        self._attacker.write_root_pose_to_sim(a_state[:, :7], env_ids)
        self._attacker.write_root_velocity_to_sim(a_state[:, 7:], env_ids)

        # ▼▼▼ NEW! —  re-randomise the chase on reset ▼▼▼
        n = len(env_ids)
        self._atk_phase[env_ids] = torch.rand(n, device=self.device) * 2 * math.pi
        self._atk_dir[env_ids] = torch.where(
            torch.rand(n, device=self.device) > 0.5, 1.0, -1.0
        )
        self._atk_t[env_ids] = 0.0
        # ▲▲▲ END OF INSERT ▲▲▲

        # ▼▼▼ NEW! clear the history and draw a new lag, picks a new random delay (2, 3, 4 or 5 steps) for that environment: ▼▼▼
        self._reading_history[env_ids] = 0.0
        self._delay_steps[env_ids] = torch.randint(
            self.cfg.obs_delay_min, self.cfg.obs_delay_max + 1,
            (len(env_ids),), device=self.device
        )
        self._prev_bx[env_ids] = 0.0
        self._prev_by[env_ids] = 0.0
        self._prev_asz[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        # ▲▲▲ END OF INSERT ▲▲▲

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
