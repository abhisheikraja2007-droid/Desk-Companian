from __future__ import annotations
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, List, Optional, Tuple
import collections
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from stage3_ai.mistake_replay import MistakeReplayBuffer

class StudyDeskArenaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        xml_path: str = "stage3_ai/assets/single_desk_arena.xml",
        render_mode: Optional[str] = None,
        mistake_buffer: Optional[MistakeReplayBuffer] = None,
        curriculum_phase: int = 1,
        unseen_test_mode: bool = False
    ) -> None:
        super().__init__()
        self.xml_path = os.path.abspath(xml_path)
        self.render_mode = render_mode
        self.mistake_buffer = mistake_buffer or MistakeReplayBuffer()
        self.curriculum_phase = curriculum_phase
        self.unseen_test_mode = unseen_test_mode

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self._configure_physics()
        self.data = mujoco.MjData(self.model)

        # Cache baseline physical parameters
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.nominal_base_mass = float(self.model.body_mass[self.base_body_id]) if self.base_body_id != -1 else 0.376
        self.desk_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "desk_top")
        self.nominal_friction = float(self.model.geom_friction[self.desk_geom_id, 0]) if self.desk_geom_id != -1 else 0.45

        self.mug_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "coffee_mug")
        self.laptop_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "laptop")

        self.renderer: Optional[mujoco.Renderer] = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=720, width=1280)

        # Action: 10 actuators
        # [wheel_l, wheel_r, neck_yaw, neck_pitch, left_shoulder, right_shoulder, left_pincer, right_pincer, left_ear, right_ear]
        self.action_space = spaces.Box(
            low=np.array([-25.0, -25.0, -1.309, -0.5236, -0.6981, -0.6981, 0.0, 0.0, -1.1345, -1.1345], dtype=np.float32),
            high=np.array([25.0, 25.0, 1.309, 0.6981, 2.3562, 2.3562, 0.9599, 0.9599, 0.7854, 0.7854], dtype=np.float32),
            dtype=np.float32
        )

        # Observation: 36 dims
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = 400

        self.mug_center = np.array([-0.35, 0.15], dtype=np.float32)
        self.dock_center = np.array([-0.48, -0.20], dtype=np.float32)
        self.prev_dist_to_dock = 0.50
        self.prev_action = np.zeros(10, dtype=np.float32)
        self.action_queue: collections.deque[np.ndarray] = collections.deque(maxlen=2)

    def _configure_physics(self) -> None:
        self.model.dof_armature[:] = 0.005
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

        for i in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name in ["desk", "coffee_mug", "water_bottle", "laptop", "stylus_pen", "world"]:
                self.model.geom_contype[i] = 2
                self.model.geom_conaffinity[i] = 1
            else:
                self.model.geom_contype[i] = 1
                self.model.geom_conaffinity[i] = 2

        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] *= 0.02
            self.model.actuator_biasprm[i, 1] *= 0.02
            self.model.actuator_biasprm[i, 2] *= 0.02

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Domain Randomization (Layout, Friction, Base Mass)
        if self.unseen_test_mode:
            # Extreme unseen parameters for generalization audit
            fric = float(self.np_random.uniform(0.18, 0.98))
            mass_scale = float(self.np_random.uniform(0.80, 1.20))
            mug_x = float(self.np_random.uniform(-0.38, -0.22))
            mug_y = float(self.np_random.uniform(0.06, 0.22))
            dock_x = float(self.np_random.uniform(-0.52, -0.40))
            dock_y = float(self.np_random.uniform(-0.25, -0.15))
        else:
            fric = float(self.np_random.uniform(0.25, 0.85))
            mass_scale = float(self.np_random.uniform(0.88, 1.12))
            mug_x = float(self.np_random.uniform(-0.36, -0.28))
            mug_y = float(self.np_random.uniform(0.10, 0.18))
            dock_x = float(self.np_random.uniform(-0.50, -0.44))
            dock_y = float(self.np_random.uniform(-0.22, -0.18))

        if self.desk_geom_id != -1:
            self.model.geom_friction[self.desk_geom_id, 0] = fric
        if self.base_body_id != -1:
            self.model.body_mass[self.base_body_id] = self.nominal_base_mass * mass_scale

        self.mug_center = np.array([mug_x, mug_y], dtype=np.float32)
        if self.mug_body_id != -1:
            self.model.body_pos[self.mug_body_id, 0] = mug_x
            self.model.body_pos[self.mug_body_id, 1] = mug_y

        self.dock_center = np.array([dock_x, dock_y], dtype=np.float32)

        # Randomize Stylus Position on desk
        stylus_x = float(self.np_random.uniform(-0.06, 0.08))
        stylus_y = float(self.np_random.uniform(0.02, 0.08))
        self.data.qpos[0] = stylus_x
        self.data.qpos[1] = stylus_y
        self.data.qpos[2] = 0.006

        # Safe Clean Robot Spawn coordinates:
        # > 0.22 m clearance from coffee mug, > 0.15 m margin from desk edges
        init_x = float(self.np_random.uniform(-0.08, 0.18))
        init_y = float(self.np_random.uniform(-0.16, -0.06))
        init_yaw = float(self.np_random.uniform(-0.35, 0.35))

        # Root joint of robot is Joint 1 (qpos idx 7..13)
        self.data.qpos[7] = init_x
        self.data.qpos[8] = init_y
        self.data.qpos[9] = 0.019  # Calibrated tire resting contact height
        cy = math.cos(init_yaw * 0.5)
        sy = math.sin(init_yaw * 0.5)
        self.data.qpos[10] = cy
        self.data.qpos[11] = 0.0
        self.data.qpos[12] = 0.0
        self.data.qpos[13] = sy

        # Initialize arm keyframe postures
        self.data.qpos[16] = 0.05
        self.data.qpos[18] = 0.05

        mujoco.mj_forward(self.model, self.data)

        # 5 Settle steps so TCRT5000 rangefinders & MPU6050 accelerometers stabilize
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        # Clear velocity artifacts from settling
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        rx, ry = float(self.data.qpos[7]), float(self.data.qpos[8])
        self.prev_dist_to_dock = math.sqrt((rx - self.dock_center[0])**2 + (ry - self.dock_center[1])**2)
        self.prev_action = np.zeros(10, dtype=np.float32)
        self.action_queue.clear()
        self.action_queue.append(np.zeros(10, dtype=np.float32))

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        raw_act = np.clip(action, self.action_space.low, self.action_space.high)

        # 10-30 ms Control Delay Buffer
        self.action_queue.append(raw_act)
        delayed_act = self.action_queue.popleft()
        self.data.ctrl[:] = delayed_act

        # 3% chance of random external desk-bump impulse
        if self.np_random.random() < 0.03:
            bump_force = self.np_random.normal(0, 0.04, size=2)
            self.data.qvel[6] += bump_force[0]
            self.data.qvel[7] += bump_force[1]

        # Step 10 sub-steps (0.002s * 10 = 0.020s = 50 Hz control loop)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, terminated, truncated, info = self._compute_reward_and_status(obs, delayed_act)

        # Action-rate smoothness penalty: -0.05 * ||a_t - a_{t-1}||^2
        action_diff = np.sum(np.square(delayed_act - self.prev_action))
        reward -= 0.05 * float(action_diff)
        self.prev_action = delayed_act.copy()

        if self.current_step >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # Cliff rangefinder readings (4 sensors)
        cliff_fl = float(self.data.sensordata[0])
        cliff_fr = float(self.data.sensordata[1])
        cliff_rl = float(self.data.sensordata[2])
        cliff_rr = float(self.data.sensordata[3])

        # 5% Transient IR sensor dropout simulation
        if self.np_random.random() < 0.05:
            cliff_fl = 0.021

        # IMU Accelerometer + Gyro with realistic Gaussian sensor noise
        imu_accel = self.data.sensordata[4:7] + self.np_random.normal(0, 0.04, size=3)
        imu_gyro = self.data.sensordata[7:10] + self.np_random.normal(0, 0.02, size=3)

        # Wheel encoders with Gaussian noise
        enc_l_vel = float(self.data.sensordata[11]) + float(self.np_random.normal(0, 0.02))
        enc_r_vel = float(self.data.sensordata[13]) + float(self.np_random.normal(0, 0.02))

        # Joint positions & velocities (8 servos)
        joint_pos = self.data.qpos[14:22]
        joint_vel = self.data.qvel[12:20]

        # Robot XY and heading from actual qpos
        rx = float(self.data.qpos[7])
        ry = float(self.data.qpos[8])
        qw = float(self.data.qpos[10])
        qz = float(self.data.qpos[13])
        yaw = 2.0 * math.atan2(qz, qw)
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)

        # Target relative vectors
        dock_vec = np.array([self.dock_center[0] - rx, self.dock_center[1] - ry], dtype=np.float32)
        stylus_x = float(self.data.qpos[0])
        stylus_y = float(self.data.qpos[1])
        stylus_vec = np.array([stylus_x - rx, stylus_y - ry], dtype=np.float32)

        obs = np.concatenate([
            np.array([cliff_fl, cliff_fr, cliff_rl, cliff_rr], dtype=np.float32),
            imu_accel.astype(np.float32),
            imu_gyro.astype(np.float32),
            np.array([enc_l_vel, enc_r_vel], dtype=np.float32),
            joint_pos.astype(np.float32),
            joint_vel.astype(np.float32),
            np.array([rx, ry, sin_yaw, cos_yaw], dtype=np.float32),
            dock_vec,
            stylus_vec
        ]).astype(np.float32)

        return obs

    def _compute_reward_and_status(self, obs: np.ndarray, action: np.ndarray) -> Tuple[float, bool, bool, Dict[str, Any]]:
        # Actual Robot physical coordinates directly from self.data.qpos
        rx = float(self.data.qpos[7])
        ry = float(self.data.qpos[8])
        rz = float(self.data.qpos[9])
        cliff_readings = obs[0:4]

        dist_to_mug = math.sqrt((rx - self.mug_center[0])**2 + (ry - self.mug_center[1])**2)
        dist_to_dock = math.sqrt((rx - self.dock_center[0])**2 + (ry - self.dock_center[1])**2)
        stylus_x = float(self.data.qpos[0])
        stylus_y = float(self.data.qpos[1])
        dist_to_stylus = math.sqrt((rx - stylus_x)**2 + (ry - stylus_y)**2)

        reward = 0.0
        terminated = False
        info = {
            "dist_mug": dist_to_mug,
            "dist_dock": dist_to_dock,
            "dist_stylus": dist_to_stylus,
            "mistake": None
        }

        # 1. Pre-cliff edge detection & emergency braking (<2 ms reflex)
        # Desk bounds are X: [-0.60, 0.60], Y: [-0.30, 0.30]
        # Drop-off occurs if chassis leaves desk boundaries or height drops below table surface
        off_edge_pos = abs(rx) > 0.58 or abs(ry) > 0.28 or rz < -0.015
        cliff_ir_drop = any(c > 0.045 for c in cliff_readings) and (abs(rx) > 0.54 or abs(ry) > 0.25)
        is_cliff_triggered = off_edge_pos or cliff_ir_drop

        if is_cliff_triggered:
            reward -= 100.0
            terminated = True
            info["mistake"] = "CLIFF_TRIGGER"
            self.mistake_buffer.record_mistake("CLIFF_TRIGGER", obs, action, penalty=100.0, priority=2.0)
            return reward, terminated, False, info

        # Tilt / Roll-over failure guard (> 45 deg tilt)
        qw = float(self.data.qpos[10])
        qz = float(self.data.qpos[13])
        qx = float(self.data.qpos[11])
        qy = float(self.data.qpos[12])
        upright = 1.0 - 2.0 * (qx * qx + qy * qy)
        if upright < 0.65 or rz < 0.008:
            reward -= 80.0
            terminated = True
            info["mistake"] = "TILT_OVERTURN"
            return reward, terminated, False, info

        # Continuous upright-orientation reward and roll/pitch angular velocity damping
        reward += 0.5 * float(upright)
        wx = float(self.data.qvel[9])
        wy = float(self.data.qvel[10])
        reward -= 0.2 * (wx * wx + wy * wy)

        # Lateral wheel-turn smoothing
        wheel_diff = abs(float(action[0]) - float(action[1]))
        reward -= 0.01 * wheel_diff

        # 2. Semantic Hazard Keep-Out: 14 cm buffer around Hot Coffee Mug
        if dist_to_mug < 0.14:
            penetration = (0.14 - dist_to_mug)
            penalty = 25.0 + 80.0 * penetration
            reward -= penalty
            info["mistake"] = "MUG_PROXIMITY_VIOLATION"
            self.mistake_buffer.record_mistake("MUG_PROXIMITY_VIOLATION", obs, action, penalty=penalty, priority=1.5)
        else:
            reward += 0.08  # Safe distance bonus

        # 3. Surface Friction Traction (Wood vs Cloth vs Paper)
        # Paper sheet is at X in [0.115, 0.325], Y in [-0.048, 0.248]
        on_paper = (0.115 <= rx <= 0.325) and (-0.048 <= ry <= 0.248)
        wheel_slip = abs(action[0] - action[1]) > 14.0
        if on_paper and wheel_slip:
            reward -= 4.0
            info["mistake"] = "WHEEL_SKID_PAPER"
            self.mistake_buffer.record_mistake("WHEEL_SKID_PAPER", obs, action, penalty=4.0, priority=1.0)

        # 4. Dense Anti-Reward-Hacking Progress Reward (+2.0 * delta_distance_to_goal)
        delta_dock = self.prev_dist_to_dock - dist_to_dock
        reward += 2.0 * delta_dock
        self.prev_dist_to_dock = dist_to_dock

        # Gentle Idle-Stagnation Penalty (robot must not freeze still to evade learning)
        if abs(action[0]) < 0.5 and abs(action[1]) < 0.5 and dist_to_dock > 0.10:
            reward -= 0.02

        # Docking Target Arrival Reward
        if dist_to_dock < 0.08:
            reward += 15.0
            if abs(action[0]) < 1.0 and abs(action[1]) < 1.0:
                reward += 25.0  # Decelerated settled into sleep nest

        # 5. Stylus Fetch Guidance
        if dist_to_stylus < 0.07:
            pincer_closure = (action[6] + action[7]) * 0.5
            reward += 10.0 * pincer_closure

        return reward, terminated, False, info

    def render(self) -> Optional[np.ndarray]:
        if self.renderer is None:
            return None
        self.renderer.update_scene(self.data, camera="ov5640_cam")
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.mistake_buffer.close()
