from __future__ import annotations
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, List, Optional, Tuple
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
        curriculum_phase: int = 1
    ) -> None:
        super().__init__()
        self.xml_path = os.path.abspath(xml_path)
        self.render_mode = render_mode
        self.mistake_buffer = mistake_buffer or MistakeReplayBuffer()
        self.curriculum_phase = curriculum_phase

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        # Apply stable numerical properties and collision bitmasks
        self._configure_physics()
        self.data = mujoco.MjData(self.model)

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
        # 4 cliff rangefinders, 3 imu accel, 3 imu gyro, 2 wheel encoders, 8 joint pos, 8 joint vel,
        # 2 robot xy, 2 robot yaw (sin, cos), 2 target vector to dock, 2 target vector to stylus
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = 400
        self.mug_center = np.array([-0.35, 0.15], dtype=np.float32)
        self.dock_center = np.array([-0.48, -0.20], dtype=np.float32)
        self.stylus_init_pos = np.array([0.00, 0.06], dtype=np.float32)

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

        # Randomize initial robot pose on safe area of the desk
        # Stylus at (0, 0.06), robot spawned around (0, -0.15)
        init_x = float(self.np_random.uniform(-0.15, 0.15))
        init_y = float(self.np_random.uniform(-0.18, -0.05))
        init_yaw = float(self.np_random.uniform(-0.5, 0.5))

        # Root joint is Joint 1 (dof 6), qpos idx 7 (since Joint 0 stylus has 7 qpos)
        self.data.qpos[7] = init_x
        self.data.qpos[8] = init_y
        self.data.qpos[9] = 0.022
        cy = math.cos(init_yaw * 0.5)
        sy = math.sin(init_yaw * 0.5)
        self.data.qpos[10] = cy
        self.data.qpos[11] = 0.0
        self.data.qpos[12] = 0.0
        self.data.qpos[13] = sy

        # S-curve warm-start keyframe posture for arms and ears
        self.data.qpos[16] = 0.05  # left shoulder
        self.data.qpos[18] = 0.05  # right shoulder

        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        act = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = act

        # Step 10 sub-steps (0.002s * 10 = 0.020s = 50 Hz control loop)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, terminated, truncated, info = self._compute_reward_and_status(obs, act)

        if self.current_step >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # Cliff rangefinder readings (4 sensors)
        # In sensor array: indices 0..3 are rangefinders
        cliff_fl = float(self.data.sensordata[0])
        cliff_fr = float(self.data.sensordata[1])
        cliff_rl = float(self.data.sensordata[2])
        cliff_rr = float(self.data.sensordata[3])

        # IMU Accelerometer + Gyro
        imu_accel = self.data.sensordata[4:7]
        imu_gyro = self.data.sensordata[7:10]

        # Wheel encoders
        enc_l_vel = float(self.data.sensordata[11])
        enc_r_vel = float(self.data.sensordata[13])

        # Joint positions & velocities (8 servos)
        # qpos 14..21 are the 8 revolute joints
        joint_pos = self.data.qpos[14:22]
        joint_vel = self.data.qvel[12:20]

        # Robot XY and heading
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
            imu_accel,
            imu_gyro,
            np.array([enc_l_vel, enc_r_vel], dtype=np.float32),
            joint_pos,
            joint_vel,
            np.array([rx, ry, sin_yaw, cos_yaw], dtype=np.float32),
            dock_vec,
            stylus_vec
        ]).astype(np.float32)

        return obs

    def _compute_reward_and_status(self, obs: np.ndarray, action: np.ndarray) -> Tuple[float, bool, bool, Dict[str, Any]]:
        rx, ry = obs[26], obs[27]
        cliff_readings = obs[0:4]

        dist_to_mug = math.sqrt((rx - self.mug_center[0])**2 + (ry - self.mug_center[1])**2)
        dist_to_dock = math.sqrt((rx - self.dock_center[0])**2 + (ry - self.dock_center[1])**2)
        dist_to_stylus = math.sqrt((rx - float(self.data.qpos[0]))**2 + (ry - float(self.data.qpos[1]))**2)

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
        is_cliff_triggered = any(c > 0.038 for c in cliff_readings) or abs(rx) > 0.58 or abs(ry) > 0.28
        if is_cliff_triggered:
            reward -= 100.0
            terminated = True
            info["mistake"] = "CLIFF_TRIGGER"
            self.mistake_buffer.record_mistake("CLIFF_TRIGGER", obs, action, penalty=100.0, priority=2.0)
            return reward, terminated, False, info

        # 2. Semantic Hazard Keep-Out: 12 cm buffer around Hot Coffee Mug
        if dist_to_mug < 0.12:
            penetration = (0.12 - dist_to_mug)
            penalty = 30.0 + 100.0 * penetration
            reward -= penalty
            info["mistake"] = "MUG_PROXIMITY_VIOLATION"
            self.mistake_buffer.record_mistake("MUG_PROXIMITY_VIOLATION", obs, action, penalty=penalty, priority=1.5)
        else:
            reward += 0.05  # Safe distance bonus

        # 3. Surface Friction Traction (Wood vs Mat vs Paper)
        # Paper sheet is at X in [0.115, 0.325], Y in [-0.048, 0.248]
        on_paper = (0.115 <= rx <= 0.325) and (-0.048 <= ry <= 0.248)
        wheel_slip = abs(action[0] - action[1]) > 12.0
        if on_paper and wheel_slip:
            reward -= 5.0
            info["mistake"] = "WHEEL_SKID_PAPER"
            self.mistake_buffer.record_mistake("WHEEL_SKID_PAPER", obs, action, penalty=5.0, priority=1.0)

        # 4. Docking & Navigation Progress
        reward -= 0.8 * dist_to_dock
        if dist_to_dock < 0.08:
            reward += 15.0  # In docking pad
            if abs(action[0]) < 1.0 and abs(action[1]) < 1.0:
                reward += 25.0  # Decelerated smoothly into sleep nest

        # 5. Stylus Fetch Guidance
        if dist_to_stylus < 0.07:
            # Pincer closure reward
            pincer_closure = (action[6] + action[7]) * 0.5
            reward += 10.0 * pincer_closure

        # 6. Action Smoothness & Energy Efficiency
        reward -= 0.002 * float(np.sum(np.square(action)))

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
