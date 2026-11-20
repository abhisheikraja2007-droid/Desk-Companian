from __future__ import annotations
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import mujoco
try:
    import mujoco.viewer
except Exception:
    pass
import numpy as np
import warnings

warnings.filterwarnings("ignore", message=".*WGL: Failed to make context current.*")
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stage3_ai.desk_arena import StudyDeskArenaEnv

class SimLoopBridge:
    def __init__(self, render_mode: Optional[str] = None, launch_native_viewer: bool = False) -> None:
        self.env = StudyDeskArenaEnv(render_mode=None, curriculum_phase=2)
        self.obs, _ = self.env.reset()
        self.renderer = mujoco.Renderer(self.env.model, height=480, width=640)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0.0, 0.0, 0.06]
        self.cam.distance = 0.90
        self.cam.elevation = -22.0
        self.cam.azimuth = 65.0

        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        self.root_jnt_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qpos_adr = self.env.model.jnt_qposadr[self.root_jnt_id]
        self.root_dof_adr = self.env.model.jnt_dofadr[self.root_jnt_id]

        self.mug_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "coffee_mug")
        self.laptop_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "laptop")

        self.passive_viewer = None
        if launch_native_viewer and hasattr(mujoco, "viewer"):
            try:
                self.passive_viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
                print("[SIM] Native MuJoCo passive viewer launched.")
            except Exception as e:
                print(f"[SIM WARN] Native viewer launch fallback to OpenCV: {e}")
                self.passive_viewer = None

    def step_action(self, action_10d: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
        # Autonomous Collision Avoidance Guard for Patrol
        rx = float(self.env.data.qpos[self.root_qpos_adr + 0])
        ry = float(self.env.data.qpos[self.root_qpos_adr + 1])

        # Coffee Mug keep-out (>16cm clearance)
        if self.mug_id != -1:
            mx = float(self.env.data.xpos[self.mug_id, 0])
            my = float(self.env.data.xpos[self.mug_id, 1])
            d_mug = math.sqrt((rx - mx)**2 + (ry - my)**2)
            if d_mug < 0.18:
                action_10d[0] = -2.0
                action_10d[1] = 4.0

        # Desk Boundary Margin (>10cm margin)
        if abs(rx) > 0.36 or abs(ry) > 0.26:
            heading_to_center = math.atan2(-ry, -rx)
            action_10d[0] = 3.5 * math.cos(heading_to_center) + 1.5
            action_10d[1] = 3.5 * math.sin(heading_to_center) - 1.5

        self.obs, rew, term, trunc, info = self.env.step(action_10d)
        if term or trunc:
            self.obs, _ = self.env.reset()

        # Enforce upright attitude
        qw = float(self.env.data.qpos[self.root_qpos_adr + 3])
        qz = float(self.env.data.qpos[self.root_qpos_adr + 6])
        norm = math.sqrt(qw * qw + qz * qz)
        if norm > 1e-6:
            self.env.data.qpos[self.root_qpos_adr + 3] = qw / norm
            self.env.data.qpos[self.root_qpos_adr + 4] = 0.0
            self.env.data.qpos[self.root_qpos_adr + 5] = 0.0
            self.env.data.qpos[self.root_qpos_adr + 6] = qz / norm
        self.env.data.qpos[self.root_qpos_adr + 2] = 0.017
        mujoco.mj_forward(self.env.model, self.env.data)

        if self.passive_viewer is not None and self.passive_viewer.is_running():
            self.passive_viewer.sync()

        rx = float(self.env.data.qpos[self.root_qpos_adr + 0])
        ry = float(self.env.data.qpos[self.root_qpos_adr + 1])
        self.cam.lookat[0] = rx * 0.7
        self.cam.lookat[1] = ry * 0.7

        self.renderer.update_scene(self.env.data, camera=self.cam, scene_option=self.scene_option)
        rgb = self.renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        return self.obs, info, bgr

    def close(self) -> None:
        if self.passive_viewer is not None:
            try:
                self.passive_viewer.close()
            except Exception:
                pass
        self.env.close()
