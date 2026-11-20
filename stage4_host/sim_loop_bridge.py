from __future__ import annotations
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stage3_ai.desk_arena import StudyDeskArenaEnv

class SimLoopBridge:
    def __init__(self, render_mode: str = "rgb_array") -> None:
        self.env = StudyDeskArenaEnv(render_mode=render_mode, curriculum_phase=2)
        self.obs, _ = self.env.reset()
        self.renderer = mujoco.Renderer(self.env.model, height=480, width=640)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0.0, 0.0, 0.05]
        self.cam.distance = 0.95
        self.cam.elevation = -22.0
        self.cam.azimuth = 60.0

        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        self.root_jnt_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qpos_adr = self.env.model.jnt_qposadr[self.root_jnt_id]
        self.root_dof_adr = self.env.model.jnt_dofadr[self.root_jnt_id]

    def step_action(self, action_10d: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
        # Step environment physics
        self.obs, rew, term, trunc, info = self.env.step(action_10d)
        if term or trunc:
            self.obs, _ = self.env.reset()

        # Enforce upright orientation guard
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

        # Render 3D frame
        self.renderer.update_scene(self.env.data, camera=self.cam, scene_option=self.scene_option)
        rgb = self.renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        return self.obs, info, bgr

    def close(self) -> None:
        self.env.close()
