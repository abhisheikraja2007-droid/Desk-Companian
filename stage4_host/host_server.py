from __future__ import annotations
import argparse
import asyncio
import json
import math
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set
import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from stable_baselines3 import PPO
import websockets

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stage3_ai.companion_brain import HumanAdaptiveCompanionBrain
from stage4_host.vision_perception import VisionPerception
from stage4_host.audio_stt_tts import AudioPipeline
from stage4_host.sim_loop_bridge import SimLoopBridge

console = Console()

class Stage4HostServer:
    def __init__(
        self,
        mode: str = "laptop",
        host: str = "0.0.0.0",
        port: int = 8765,
        model_path: str = "stage3_ai/models/ppo_desk_pet_final.zip",
        viewer: str = "composite",
        enable_gui: Optional[bool] = None
    ) -> None:
        self.mode = mode.lower()
        self.host = host
        self.port = port
        self.model_path = model_path
        if enable_gui is not None:
            self.viewer = "composite" if enable_gui else "none"
        else:
            self.viewer = viewer
        self.enable_gui = (self.viewer != "none")

        self.rl_model = None
        if os.path.exists(self.model_path):
            try:
                self.rl_model = PPO.load(self.model_path, device="cpu")
            except Exception:
                self.rl_model = None

        self.brain = HumanAdaptiveCompanionBrain(memory_db="stage3_ai/memory/companion_memory.db")
        self.vision = VisionPerception(camera_index=0)
        self.audio = AudioPipeline(model_size="base.en", enable_mic=(self.mode == "laptop"))
        self.audio.speech_callback = self._on_user_spoken_phrase

        self.sim_bridge: Optional[SimLoopBridge] = None
        if self.mode == "laptop":
            launch_native = (self.viewer == "native")
            self.sim_bridge = SimLoopBridge(render_mode="rgb_array", launch_native_viewer=launch_native)

        self.running = True
        self.connected_clients: Set[Any] = set()
        self.active_mode = "Body-Doubling Silent Presence"
        self.person_id = "Owner"
        self.bond_level = 78.5
        self.last_speech_reply = "Companion active and listening."
        self.last_input_text = "System Initialized"
        self.last_input_source = "SYSTEM"
        self.current_joints = {
            "neck_yaw": 0.0,
            "neck_pitch": 8.0,
            "left_ear": 10.0,
            "right_ear": 10.0,
            "left_arm": 5.0,
            "right_arm": 5.0,
            "left_pincer": 0.0,
            "right_pincer": 0.0
        }
        self.wheel_vels = [0.0, 0.0]
        self.cliff_states = [False, False, False, False]
        self.crown_touched = False
        self.round_trip_ms = 12.5
        self.last_status_print_time = 0.0

    def _on_user_spoken_phrase(self, phrase: str) -> None:
        self._on_user_input_command(phrase, source="VOICE")

    def _on_user_input_command(self, text: str, source: str = "KEYBOARD") -> None:
        text = text.strip()
        if not text:
            return

        self.last_input_text = text
        self.last_input_source = source
        print(f"\n[{source} INPUT]: \"{text}\"", flush=True)

        if text.lower() in ["exit", "quit", "q"]:
            self.running = False
            return

        text_lower = text.lower()
        cliff_sim = [0.06 if c else 0.02 for c in self.cliff_states]
        brain_out = self.brain.step(cliff_sim, self.crown_touched, [0, 0, 9.81], {"command": text_lower}, self.rl_model)
        self._apply_brain_output(brain_out)

        mood = brain_out.get("human_mood", "Focused")
        reply = self.brain.chat_engine.generate_reply(text, self.active_mode, self.bond_level, mood, self.person_id)
        self.last_speech_reply = reply
        self.audio.speak_async(reply)
        self._print_live_status()

    def _apply_brain_output(self, brain_out: Dict[str, Any]) -> None:
        self.active_mode = brain_out["mode"]
        self.person_id = brain_out["person_id"]
        self.bond_level = brain_out["bond_level"]
        for k, v in brain_out["target_joints_deg"].items():
            self.current_joints[k] = v

        if brain_out.get("action_intent") == "EMERGENCY_REVERSE":
            self.wheel_vels = [-6.0, -6.0]
        elif brain_out.get("action_intent") == "PARK_BEHIND_LAPTOP":
            self.wheel_vels = [0.0, 0.0]
        elif brain_out.get("action_intent") == "DOCK_SLEEP_NEST":
            self.wheel_vels = [2.0, 2.0]
        elif brain_out.get("action_intent") == "PURR_AND_WAVE":
            self.wheel_vels = [0.0, 0.0]
        else:
            self.wheel_vels = [3.0, 3.0]

    def _print_live_status(self) -> None:
        yaw = self.current_joints["neck_yaw"]
        pitch = self.current_joints["neck_pitch"]
        print(
            f"[STATUS] Mode: {self.active_mode:<30} | Bond: {self.bond_level:5.1f}% | "
            f"Head: ({yaw:+04.1f}°, {pitch:+04.1f}°) | Wheels: ({self.wheel_vels[0]:+.1f}, {self.wheel_vels[1]:+.1f}) | "
            f"Person: {self.person_id}",
            flush=True
        )
        self.last_status_print_time = time.time()

    def _keyboard_input_worker(self) -> None:
        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                self._on_user_input_command(line, source="KEYBOARD")
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break
            except Exception:
                time.sleep(0.1)

    async def _handle_websocket(self, websocket: Any, *args: Any, **kwargs: Any) -> None:
        self.connected_clients.add(websocket)
        console.print(f"[bold green]✔ Client Connected from {websocket.remote_address}[/bold green]")
        try:
            async for message in websocket:
                t_recv = time.time()
                try:
                    data = json.loads(message)
                except Exception:
                    continue

                if data.get("type") == "telemetry":
                    self.cliff_states = data.get("cliff", [False, False, False, False])
                    self.crown_touched = data.get("touch", False)
                    imu_data = data.get("imu", [0.0, 0.0, 9.81, 0.0, 0.0, 0.0])

                    cliff_sim = [0.06 if c else 0.02 for c in self.cliff_states]
                    brain_out = self.brain.step(
                        cliff_distances=cliff_sim,
                        crown_touch=self.crown_touched,
                        imu_accel=imu_data[:3],
                        human_signals={"guest": (self.person_id == "Guest")},
                        rl_model=self.rl_model
                    )
                    self._apply_brain_output(brain_out)

                    targets = [
                        self.current_joints["neck_yaw"] + 90.0,
                        self.current_joints["neck_pitch"] + 90.0,
                        self.current_joints["left_ear"] + 90.0,
                        self.current_joints["right_ear"] + 90.0,
                        self.current_joints["left_arm"] + 90.0,
                        self.current_joints["right_arm"] + 90.0,
                        self.current_joints["left_pincer"] + 90.0,
                        self.current_joints["right_pincer"] + 90.0
                    ]

                    expr = "happy" if self.crown_touched else ("alert" if any(self.cliff_states) else "focused")

                    response_pkt = {
                        "targets": targets,
                        "vel_l": self.wheel_vels[0],
                        "vel_r": self.wheel_vels[1],
                        "expression": expr,
                        "mode": self.active_mode,
                        "duration_ms": 200
                    }

                    await websocket.send(json.dumps(response_pkt))
                    self.round_trip_ms = (time.time() - t_recv) * 1000.0

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)
            console.print(f"[yellow]Client disconnected: {websocket.remote_address}[/yellow]")

    def run_laptop_hil_loop(self) -> None:
        console.print(Panel.fit(
            f"[bold #06B6D4]DESK PET: LAPTOP HARDWARE-IN-THE-LOOP (HIL) LIVE RUNTIME[/bold #06B6D4]\n"
            f"[dim]Mode: LAPTOP | Port: {self.port} | Mic/Cam Active | 3D MuJoCo Virtual Physics\n"
            f"Commands: Type in terminal (e.g. 'focus mode', 'fetch pen', 'hello') or speak aloud.[/dim]",
            border_style="#06B6D4"
        ))

        self.audio.speak_async("Hello Abhisheik! Desk companion hardware in the loop mode active on your RTX 5050.")

        last_step_t = time.time()
        last_smile_time = 0.0

        while self.running:
            now = time.time()
            dt = now - last_step_t
            if dt < 0.033:
                time.sleep(0.005)
                continue
            last_step_t = now

            ret, cam_frame = self.vision.read_frame()
            vis_info = {"face_detected": False, "smile_detected": False, "person_id": "Owner", "gaze_offset_x": 0.0, "gaze_offset_y": 0.0, "annotated_frame": cam_frame}
            if ret and cam_frame is not None:
                vis_info = self.vision.process_frame(cam_frame)
                if vis_info["smile_detected"] and (now - last_smile_time > 6.0):
                    last_smile_time = now
                    print("\n[WEBCAM]: Smile detected from Owner -> Perking ears & happy wave", flush=True)
                    brain_out = self.brain.step([0.02]*4, False, [0, 0, 9.81], {"smile": True}, self.rl_model)
                    self._apply_brain_output(brain_out)
                    self.audio.speak_async("I see you smiling Abhisheik! Ready to code.")
                    self._print_live_status()

                if vis_info["face_detected"]:
                    # Smoothly steer head to track user gaze offset
                    target_yaw = float(np.clip(-vis_info["gaze_offset_x"] * 28.0, -45.0, 45.0))
                    target_pitch = float(np.clip(-vis_info["gaze_offset_y"] * 18.0 + 8.0, -15.0, 25.0))
                    self.current_joints["neck_yaw"] = 0.85 * self.current_joints["neck_yaw"] + 0.15 * target_yaw
                    self.current_joints["neck_pitch"] = 0.85 * self.current_joints["neck_pitch"] + 0.15 * target_pitch

            render_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            if self.sim_bridge is not None:
                action_10d = np.array([
                    self.wheel_vels[0],
                    self.wheel_vels[1],
                    math.radians(self.current_joints["neck_yaw"]),
                    math.radians(self.current_joints["neck_pitch"]),
                    math.radians(self.current_joints["left_arm"]),
                    math.radians(self.current_joints["right_arm"]),
                    math.radians(self.current_joints["left_pincer"]),
                    math.radians(self.current_joints["right_pincer"]),
                    math.radians(self.current_joints["left_ear"]),
                    math.radians(self.current_joints["right_ear"])
                ], dtype=np.float32)

                obs, info, render_bgr = self.sim_bridge.step_action(action_10d)

            if now - self.last_status_print_time > 4.0:
                self._print_live_status()

            if self.enable_gui:
                composite = np.zeros((640, 1280, 3), dtype=np.uint8)
                composite[:50, :] = (23, 15, 11)   # Top Bar (#0B0F17)
                composite[530:, :] = (39, 24, 17)  # Bottom Bar (#111827)

                # Header text
                cv2.putText(composite, "DESK PET: 3D HIL COMPANION (RTX 5050)", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (212, 182, 6), 1)
                cv2.putText(composite, f"MODE: {self.active_mode}", (480, 32), cv2.FONT_HERSHEY_DUPLEX, 0.55, (8, 179, 234), 1)
                cv2.putText(composite, f"BOND: {self.bond_level:.1f}%", (1120, 32), cv2.FONT_HERSHEY_DUPLEX, 0.55, (166, 184, 20), 1)

                # Left side: Webcam Feed (640x480)
                webcam_img = vis_info.get("annotated_frame", cam_frame)
                if webcam_img is not None:
                    if webcam_img.shape[:2] != (480, 640):
                        webcam_img = cv2.resize(webcam_img, (640, 480))
                    composite[50:530, 0:640] = webcam_img
                cv2.rectangle(composite, (0, 50), (640, 530), (50, 40, 30), 2)
                cv2.putText(composite, "LIVE WEBCAM PERCEPTION (CAMERA 0)", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (212, 182, 6), 1)

                # Right side: MuJoCo 3D Arena (640x480)
                if render_bgr is not None:
                    if render_bgr.shape[:2] != (480, 640):
                        render_bgr = cv2.resize(render_bgr, (640, 480))
                    composite[50:530, 640:1280] = render_bgr
                cv2.rectangle(composite, (640, 50), (1280, 530), (50, 40, 30), 2)
                cv2.putText(composite, "3D MUJOCO DESK ARENA (8-DOF PHYSICS)", (655, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (166, 184, 20), 1)

                # Footer text
                cv2.putText(composite, f"LAST INPUT [{self.last_input_source}]: \"{self.last_input_text}\"", (20, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (234, 179, 8), 1)
                cv2.putText(composite, f"COMPANION: \"{self.last_speech_reply}\"", (20, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (166, 184, 20), 1)

                yaw_val = self.current_joints["neck_yaw"]
                pitch_val = self.current_joints["neck_pitch"]
                ear_l = self.current_joints["left_ear"]
                ear_r = self.current_joints["right_ear"]
                arm_l = self.current_joints["left_arm"]
                arm_r = self.current_joints["right_arm"]
                pin_l = self.current_joints["left_pincer"]
                pin_r = self.current_joints["right_pincer"]
                joints_str = f"8-DOF JOINTS: Yaw:{yaw_val:+04.1f} Pitch:{pitch_val:+04.1f} | Ears:({ear_l:.0f}, {ear_r:.0f}) | Arms:({arm_l:.0f}, {arm_r:.0f}) | Pincers:({pin_l:.0f}, {pin_r:.0f}) | Wheels:({self.wheel_vels[0]:+.1f}, {self.wheel_vels[1]:+.1f})"
                cv2.putText(composite, joints_str, (20, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1)
                cv2.putText(composite, "Press 'Q' or ESC in this window to exit | Type commands in PowerShell anytime", (20, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)

                cv2.imshow("Desk Pet 3D HIL Runtime (Webcam + 3D Desk Arena)", composite)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    self.running = False
                    break

    def start(self) -> None:
        async def run_server():
            async with websockets.serve(self._handle_websocket, self.host, self.port):
                console.print(f"[bold #14B8A6]★ WebSocket Server Running on ws://{self.host}:{self.port}/robot[/bold #14B8A6]")
                while self.running:
                    await asyncio.sleep(0.5)

        def ws_thread_fn():
            asyncio.run(run_server())

        t_ws = threading.Thread(target=ws_thread_fn, daemon=True)
        t_ws.start()

        if self.mode == "laptop":
            t_kb = threading.Thread(target=self._keyboard_input_worker, daemon=True)
            t_kb.start()
            self.run_laptop_hil_loop()
        else:
            console.print(Panel.fit(
                f"[bold #06B6D4]DESK PET: PHYSICAL ESP32-S3 ROBOT HOST RUNTIME[/bold #06B6D4]\n"
                f"[dim]Awaiting Wi-Fi WebSocket telemetry from ESP32-S3 on port {self.port}...[/dim]",
                border_style="#06B6D4"
            ))
            try:
                while self.running:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass

        self.cleanup()

    def cleanup(self) -> None:
        self.running = False
        self.vision.close()
        self.audio.close()
        if self.sim_bridge is not None:
            self.sim_bridge.close()
        cv2.destroyAllWindows()
        console.print("[dim]Stage 4 Host Server cleanly shut down.[/dim]")

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 Live RTX 5050 AI Host Server for Desk Pet")
    parser.add_argument("--mode", type=str, default="laptop", choices=["laptop", "esp32"], help="Execution mode (laptop HIL or physical esp32)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="WebSocket host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port (default: 8765)")
    parser.add_argument("--viewer", type=str, default="composite", choices=["composite", "native", "none"], help="3D viewer mode (composite OpenCV window, native MuJoCo, or none)")
    parser.add_argument("--no-gui", action="store_true", help="Run headless without OpenCV window")
    args = parser.parse_args()

    viewer_mode = "none" if args.no_gui else args.viewer

    server = Stage4HostServer(
        mode=args.mode,
        host=args.host,
        port=args.port,
        viewer=viewer_mode
    )
    server.start()

if __name__ == "__main__":
    main()
