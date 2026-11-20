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
from rich.table import Table
from rich.live import Live
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
        enable_gui: bool = True
    ) -> None:
        self.mode = mode.lower()
        self.host = host
        self.port = port
        self.model_path = model_path
        self.enable_gui = enable_gui

        # 1. Load Converged PPO RL Policy
        self.rl_model = None
        if os.path.exists(self.model_path):
            try:
                self.rl_model = PPO.load(self.model_path, device="cpu")
            except Exception:
                self.rl_model = None

        # 2. Companion Brain & Memory
        self.brain = HumanAdaptiveCompanionBrain(memory_db="stage3_ai/memory/companion_memory.db")

        # 3. Vision & Audio Subsystems
        self.vision = VisionPerception(camera_index=0)
        self.audio = AudioPipeline(model_size="base.en", enable_mic=(self.mode == "laptop"))
        self.audio.speech_callback = self._on_user_spoken_phrase

        # 4. Simulation Bridge (for Laptop HIL Mode)
        self.sim_bridge: Optional[SimLoopBridge] = None
        if self.mode == "laptop":
            self.sim_bridge = SimLoopBridge(render_mode="rgb_array")

        # 5. Live State & Telemetry
        self.running = True
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.active_mode = "Body-Doubling Silent Presence"
        self.person_id = "Owner"
        self.bond_level = 78.5
        self.last_speech_reply = "Companion active and listening."
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

    def _on_user_spoken_phrase(self, phrase: str) -> None:
        console.print(f"[bold #EAB308][STT Transcribed]:[/] \"{phrase}\"")
        # Route spoken intent through conversational engine
        reply = self.brain.chat_engine.generate_reply(phrase, self.active_mode)
        self.last_speech_reply = reply
        console.print(f"[bold #14B8A6][Companion Reply]:[/] \"{reply}\"")
        self.audio.speak_async(reply)

        # Trigger mode override if recognized
        phrase_lower = phrase.lower()
        cliff_sim = [0.06 if c else 0.02 for c in self.cliff_states]
        brain_out = self.brain.step(cliff_sim, self.crown_touched, [0, 0, 9.81], {"command": phrase_lower}, self.rl_model)
        self._apply_brain_output(brain_out)

    def _apply_brain_output(self, brain_out: Dict[str, Any]) -> None:
        self.active_mode = brain_out["mode"]
        self.person_id = brain_out["person_id"]
        self.bond_level = brain_out["bond_level"]
        for k, v in brain_out["target_joints_deg"].items():
            self.current_joints[k] = v

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

                # Process incoming ESP32 telemetry packet
                if data.get("type") == "telemetry":
                    self.cliff_states = data.get("cliff", [False, False, False, False])
                    self.crown_touched = data.get("touch", False)
                    imu_data = data.get("imu", [0.0, 0.0, 9.81, 0.0, 0.0, 0.0])

                    # Step brain with live sensor inputs
                    cliff_sim = [0.06 if c else 0.02 for c in self.cliff_states]
                    brain_out = self.brain.step(
                        cliff_distances=cliff_sim,
                        crown_touch=self.crown_touched,
                        imu_accel=imu_data[:3],
                        human_signals={"guest": (self.person_id == "Guest")},
                        rl_model=self.rl_model
                    )
                    self._apply_brain_output(brain_out)

                    # Compute wheel control commands
                    if brain_out["action_intent"] == "EMERGENCY_REVERSE":
                        vl, vr = -6.0, -6.0
                        expr = "alert"
                    elif brain_out["action_intent"] == "PARK_BEHIND_LAPTOP":
                        vl, vr = 0.0, 0.0
                        expr = "focused"
                    elif brain_out["action_intent"] == "DOCK_SLEEP_NEST":
                        vl, vr = 2.0, 2.0
                        expr = "sleepy"
                    elif brain_out["action_intent"] == "PURR_AND_WAVE":
                        vl, vr = 0.0, 0.0
                        expr = "happy"
                    else:
                        vl, vr = 3.5, 3.5
                        expr = "neutral"

                    self.wheel_vels = [vl, vr]

                    # Prepare outgoing command packet for ESP32
                    # SCServo targets: [Yaw, Pitch, EarL, EarR, ArmL, ArmR, PinL, PinR]
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

                    response_pkt = {
                        "targets": targets,
                        "vel_l": vl,
                        "vel_r": vr,
                        "expression": expr,
                        "mode": self.active_mode,
                        "duration_ms": 200
                    }

                    await websocket.send(json.dumps(response_pkt))
                    self.round_trip_ms = (time.time() - t_recv) * 1000.0

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.remove(websocket)
            console.print(f"[yellow]Client disconnected: {websocket.remote_address}[/yellow]")

    def run_laptop_hil_loop(self) -> None:
        console.print(Panel.fit(
            f"[bold #06B6D4]DESK PET: LAPTOP HARDWARE-IN-THE-LOOP (HIL) LIVE RUNTIME[/bold #06B6D4]\n"
            f"[dim]Mode: {self.mode.upper()} | Port: {self.port} | Mic/Cam Active | 3D MuJoCo Virtual Physics[/dim]",
            border_style="#06B6D4"
        ))

        # Initial Welcome Speech
        self.audio.speak_async("Hello Abhisheik! Desk companion hardware in the loop mode active on your RTX 5050.")

        last_step_t = time.time()
        while self.running:
            now = time.time()
            dt = now - last_step_t
            if dt < 0.033:  # ~30 Hz loop
                time.sleep(0.005)
                continue
            last_step_t = now

            # 1. Vision Perception (Webcam / Synthetic Avatar)
            ret, cam_frame = self.vision.read_frame()
            if ret:
                vis_info = self.vision.process_frame(cam_frame)
                if vis_info["smile_detected"] and self.active_mode != "Playful Banter Routine":
                    brain_out = self.brain.step([0.02]*4, False, [0, 0, 9.81], {"smile": True}, self.rl_model)
                    self._apply_brain_output(brain_out)
                    self.audio.speak_async("I see you smiling Abhisheik! Ready to code.")

            # 2. Step 3D MuJoCo Physics Simulation
            if self.sim_bridge is not None:
                # Convert joints to radian action vector for MuJoCo
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

                # Overlay HUD on HIL window
                cv2.rectangle(render_bgr, (0, 0), (640, 40), (23, 15, 11), -1)
                cv2.putText(render_bgr, f"MODE: {self.active_mode}", (15, 26), cv2.FONT_HERSHEY_DUPLEX, 0.48, (212, 182, 6), 1)
                cv2.putText(render_bgr, f"BOND: {self.bond_level:.1f}%", (490, 26), cv2.FONT_HERSHEY_DUPLEX, 0.48, (8, 179, 234), 1)

                if self.enable_gui:
                    cv2.imshow("Desk Pet 3D HIL Runtime (Press Q to Exit)", render_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
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
    parser.add_argument("--no-gui", action="store_true", help="Run headless without OpenCV window")
    args = parser.parse_args()

    server = Stage4HostServer(
        mode=args.mode,
        host=args.host,
        port=args.port,
        enable_gui=not args.no_gui
    )
    server.start()

if __name__ == "__main__":
    main()
