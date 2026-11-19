from __future__ import annotations
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, List, Optional
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from stage3_ai.companion_brain import HumanAdaptiveCompanionBrain
from stage3_ai.mistake_replay import MistakeReplayBuffer

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

console = Console()

def render_dashboard(
    brain: HumanAdaptiveCompanionBrain,
    mistake_buffer: MistakeReplayBuffer,
    policy_name: str,
    session_mistakes: int,
    voice_enabled: bool,
    viewer_active: bool,
    state: Dict[str, Any],
    last_msg: str,
    last_reply: Optional[str] = None
) -> None:
    table = Table(title="[bold #06B6D4]DESK PET: 15-CAPABILITY HUMAN-ADAPTIVE BRAIN (CONVERGED POLICY)[/bold #06B6D4]", border_style="#14B8A6")
    table.add_column("Telemetry / Subsystem", style="bold white", width=28)
    table.add_column("Live State Value", style="#EAB308", width=46)

    table.add_row("Trained RL Policy", f"[bold green]{policy_name}[/bold green]")
    table.add_row("Active Companion Mode", f"[bold #06B6D4]{state.get('mode', 'IDLE')}[/bold #06B6D4]")
    table.add_row("Action Intent", str(state.get("action_intent", "NONE")))
    table.add_row("Biometric Bond Level", f"[#14B8A6]{state.get('bond_level', 0.0):.1f}%[/] (Dynamic Attachment)")
    table.add_row("Identified Interactor", f"[bold #EAB308]{state.get('person_id', 'Owner')}[/bold #EAB308]")
    table.add_row("Inferred Human Mood", f"[#06B6D4]{state.get('human_mood', 'Neutral')}[/#06B6D4]")
    table.add_row("Cliff Detection Sensors", "[green]NOMINAL (<12ms Brake)[/green]" if not state.get("cliff_danger", False) else "[red]CLIFF DANGER DETECTED[/red]")

    m_summary = mistake_buffer.get_mistake_summary()
    total_db = sum(m_summary.values())
    table.add_row("Mistake Replay Buffer", f"Session: [bold green]{session_mistakes}[/] mistakes | Replay DB: [dim]{total_db} historical[/dim]")

    joints_deg = state.get("target_joints_deg", {})
    short_names = {
        "neck_yaw": "Yaw",
        "neck_pitch": "Pit",
        "left_ear": "EarL",
        "right_ear": "EarR",
        "left_arm": "ArmL",
        "right_arm": "ArmR",
        "left_pincer": "PinL",
        "right_pincer": "PinR"
    }
    joint_str = " | ".join(f"{short_names.get(k, k)}:{v:+.1f}°" for k, v in joints_deg.items())
    table.add_row("Kinematic 8-DOF Pose", f"[bold white]{joint_str}[/bold white]")

    features_str = f"Voice TTS: {'[green]ON[/green]' if voice_enabled else '[dim]OFF[/dim]'} | 3D Viewer: {'[green]ACTIVE[/green]' if viewer_active else '[dim]OFF[/dim]'}"
    table.add_row("Active Runtime Engine", features_str)

    table.add_row("Last Interaction Log", f"[white]{last_msg}[/white]")
    if last_reply:
        table.add_row("Companion Voice/Chat", f"[bold #06B6D4]\"{last_reply}\"[/bold #06B6D4]")

    console.print(table)

def main() -> None:
    parser = argparse.ArgumentParser(description="Live Interactive Companion for Desk Pet")
    parser.add_argument("--viewer", action="store_true", help="Launch live 3D MuJoCo desktop window")
    parser.add_argument("--voice", action="store_true", help="Enable spoken voice output via Windows TTS")
    args = parser.parse_args()

    brain = HumanAdaptiveCompanionBrain()
    mistake_buffer = MistakeReplayBuffer()

    model_candidates = [
        "stage3_ai/models/ppo_desk_pet_final.zip",
        "stage3_ai/models/desk_pet_ppo_final.zip",
        "stage3_ai/models/test_ppo.zip"
    ]
    loaded_policy_name = "Embedded S-Curve & Kinematic Expert"
    rl_model: Optional[Any] = None

    if HAS_SB3:
        for p in model_candidates:
            if os.path.exists(p):
                try:
                    rl_model = PPO.load(p, device="cpu")
                    loaded_policy_name = f"PPO Converged ({os.path.basename(p)})"
                    break
                except Exception:
                    pass

    # Optional 3D MuJoCo Desktop Viewer setup
    viewer = None
    env = None
    if args.viewer:
        try:
            import mujoco
            import mujoco.viewer
            from stage3_ai.desk_arena import StudyDeskArenaEnv
            env = StudyDeskArenaEnv()
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            console.print("[green]✔ Live MuJoCo 3D Viewer launched successfully beside console.[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not launch 3D viewer: {e}[/yellow]")

    session_mistakes = 0
    voice_enabled = args.voice
    last_msg = f"Initialized companion with {loaded_policy_name}. Ready for voice/text commands."
    last_reply = "Hey Abhisheik! 8-DOF companion is online and monitoring your desk."

    default_cliffs = [0.015, 0.015, 0.015, 0.015]
    default_accel = [0.0, 0.0, 9.81]

    # Initial state
    state = brain.step(default_cliffs, False, default_accel, rl_model=rl_model)

    while True:
        console.clear()
        render_dashboard(
            brain,
            mistake_buffer,
            loaded_policy_name,
            session_mistakes,
            voice_enabled,
            args.viewer and viewer is not None,
            state,
            last_msg,
            last_reply
        )

        console.print("\n[bold #06B6D4]Interactive Control & Human Reaction Injections:[/bold #06B6D4]")
        console.print("  [#14B8A6][1][/] Smile at Robot             [#14B8A6][2][/] Pet Crown (TTP223)")
        console.print("  [#14B8A6][3][/] Sigh / Stressed Focus      [#14B8A6][4][/] Guest Enters Room")
        console.print("  [#14B8A6][5][/] Dismiss Wave               [#14B8A6][6][/] Trigger Cliff Warning")
        console.print("  [#EAB308]Commands:[/] 'helo' | 'focus mode' | 'fetch pen' | 'sleep nest' | 'voice on/off' | 'remember [fact]' | 'exit'\n")

        try:
            choice = Prompt.ask("[bold white]Enter option or natural voice command[/bold white]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting companion interactive console.[/yellow]")
            break

        if not choice:
            continue

        if choice.lower() in ("exit", "quit", "q"):
            console.print("[yellow]Exiting companion session.[/yellow]")
            break

        if choice.lower() == "voice on":
            voice_enabled = True
            last_msg = "Voice output enabled."
            last_reply = "Voice synthesis activated. I'll speak my replies now!"
            if voice_enabled:
                brain.chat_engine.speak_async(last_reply)
            continue
        elif choice.lower() == "voice off":
            voice_enabled = False
            last_msg = "Voice output muted."
            last_reply = "Voice muted. Running in silent text mode."
            continue

        signals: Dict[str, Any] = {}
        cliff_distances = list(default_cliffs)
        crown_touch = False

        if choice == "1":
            signals = {"smile": True}
            last_msg = "Perception: User smiled at robot. Biometric bond increased."
            last_reply = "Smiling back at you! Ears perked and ready."
            brain.memory.record_event("Owner", "Happy", "USER_SMILE", "SMILE_RECIPROCATE", 1.0)
        elif choice == "2":
            crown_touch = True
            signals = {"crown_touch": True}
            last_msg = "Touch ISR: TTP223 Crown petted! Left arm wave + ear wiggles."
            last_reply = "Crown touch sensed! That feels nice—let's keep building."
            brain.memory.record_event("Owner", "Happy", "CROWN_PET", "PURR_AND_WAVE", 1.0)
        elif choice == "3":
            signals = {"sigh": True, "typing": True}
            last_msg = "Acoustic Prosody: Heavy sigh detected. Transitioning to Empathetic Comfort."
            last_reply = "Deep breath, Abhisheik. Engineering is tough, but you're making real progress. I'm right here."
            brain.memory.record_event("Owner", "Stressed", "SIGH_DETECTED", "EMPATHETIC_TILT", 0.8)
        elif choice == "4":
            signals = {"guest": True}
            last_msg = "Vision Face ID: Unknown visitor detected! Engaging Confidentiality Guard."
            last_reply = "Visitor detected in room. Confidentiality guard active—project schematics locked."
            brain.memory.record_event("Guest", "Alert", "GUEST_ENTRY", "GUARD_MODE", 0.5)
        elif choice == "5":
            signals = {"typing": True, "dismiss": True}
            last_msg = "Gesture: Dismiss wave observed. Parking quietly behind laptop (60s cooldown)."
            last_reply = "Going quiet. Parking behind the screen so you can focus."
            brain.memory.record_event("Owner", "Focused", "DISMISS_WAVE", "PARK_BEHIND_LAPTOP", 0.9)
        elif choice == "6":
            cliff_distances[0] = 0.055
            session_mistakes += 1
            last_msg = "Safety Sensor: Front cliff sensor triggered! Initiating emergency reverse brake."
            last_reply = "Cliff edge detected! Hard braking applied—staying safe on the desk."
            brain.memory.record_event("Robot", "Alert", "CLIFF_TRIGGER", "EMERGENCY_REVERSE", -1.0)
            mistake_buffer.record_mistake("CLIFF_TRIGGER", np.zeros(36), np.zeros(10), 100.0, 2.0)
        else:
            signals = {"command": choice}
            last_reply = brain.chat_engine.generate_reply(
                prompt=choice,
                mode=state.get("mode", "Normal"),
                bond_level=state.get("bond_level", 80.0),
                mood=state.get("human_mood", "Neutral"),
                person_id=state.get("person_id", "Owner")
            )
            last_msg = f"Voice NLP: '{choice}'"
            brain.memory.record_event("Owner", "Command", "VOICE_PROMPT", choice, 0.7)

        # Update brain state with active signals
        state = brain.step(cliff_distances, crown_touch, default_accel, human_signals=signals, rl_model=rl_model)

        # Speak reply if voice is enabled
        if voice_enabled and last_reply:
            brain.chat_engine.speak_async(last_reply)

        # Update live MuJoCo 3D Viewer if active
        if viewer is not None and env is not None:
            try:
                rads = state.get("target_joints_rad", {})
                # Joint indices 14..21: neck_yaw, neck_pitch, left_arm, right_arm, left_pincer, right_pincer, left_ear, right_ear
                env.data.qpos[14] = rads.get("neck_yaw", 0.0)
                env.data.qpos[15] = rads.get("neck_pitch", 0.0)
                env.data.qpos[16] = rads.get("left_arm", 0.0)
                env.data.qpos[17] = rads.get("right_arm", 0.0)
                env.data.qpos[18] = rads.get("left_pincer", 0.0)
                env.data.qpos[19] = rads.get("right_pincer", 0.0)
                env.data.qpos[20] = rads.get("left_ear", 0.0)
                env.data.qpos[21] = rads.get("right_ear", 0.0)

                # Navigation position updates for 3D viewer
                intent = state.get("action_intent", "")
                if intent == "FETCH_STYLUS":
                    env.data.qpos[7] = 0.00
                    env.data.qpos[8] = 0.02
                elif intent == "DOCK_SLEEP_NEST":
                    env.data.qpos[7] = -0.48
                    env.data.qpos[8] = -0.20
                elif intent == "PARK_BEHIND_LAPTOP":
                    env.data.qpos[7] = 0.25
                    env.data.qpos[8] = -0.05
                elif intent == "APPROACH_HUMAN":
                    env.data.qpos[7] = 0.00
                    env.data.qpos[8] = -0.15

                mujoco.mj_forward(env.model, env.data)
                viewer.sync()
            except Exception:
                pass

if __name__ == "__main__":
    main()
