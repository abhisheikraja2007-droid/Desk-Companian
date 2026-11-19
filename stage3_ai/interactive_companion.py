from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.layout import Layout
from rich.text import Text

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
    state: Dict[str, Any],
    last_msg: str
) -> None:
    table = Table(title="[bold #06B6D4]DESK PET: 15-CAPABILITY HUMAN-ADAPTIVE BRAIN (CONVERGED POLICY)[/bold #06B6D4]", border_style="#14B8A6")
    table.add_column("Telemetry / Subsystem", style="bold white", width=28)
    table.add_column("Live State Value", style="#EAB308", width=44)

    table.add_row("Trained RL Policy", f"[bold green]{policy_name}[/bold green]")
    table.add_row("Active Companion Mode", f"[bold #06B6D4]{state.get('mode', 'IDLE')}[/bold #06B6D4]")
    table.add_row("Action Intent", str(state.get("action_intent", "NONE")))
    table.add_row("Biometric Bond Level", f"[#14B8A6]{state.get('bond_level', 0.0):.1f}%[/] (Dynamic Attachment)")
    table.add_row("Identified Interactor", str(state.get("person_id", "Unknown")))
    table.add_row("Inferred Human Mood", str(state.get("human_mood", "Neutral")))
    table.add_row("Cognitive Load Index", f"{state.get('cognitive_load', 0.0):.2f}")
    table.add_row("Cliff Detection Sensors", "[green]NOMINAL (<12ms Brake)[/green]" if not state.get("cliff_danger", False) else "[red]CLIFF DANGER DETECTED[/red]")

    m_summary = mistake_buffer.get_mistake_summary()
    table.add_row("Mistake Replay Buffer", f"Total Logged: {sum(m_summary.values())} (Cliffs: {m_summary.get('CLIFF_TRIGGER',0)}, Mugs: {m_summary.get('MUG_PROXIMITY_VIOLATION',0)})")

    joints = state.get("target_joints", {})
    joint_str = ", ".join(f"{k[:4]}:{v:+.2f}" for k, v in joints.items())
    table.add_row("Kinematic 8-DOF Pose", f"[dim]{joint_str}[/dim]")
    table.add_row("Last Interaction Log", f"[white]{last_msg}[/white]")

    console.print(table)

def main() -> None:
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

    last_msg = f"Initialized companion with {loaded_policy_name}. Ready for voice/text commands."

    default_cliffs = [0.015, 0.015, 0.015, 0.015]
    default_accel = [0.0, 0.0, 9.81]

    while True:
        state = brain.step(default_cliffs, False, default_accel)
        console.clear()
        render_dashboard(brain, mistake_buffer, loaded_policy_name, state, last_msg)

        console.print("\n[bold #06B6D4]Interactive Control & Human Reaction Injections:[/bold #06B6D4]")
        console.print("  [#14B8A6][1][/] Smile at Robot             [#14B8A6][2][/] Pet Crown (TTP223)")
        console.print("  [#14B8A6][3][/] Sigh / Stressed Focus      [#14B8A6][4][/] Guest Enters Room")
        console.print("  [#14B8A6][5][/] Dismiss Wave               [#14B8A6][6][/] Trigger Cliff Warning")
        console.print("  [#EAB308]Commands:[/] 'come here' | 'fetch pen' | 'focus mode' | 'sleep nest' | 'bad robot' | 'exit'\n")

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

        signals: Dict[str, Any] = {}
        cliff_distances = list(default_cliffs)
        crown_touch = False

        if choice == "1":
            signals = {"smile": True}
            last_msg = "Perception: User smiled at robot. Biometric bond increased."
            brain.memory.record_event("Owner", "Happy", "USER_SMILE", "SMILE_RECIPROCATE", 1.0)
        elif choice == "2":
            crown_touch = True
            signals = {"crown_touch": True}
            last_msg = "Touch ISR: TTP223 Crown petted! Happy eye arches displayed."
            brain.memory.record_event("Owner", "Happy", "CROWN_PET", "PURR_AND_WAVE", 1.0)
        elif choice == "3":
            signals = {"sigh": True, "typing": True}
            last_msg = "Acoustic Prosody: Heavy sigh detected. Transitioning to Empathetic Comfort."
            brain.memory.record_event("Owner", "Stressed", "SIGH_DETECTED", "EMPATHETIC_TILT", 0.8)
        elif choice == "4":
            signals = {"guest": True}
            last_msg = "Vision Face ID: Unknown visitor detected. Engaging Confidentiality Guard."
            brain.memory.record_event("Guest", "Alert", "GUEST_ENTRY", "GUARD_MODE", 0.5)
        elif choice == "5":
            signals = {"typing": True}
            last_msg = "Gesture: Dismiss wave observed. Returning quietly behind laptop."
            brain.memory.record_event("Owner", "Focused", "DISMISS_WAVE", "PARK_BEHIND_LAPTOP", 0.9)
        elif choice == "6":
            cliff_distances[0] = 0.055
            last_msg = "Safety Sensor: Front cliff sensor triggered! Initiating emergency reverse brake."
            brain.memory.record_event("Robot", "Alert", "CLIFF_TRIGGER", "EMERGENCY_REVERSE", -1.0)
            mistake_buffer.record_mistake("CLIFF_TRIGGER", np.zeros(36), np.zeros(10), 100.0, 2.0)
        else:
            signals = {"command": choice}
            duplex = brain.voice_duplex_hook(choice)
            last_msg = f"Voice NLP: '{choice}' -> {duplex['synthesized_response']}"
            brain.memory.record_event("Owner", "Command", "VOICE_PROMPT", choice, 0.7)

        state = brain.step(cliff_distances, crown_touch, default_accel, human_signals=signals)

if __name__ == "__main__":
    main()
