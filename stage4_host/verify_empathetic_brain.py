from __future__ import annotations
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stage3_ai.companion_brain import HumanAdaptiveCompanionBrain, ConversationalEngine, CompanionMemory
from stage4_host.host_server import Stage4HostServer

def test_conversational_variety_and_empathy() -> None:
    mem = CompanionMemory("stage3_ai/memory/companion_memory.db")
    engine = ConversationalEngine(mem)

    # Test 1: Sad / Stressed / Tired
    reply_sad1 = engine.generate_reply("im sad", "Attentive Desk Companion", 80.0, "sad", "Owner")
    reply_sad2 = engine.generate_reply("so tired and stressed", "Attentive Desk Companion", 80.0, "sad", "Owner")
    assert "right here" in reply_sad1 or "breather" in reply_sad1 or "marathon" in reply_sad1 or "pressure" in reply_sad1 or "alone" in reply_sad1, f"Unexpected sad reply: {reply_sad1}"
    assert reply_sad1 != reply_sad2, f"Expected varied sad responses! Got identical: '{reply_sad1}'"
    print(f"[TEST PASS] Sad Response 1: {reply_sad1}")
    print(f"[TEST PASS] Sad Response 2: {reply_sad2}")

    # Test 2: Come Here
    reply_come = engine.generate_reply("come here", "Attentive Desk Companion", 80.0, "Focused", "Owner")
    assert any(k in reply_come for k in ["Rolling", "Approaching", "On my way", "station", "buffer"]), f"Unexpected come reply: {reply_come}"
    print(f"[TEST PASS] Come Here Response: {reply_come}")

    # Test 3: Go Away
    reply_away = engine.generate_reply("go away and give me space", "Attentive Desk Companion", 80.0, "Neutral", "Owner")
    assert any(k in reply_away for k in ["space", "quietly parked", "corner nest", "dock"]), f"Unexpected retreat reply: {reply_away}"
    print(f"[TEST PASS] Go Away Response: {reply_away}")

    # Test 4: Varied Joyful Responses (No Duplicate Strings)
    seen_happy = set()
    for _ in range(4):
        rep = engine.generate_reply("smile", "Playful Banter Routine", 85.0, "Happy", "Owner")
        assert rep not in seen_happy, f"Duplicate reply returned: {rep}"
        seen_happy.add(rep)
        print(f"  [HAPPY VARIANT]: {rep}")
    print(f"[TEST PASS] Generated {len(seen_happy)} unique joyful responses without repetition.")

    # Test 5: Varied Greetings (No Duplicate Strings)
    seen_greets = set()
    for _ in range(3):
        rep = engine.generate_reply("hello", "Attentive Desk Companion", 85.0, "Focused", "Owner")
        assert rep not in seen_greets, f"Duplicate greeting returned: {rep}"
        seen_greets.add(rep)
        print(f"  [GREET VARIANT]: {rep}")
    print(f"[TEST PASS] Generated {len(seen_greets)} unique greetings without repetition.")

def test_brain_kinematics_and_modes() -> None:
    brain = HumanAdaptiveCompanionBrain("stage3_ai/memory/companion_memory.db")

    # 1. Test "im sad"
    out_sad = brain.step([0.02]*4, False, [0, 0, 9.81], {"command": "im sad"})
    assert out_sad["mode"] == "Empathetic Comfort Mode", f"Unexpected mode: {out_sad['mode']}"
    assert out_sad["action_intent"] == "EMPATHETIC_TILT", f"Unexpected intent: {out_sad['action_intent']}"
    assert out_sad["human_mood"] == "sad", f"Unexpected mood: {out_sad['human_mood']}"
    joints_sad = out_sad["target_joints_deg"]
    assert joints_sad["neck_yaw"] == -8.0, f"Expected neck_yaw -8.0, got {joints_sad['neck_yaw']}"
    assert joints_sad["neck_pitch"] == 12.0, f"Expected neck_pitch 12.0, got {joints_sad['neck_pitch']}"
    assert joints_sad["left_ear"] == -12.0 and joints_sad["right_ear"] == -12.0, f"Expected ears -12.0, got {joints_sad}"
    print("[TEST PASS] Brain correctly transitions to Empathetic Comfort Mode with -8° yaw, +12° pitch, -12° ear droop.")

    # 2. Test "come here"
    out_come = brain.step([0.02]*4, False, [0, 0, 9.81], {"command": "come here"})
    assert out_come["mode"] == "Attentive Approach", f"Unexpected mode: {out_come['mode']}"
    assert out_come["action_intent"] == "APPROACH_HUMAN", f"Unexpected intent: {out_come['action_intent']}"
    joints_come = out_come["target_joints_deg"]
    assert joints_come["neck_pitch"] == 15.0, f"Expected neck_pitch 15.0, got {joints_come['neck_pitch']}"
    assert joints_come["left_ear"] == 30.0 and joints_come["right_ear"] == 30.0, f"Expected ears +30.0, got {joints_come}"
    print("[TEST PASS] Brain correctly transitions to Attentive Approach with +15° pitch, +30° perked ears.")

    # 3. Test "go away"
    out_away = brain.step([0.02]*4, False, [0, 0, 9.81], {"command": "go away"})
    assert out_away["mode"] == "Circadian Night Wind-Down", f"Unexpected mode: {out_away['mode']}"
    assert out_away["action_intent"] == "RETREAT_SPACE", f"Unexpected intent: {out_away['action_intent']}"
    joints_away = out_away["target_joints_deg"]
    assert joints_away["left_ear"] == -25.0 and joints_away["right_ear"] == -25.0, f"Expected ears -25.0, got {joints_away}"
    print("[TEST PASS] Brain correctly transitions to Circadian retreat with -25° tucked ears.")

def test_host_server_timed_movement() -> None:
    server = Stage4HostServer(mode="test", viewer="none")

    # Command: "come here"
    server._on_user_input_command("come here", source="TEST")
    assert server.timed_movement_cmd == "come_here"
    assert server.timed_movement_duration == 2.5
    assert server.wheel_vels == [3.5, 3.5], f"Expected wheels [3.5, 3.5], got {server.wheel_vels}"
    print("[TEST PASS] Host server executes forward motion [3.5, 3.5] for 'come here'.")

    # Command: "go away"
    server._on_user_input_command("go away", source="TEST")
    assert server.timed_movement_cmd == "go_away"
    assert server.timed_movement_duration == 3.0
    assert server.wheel_vels == [-2.8, -2.8], f"Expected wheels [-2.8, -2.8], got {server.wheel_vels}"
    print("[TEST PASS] Host server executes reverse retreat [-2.8, -2.8] for 'go away'.")

    # Command: "im sad"
    server._on_user_input_command("im sad", source="TEST")
    assert server.timed_movement_cmd is None
    assert server.wheel_vels == [0.0, 0.0], f"Expected parked wheels [0.0, 0.0], got {server.wheel_vels}"
    assert server.active_mode == "Empathetic Comfort Mode"
    print("[TEST PASS] Host server parks cleanly (0.0, 0.0) in Empathetic Comfort Mode.")

if __name__ == "__main__":
    test_conversational_variety_and_empathy()
    test_brain_kinematics_and_modes()
    test_host_server_timed_movement()
    print("\nALL EMPATHETIC BRAIN & TIMED MOVEMENT TESTS PASSED 100%!")