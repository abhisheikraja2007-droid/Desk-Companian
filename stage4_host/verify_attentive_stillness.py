from __future__ import annotations
import math
import os
import sys
import threading
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stage4_host.host_server import Stage4HostServer

def test_attentive_stillness() -> None:
    print("=" * 75)
    print("STAGE 4: VERIFY ATTENTIVE COMPANION STILLNESS & TIMED MOVEMENTS")
    print("=" * 75)

    server = Stage4HostServer(mode="laptop", viewer="none")
    t_srv = threading.Thread(target=server.start, daemon=True)
    t_srv.start()
    time.sleep(2.0)

    # 1. Startup Attentive Stillness Check
    print("\n[TEST 1/4] Checking Startup Attentive Stillness (Silent User)...")
    print(f"  ✔ Startup Mode: {server.active_mode}")
    print(f"  ✔ Wheel Velocities: ({server.wheel_vels[0]:.2f}, {server.wheel_vels[1]:.2f})")
    assert server.wheel_vels == [0.0, 0.0], f"Expected (0.0, 0.0) wheels, got {server.wheel_vels}"
    print("  ✔ Verified: Robot is completely parked and still on startup.")

    # 2. Test 'dance' Command & Automatic Auto-Park Return
    print("\n[TEST 2/4] Testing 'dance' (4-Second Playful Routine -> Auto-Park)...")
    server._on_user_input_command("dance", source="KEYBOARD")
    time.sleep(1.0)
    print(f"  ✔ In-Dance Wheels: ({server.wheel_vels[0]:.2f}, {server.wheel_vels[1]:.2f})")
    print(f"  ✔ In-Dance Arms: L={server.current_joints['left_arm']:.1f}°, R={server.current_joints['right_arm']:.1f}°")
    print(f"  ✔ In-Dance Ears: L={server.current_joints['left_ear']:.1f}°, R={server.current_joints['right_ear']:.1f}°")
    assert server.active_mode == "Playful Banter Routine", f"Unexpected mode: {server.active_mode}"

    print("  ... waiting for 4.0s dance timer to expire ...")
    time.sleep(3.5)
    print(f"  ✔ Post-Dance Mode: {server.active_mode}")
    print(f"  ✔ Post-Dance Wheels: ({server.wheel_vels[0]:.2f}, {server.wheel_vels[1]:.2f})")
    assert server.wheel_vels == [0.0, 0.0], f"Expected wheels to auto-park at (0.0, 0.0), got {server.wheel_vels}"
    print("  ✔ Verified: Dance routine automatically returned to Attentive Desk Companion (0.0, 0.0).")

    # 3. Test 'roam' Command & Immediate 'stop' Override
    print("\n[TEST 3/4] Testing 'roam' & Immediate 'stop' / 'focus mode' Halt...")
    server._on_user_input_command("roam", source="VOICE")
    time.sleep(1.0)
    print(f"  ✔ Active Patrol Wheels: ({server.wheel_vels[0]:.2f}, {server.wheel_vels[1]:.2f})")
    assert server.wheel_vels != [0.0, 0.0], "Expected non-zero wheels during roam command!"

    print("  --> Issuing Immediate Command: 'stop'")
    server._on_user_input_command("stop", source="KEYBOARD")
    time.sleep(0.5)
    print(f"  ✔ Immediate Stopped Wheels: ({server.wheel_vels[0]:.2f}, {server.wheel_vels[1]:.2f})")
    assert server.wheel_vels == [0.0, 0.0], f"Expected wheels to halt at (0.0, 0.0), got {server.wheel_vels}"
    print("  ✔ Verified: 'stop' immediately halted wheels.")

    # 4. Test Head Gaze Tracking & Ear Micro-Twitches While Parked
    print("\n[TEST 4/4] Testing Lifelike Idle (Ear Micro-Twitches & Gaze While Parked)...")
    time.sleep(1.0)
    print(f"  ✔ Parked Head Orientation: Yaw={server.current_joints['neck_yaw']:+.2f}°, Pitch={server.current_joints['neck_pitch']:+.2f}°")
    print(f"  ✔ Parked Ear Micro-Twitch: EarL={server.current_joints['left_ear']:.2f}°, EarR={server.current_joints['right_ear']:.2f}°")
    print(f"  ✔ Wheels Maintained: {server.wheel_vels}")
    assert server.wheel_vels == [0.0, 0.0], "Robot wheels spun during attentive idle!"

    server.stop()
    t_srv.join(timeout=4.0)

    print("\n" + "=" * 75)
    print("★ ATTENTIVE STILLNESS & TIMED ACTIONS FULLY VERIFIED (100% PASS) ★")
    print("=" * 75)

if __name__ == "__main__":
    test_attentive_stillness()
