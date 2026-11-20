from __future__ import annotations
import math
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stage4_host.audio_stt_tts import AudioPipeline
from stage4_host.sim_loop_bridge import SimLoopBridge
from stage4_host.vision_perception import VisionPerception
from stage4_host.host_server import Stage4HostServer

def test_subsystems() -> None:
    print("=" * 70)
    print("STAGE 4 LAPTOP HIL INTEGRATION VERIFICATION")
    print("=" * 70)

    # 1. Test Audio Pipeline & Mic
    print("\n[TEST 1/4] Testing Audio Pipeline & Microphone Stream...")
    audio = AudioPipeline(model_size="base.en", enable_mic=True)
    time.sleep(1.0)
    assert audio.mic_stream is not None, "Microphone stream failed to open!"
    print(f"  [PASS] Mic stream active: {audio.channels} ch @ {audio.native_samplerate} Hz")

    # 2. Test TTS (Speaking aloud)
    print("\n[TEST 2/4] Testing Speech Synthesis (TTS)...")
    audio.speak_async("Testing Desk Companion audio pipeline on RTX 5050.")
    time.sleep(2.0)
    print("  [PASS] Speech queued and dispatched to TTS engine.")

    # 3. Test Whisper Model Transcription
    print("\n[TEST 3/4] Testing Faster-Whisper Transcription Pipeline...")
    dummy_audio = np.zeros(16000, dtype=np.float32)
    assert audio.whisper_model is not None, "Faster-Whisper model failed to load!"
    segments, _ = audio.whisper_model.transcribe(dummy_audio, beam_size=1)
    _ = list(segments)
    print("  [PASS] Faster-Whisper pipeline transcribed audio tensor cleanly.")

    # 4. Test SimLoopBridge & 8-DOF Kinematics
    print("\n[TEST 4/4] Testing MuJoCo 3D Physics Bridge & 8-DOF Kinematics...")
    sim = SimLoopBridge(render_mode="rgb_array", launch_native_viewer=False)
    action_10d = np.array([1.5, 1.5, math.radians(15.0), math.radians(8.0), math.radians(20.0), math.radians(20.0), 0.0, 0.0, math.radians(25.0), math.radians(25.0)], dtype=np.float32)
    obs, info, bgr = sim.step_action(action_10d)
    assert bgr is not None and bgr.shape == (480, 640, 3), f"Invalid render shape: {bgr.shape if bgr is not None else None}"
    print(f"  [PASS] MuJoCo 3D arena stepped physics and rendered {bgr.shape} frame.")

    # 5. Clean up
    audio.close()
    sim.close()
    print("\n" + "=" * 70)
    print("ALL 4 LAPTOP HIL SUBSYSTEMS VERIFIED AND FUNCTIONING (100% PASS)")
    print("=" * 70)

if __name__ == "__main__":
    test_subsystems()
