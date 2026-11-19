from __future__ import annotations
import json
import math
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

class CompanionMemory:
    def __init__(self, db_path: str = "stage3_ai/memory/companion_memory.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self) -> None:
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS milestones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                category TEXT,
                description TEXT,
                details_json TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                weight REAL,
                last_updated REAL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS episodic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                speaker TEXT,
                mood TEXT,
                event_type TEXT,
                action_taken TEXT,
                reward REAL
            )
        """)
        self.conn.commit()

        # Seed initial user preferences if empty
        c.execute("SELECT COUNT(*) FROM user_preferences")
        if c.fetchone()[0] == 0:
            defaults = [
                ("stealth_during_deep_focus", 0.95),
                ("pen_handover_readiness", 0.90),
                ("mug_keepout_strictness", 0.99),
                ("playful_banter_allowed", 0.70),
                ("night_winddown_lighting", 0.85)
            ]
            for k, w in defaults:
                c.execute("INSERT OR REPLACE INTO user_preferences VALUES (?, ?, ?)", (k, w, time.time()))
            self.conn.commit()

    def record_event(self, speaker: str, mood: str, event_type: str, action: str, reward: float) -> None:
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO episodic_events (timestamp, speaker, mood, event_type, action_taken, reward)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (time.time(), speaker, mood, event_type, action, reward))
        self.conn.commit()

    def consolidate_nightly_memory(self) -> Dict[str, float]:
        c = self.conn.cursor()
        c.execute("SELECT event_type, AVG(reward) FROM episodic_events GROUP BY event_type")
        summary = {}
        for ev, avg_r in c.fetchall():
            summary[ev] = float(avg_r)
            pref_key = f"pref_{ev.lower()}"
            c.execute("""
                INSERT INTO user_preferences (key, weight, last_updated)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET weight = weight * 0.9 + ? * 0.1, last_updated = ?
            """, (pref_key, max(0.1, min(1.0, 0.5 + avg_r * 0.1)), time.time(), max(0.1, min(1.0, 0.5 + avg_r * 0.1)), time.time()))
        self.conn.commit()
        return summary

    def get_preferences(self) -> Dict[str, float]:
        c = self.conn.cursor()
        c.execute("SELECT key, weight FROM user_preferences")
        return {k: v for k, v in c.fetchall()}

class MultimodalPerception:
    def __init__(self) -> None:
        self.cliff_states = [0, 0, 0, 0]
        self.last_pet_time = 0.0
        self.is_petted = False
        self.face_detected = True
        self.person_id = "Owner"
        self.prosody_pitch_hz = 185.0
        self.typing_cadence_wpm = 75.0
        self.imu_shake_intensity = 0.0

    def update(
        self,
        cliff_readings: List[float],
        crown_touch: bool,
        imu_accel: List[float],
        typing_active: bool = False,
        sigh_detected: bool = False,
        guest_active: bool = False
    ) -> Dict[str, Any]:
        self.cliff_states = [1 if d > 0.038 else 0 for d in cliff_readings]
        self.is_petted = crown_touch
        if crown_touch:
            self.last_pet_time = time.time()

        self.person_id = "Guest" if guest_active else "Owner"
        self.typing_cadence_wpm = 85.0 if typing_active else 0.0

        ax, ay, az = imu_accel
        accel_mag = math.sqrt(ax**2 + ay**2 + (az - 9.81)**2)
        self.imu_shake_intensity = accel_mag

        if sigh_detected:
            mood = "Stressed"
            cognitive_load = 0.85
        elif typing_active:
            mood = "Focused"
            cognitive_load = 0.90
        elif crown_touch:
            mood = "Happy"
            cognitive_load = 0.20
        elif accel_mag > 4.0:
            mood = "Alert"
            cognitive_load = 0.50
        else:
            mood = "Tired" if (time.localtime().tm_hour >= 23 or time.localtime().tm_hour < 6) else "Neutral"
            cognitive_load = 0.30

        return {
            "person_id": self.person_id,
            "mood": mood,
            "cognitive_load": cognitive_load,
            "cliff_danger": any(c == 1 for c in self.cliff_states),
            "is_petted": self.is_petted,
            "shake_intensity": self.imu_shake_intensity
        }

class HumanAdaptiveCompanionBrain:
    def __init__(self, memory_db: str = "stage3_ai/memory/companion_memory.db") -> None:
        self.memory = CompanionMemory(memory_db)
        self.perception = MultimodalPerception()
        self.bond_level = 78.5
        self.mode = "Body-Doubling Silent Presence"
        self.active_curriculum_stage = "Stage 3: Multi-Capability Adaptive Mastery"
        self.last_gaze_target = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        self.voice_speech_buffer: List[str] = []

    def step(
        self,
        cliff_distances: List[float],
        crown_touch: bool,
        imu_accel: List[float],
        human_signals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        signals = human_signals or {}
        typing = signals.get("typing", False)
        sigh = signals.get("sigh", False)
        guest = signals.get("guest", False)
        smile = signals.get("smile", False)
        command = signals.get("command", None)

        state = self.perception.update(
            cliff_distances,
            crown_touch,
            imu_accel,
            typing_active=typing,
            sigh_detected=sigh,
            guest_active=guest
        )

        # Biometric Attachment & Bond Progression
        if crown_touch:
            self.bond_level = min(100.0, self.bond_level + 0.15)
        if smile:
            self.bond_level = min(100.0, self.bond_level + 0.25)
        if state["person_id"] == "Guest":
            # Protective confidentiality mode around guests
            self.mode = "Protective Guest Greeting"
        elif state["mood"] == "Focused":
            self.mode = "Flow-State Stealth Sentinel"
        elif state["mood"] == "Stressed":
            self.mode = "Empathetic Comfort"
        elif state["mood"] == "Happy":
            self.mode = "Playful Banter"
        elif state["mood"] == "Tired":
            self.mode = "Circadian Night Wind-Down"
        else:
            self.mode = "Body-Doubling Silent Presence"

        # Executive command parsing
        action_intent = "IDLE_SENTINEL"
        if command:
            cmd = command.lower().strip()
            if "pen" in cmd or "stylus" in cmd or "fetch" in cmd:
                action_intent = "FETCH_STYLUS"
            elif "nest" in cmd or "sleep" in cmd or "dock" in cmd:
                action_intent = "DOCK_SLEEP_NEST"
            elif "focus" in cmd:
                self.mode = "Flow-State Stealth Sentinel"
                action_intent = "PARK_BEHIND_LAPTOP"
            elif "come" in cmd or "here" in cmd:
                action_intent = "APPROACH_HUMAN"
            elif "bad" in cmd or "mug" in cmd:
                action_intent = "RETREAT_MUG_BUFFER"

        # Joint angle targets corresponding to behavioral modes
        joints = self._synthesize_gesture(action_intent, state["mood"])

        return {
            "mode": self.mode,
            "action_intent": action_intent,
            "bond_level": round(self.bond_level, 2),
            "person_id": state["person_id"],
            "human_mood": state["mood"],
            "cognitive_load": state["cognitive_load"],
            "cliff_danger": state["cliff_danger"],
            "target_joints": joints
        }

    def _synthesize_gesture(self, intent: str, mood: str) -> Dict[str, float]:
        # Degrees converted to radians:
        # [neck_yaw, neck_pitch, left_shoulder, right_shoulder, left_pincer, right_pincer, left_ear, right_ear]
        if intent == "FETCH_STYLUS":
            return {
                "neck_yaw": 0.0,
                "neck_pitch": -0.35,
                "left_shoulder": 1.45,
                "right_shoulder": 1.45,
                "left_pincer": 0.75,
                "right_pincer": 0.75,
                "left_ear": 0.20,
                "right_ear": 0.20
            }
        elif intent == "DOCK_SLEEP_NEST":
            return {
                "neck_yaw": 0.0,
                "neck_pitch": 0.25,
                "left_shoulder": 0.0,
                "right_shoulder": 0.0,
                "left_pincer": 0.0,
                "right_pincer": 0.0,
                "left_ear": -0.50,
                "right_ear": -0.50
            }
        elif mood == "Focused":
            # Flow-State Stealth Sentinel: low profile, zero movement
            return {
                "neck_yaw": -0.20,
                "neck_pitch": 0.10,
                "left_shoulder": 0.0,
                "right_shoulder": 0.0,
                "left_pincer": 0.0,
                "right_pincer": 0.0,
                "left_ear": 0.0,
                "right_ear": 0.0
            }
        elif mood == "Stressed":
            # Empathetic Comfort: gentle tilted head, attentive ears
            return {
                "neck_yaw": 0.25,
                "neck_pitch": 0.20,
                "left_shoulder": 0.35,
                "right_shoulder": 0.10,
                "left_pincer": 0.0,
                "right_pincer": 0.0,
                "left_ear": 0.40,
                "right_ear": 0.15
            }
        elif mood == "Happy":
            # Playful Banter: upright posture, animated wave with left arm
            return {
                "neck_yaw": 0.15,
                "neck_pitch": 0.30,
                "left_shoulder": 1.85,
                "right_shoulder": 0.15,
                "left_pincer": 0.45,
                "right_pincer": 0.0,
                "left_ear": 0.50,
                "right_ear": 0.50
            }
        else:
            # Neutral / Body-Doubling: calm presence
            return {
                "neck_yaw": 0.0,
                "neck_pitch": 0.0,
                "left_shoulder": 0.0,
                "right_shoulder": 0.0,
                "left_pincer": 0.0,
                "right_pincer": 0.0,
                "left_ear": 0.0,
                "right_ear": 0.0
            }

    def voice_duplex_hook(self, speech_text: str) -> Dict[str, Any]:
        syllable_count = max(1, len(speech_text.split()))
        nod_amplitude = min(0.35, syllable_count * 0.05)
        eye_expression = "happy" if self.bond_level > 70 else "neutral"
        return {
            "synthesized_response": f"Companion acknowleded: {speech_text}",
            "syllables": syllable_count,
            "neck_nod_amplitude": nod_amplitude,
            "eye_expression": eye_expression
        }
