from __future__ import annotations
import json
import math
import os
import sqlite3
import threading
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

    def remember_fact(self, fact_text: str, category: str = "user_memory") -> None:
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO milestones (timestamp, category, description, details_json)
            VALUES (?, ?, ?, ?)
        """, (time.time(), category, fact_text, json.dumps({"stored_by": "Owner", "active": True})))
        self.conn.commit()

    def get_recent_facts(self, limit: int = 5) -> List[str]:
        c = self.conn.cursor()
        c.execute("SELECT description FROM milestones WHERE category = 'user_memory' ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [row[0] for row in c.fetchall()]

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

class ConversationalEngine:
    def __init__(self, memory: CompanionMemory) -> None:
        self.memory = memory
        self.owner_name = "Abhisheik"
        self.tts_engine = None
        self._init_tts()

    def _init_tts(self) -> None:
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", 175)
            self.tts_engine.setProperty("volume", 0.9)
        except Exception:
            self.tts_engine = None

    def speak_async(self, text: str) -> None:
        if self.tts_engine:
            def _speak():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except Exception:
                    pass
            threading.Thread(target=_speak, daemon=True).start()

    def generate_reply(
        self,
        prompt: str,
        mode: str,
        bond_level: float,
        mood: str,
        person_id: str
    ) -> str:
        p = prompt.strip().lower()

        # Check memory fact storage command
        if "remember" in p or "don't forget" in p:
            fact = prompt
            for prefix in ["remember that", "remember to", "remember", "don't forget that", "don't forget"]:
                if p.startswith(prefix):
                    fact = prompt[len(prefix):].strip(" :,-")
                    break
            self.memory.remember_fact(fact)
            return f"Got it locked into episodic memory, {self.owner_name}: '{fact}'. I won't let you forget!"

        # Check greetings
        if any(w in p for w in ["helo", "hello", "hi", "hey", "sup", "good morning", "good evening"]):
            if person_id == "Guest":
                return "Hello visitor! I am Abhisheik's autonomous desk companion. System files and schematics are locked in guest-safe mode."
            if bond_level > 80:
                return f"Hey {self.owner_name}! Telemetry is running nominal, sensors are green, and I'm ready to assist. What are we building today?"
            return f"Hello {self.owner_name}. 8-DOF kinematic rig and perception models initialized. How can I help?"

        # Check focus mode
        if "focus" in p or "study" in p or "work" in p:
            return f"Flow-State Stealth Sentinel activated. Parking behind your display, tucking arms, and silencing notifications. Lock in, {self.owner_name}!"

        # Check pen / stylus fetch
        if "pen" in p or "stylus" in p or "fetch" in p:
            return "Deploying dual TPU pincers at coordinate (0.00, 0.06). Stylus locked in grasp—handing it over now!"

        # Check sleep nest / dock
        if "sleep" in p or "dock" in p or "nest" in p:
            return "Navigating to Corner Sleep-Nest docking pad. Lowering ears and dimming OLED eyes for circadian rest."

        # Check approach / come here
        if "come" in p or "here" in p or "approach" in p:
            return f"Rolling forward to screen-side station. Keeping safe 12 cm buffer from your mug. Standing by your side, {self.owner_name}!"

        # Check bad robot / mug
        if "bad" in p or "mug" in p or "careful" in p:
            return "Backing off immediately! Re-anchoring 12.0 cm safety perimeter around your coffee mug. No spills on my watch!"

        # Check capabilities
        if "what can you do" in p or "capabilities" in p or "help" in p:
            return "I run 15 capabilities: 12 cm semantic mug avoidance, cliff braking, stylus pickup/handover, sleep-nest docking, 8-DOF expressive gestures, affective prosody recognition, and biometric bond progression!"

        # Check memory recall
        if "what do you remember" in p or "my notes" in p or "demo" in p:
            facts = self.memory.get_recent_facts()
            if facts:
                return f"Here is what's in my memory log: '{facts[0]}'. Let me know if you need to review earlier milestones!"
            return "My episodic database is online. Tell me to 'remember [fact]' anytime to store milestones!"

        # Default contextual in-character response
        recent_facts = self.memory.get_recent_facts(limit=1)
        context_hint = f" (remembering: {recent_facts[0]})" if recent_facts else ""
        if mood == "Stressed":
            return f"I hear the stress in your cadence, {self.owner_name}. Take a quick breath—your code and circuits will come together. I'm right here beside you."
        elif mood == "Happy":
            return f"Love the energy, {self.owner_name}! Ears perked and ready for our next milestone{context_hint}."
        else:
            return f"Understood, {self.owner_name}. Executing active {mode} protocol{context_hint}."

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

        if guest_active:
            self.person_id = "Guest"

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
            mood = "Neutral"
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
        self.chat_engine = ConversationalEngine(self.memory)
        self.bond_level = 78.5
        self.mode = "Body-Doubling Silent Presence"
        self.action_intent = "IDLE_SENTINEL"
        self.person_id = "Owner"
        self.human_mood = "Neutral"
        self.dismiss_cooldown_until = 0.0
        self.active_curriculum_stage = "Stage 3: Multi-Capability Adaptive Mastery"

    def step(
        self,
        cliff_distances: List[float],
        crown_touch: bool,
        imu_accel: List[float],
        human_signals: Optional[Dict[str, Any]] = None,
        rl_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        signals = human_signals or {}
        typing = signals.get("typing", False)
        sigh = signals.get("sigh", False)
        guest = signals.get("guest", False)
        smile = signals.get("smile", False)
        command = signals.get("command", None)

        now = time.time()

        # Update perception state
        state = self.perception.update(
            cliff_distances,
            crown_touch,
            imu_accel,
            typing_active=typing,
            sigh_detected=sigh,
            guest_active=guest
        )

        # Biometric Attachment progression
        if crown_touch:
            self.bond_level = min(100.0, self.bond_level + 0.15)
            self.human_mood = "Happy"
        if smile:
            self.bond_level = min(100.0, self.bond_level + 0.25)
            self.human_mood = "Happy"
        if sigh:
            self.human_mood = "Stressed"
        if typing:
            self.human_mood = "Focused"

        # Explicit Command & Event Priority Logic
        if state["cliff_danger"]:
            self.mode = "Emergency Cliff Brake"
            self.action_intent = "EMERGENCY_REVERSE"
            self.human_mood = "Alert"
        elif guest or signals.get("guest", False):
            self.person_id = "Guest"
            self.mode = "Protective Guest Greeting"
            self.action_intent = "GUARD_CONFIDENTIALITY"
            self.human_mood = "Alert"
        elif crown_touch or smile:
            self.person_id = "Owner"
            self.mode = "Playful Banter Routine"
            self.action_intent = "PURR_AND_WAVE"
            self.human_mood = "Happy"
        elif sigh:
            self.person_id = "Owner"
            self.mode = "Empathetic Comfort Mode"
            self.action_intent = "EMPATHETIC_TILT"
            self.human_mood = "Stressed"
        elif command:
            cmd = command.lower().strip()
            self.person_id = "Owner"
            if "dismiss" in cmd or "not now" in cmd:
                self.dismiss_cooldown_until = now + 60.0
                self.mode = "Flow-State Stealth Sentinel"
                self.action_intent = "PARK_BEHIND_LAPTOP"
                self.human_mood = "Focused"
            elif "focus" in cmd or "study" in cmd or "work" in cmd:
                self.mode = "Flow-State Stealth Sentinel"
                self.action_intent = "PARK_BEHIND_LAPTOP"
                self.human_mood = "Focused"
            elif "fetch" in cmd or "pen" in cmd or "stylus" in cmd:
                self.mode = "Task Assistance (Stylus Fetch)"
                self.action_intent = "FETCH_STYLUS"
                self.human_mood = "Focused"
            elif "sleep" in cmd or "nest" in cmd or "dock" in cmd:
                self.mode = "Circadian Night Wind-Down"
                self.action_intent = "DOCK_SLEEP_NEST"
                self.human_mood = "Tired"
            elif "come" in cmd or "here" in cmd or "approach" in cmd:
                self.mode = "Body-Doubling Silent Presence"
                self.action_intent = "APPROACH_HUMAN"
                self.human_mood = "Neutral"
            elif "bad" in cmd or "mug" in cmd:
                self.mode = "Hazard Keep-Out Retraction"
                self.action_intent = "RETREAT_MUG_BUFFER"
                self.human_mood = "Alert"
            elif "play" in cmd or "dance" in cmd or "wave" in cmd:
                self.mode = "Playful Banter Routine"
                self.action_intent = "PURR_AND_WAVE"
                self.human_mood = "Happy"
        elif now < self.dismiss_cooldown_until:
            self.mode = "Flow-State Stealth Sentinel"
            self.action_intent = "PARK_BEHIND_LAPTOP"

        # Compute Kinematic 8-DOF Pose in Degrees
        joints_deg = self._synthesize_gesture_deg(self.action_intent, self.human_mood, rl_model)

        return {
            "mode": self.mode,
            "action_intent": self.action_intent,
            "bond_level": round(self.bond_level, 2),
            "person_id": self.person_id,
            "human_mood": self.human_mood,
            "cognitive_load": state["cognitive_load"],
            "cliff_danger": state["cliff_danger"],
            "target_joints_deg": joints_deg,
            "target_joints_rad": {k: math.radians(v) for k, v in joints_deg.items()}
        }

    def _synthesize_gesture_deg(self, intent: str, mood: str, rl_model: Optional[Any] = None) -> Dict[str, float]:
        # Exact kinematic angle targets in degrees (°)
        # [neck_yaw, neck_pitch, left_ear, right_ear, left_arm, right_arm, left_pincer, right_pincer]
        if intent == "FETCH_STYLUS":
            base = {
                "neck_yaw": 0.0,
                "neck_pitch": -22.5,
                "left_ear": 15.0,
                "right_ear": 15.0,
                "left_arm": 85.0,
                "right_arm": 85.0,
                "left_pincer": 45.0,
                "right_pincer": 45.0
            }
        elif intent == "PURR_AND_WAVE":
            base = {
                "neck_yaw": 15.0,
                "neck_pitch": 22.0,
                "left_ear": 35.0,
                "right_ear": 35.0,
                "left_arm": 115.0,
                "right_arm": 10.0,
                "left_pincer": 30.0,
                "right_pincer": 0.0
            }
        elif intent == "PARK_BEHIND_LAPTOP":
            base = {
                "neck_yaw": -12.0,
                "neck_pitch": 5.0,
                "left_ear": 0.0,
                "right_ear": 0.0,
                "left_arm": 0.0,
                "right_arm": 0.0,
                "left_pincer": 0.0,
                "right_pincer": 0.0
            }
        elif intent == "DOCK_SLEEP_NEST":
            base = {
                "neck_yaw": 0.0,
                "neck_pitch": 15.0,
                "left_ear": -35.0,
                "right_ear": -35.0,
                "left_arm": 0.0,
                "right_arm": 0.0,
                "left_pincer": 0.0,
                "right_pincer": 0.0
            }
        elif intent == "GUARD_CONFIDENTIALITY":
            base = {
                "neck_yaw": 22.0,
                "neck_pitch": 12.0,
                "left_ear": 25.0,
                "right_ear": 25.0,
                "left_arm": 20.0,
                "right_arm": 20.0,
                "left_pincer": 0.0,
                "right_pincer": 0.0
            }
        elif intent == "EMPATHETIC_TILT":
            base = {
                "neck_yaw": 25.0,
                "neck_pitch": 18.0,
                "left_ear": 30.0,
                "right_ear": 10.0,
                "left_arm": 25.0,
                "right_arm": 10.0,
                "left_pincer": 0.0,
                "right_pincer": 0.0
            }
        elif intent == "EMERGENCY_REVERSE":
            base = {
                "neck_yaw": 0.0,
                "neck_pitch": -28.0,
                "left_ear": -45.0,
                "right_ear": -45.0,
                "left_arm": 45.0,
                "right_arm": 45.0,
                "left_pincer": 50.0,
                "right_pincer": 50.0
            }
        elif intent == "APPROACH_HUMAN":
            base = {
                "neck_yaw": 0.0,
                "neck_pitch": 16.0,
                "left_ear": 25.0,
                "right_ear": 25.0,
                "left_arm": 45.0,
                "right_arm": 15.0,
                "left_pincer": 0.0,
                "right_pincer": 0.0
            }
        else:
            base = {
                "neck_yaw": 0.0,
                "neck_pitch": 8.0,
                "left_ear": 10.0,
                "right_ear": 10.0,
                "left_arm": 5.0,
                "right_arm": 5.0,
                "left_pincer": 0.0,
                "right_pincer": 0.0
            }

        # Fine-tune with RL policy if loaded
        if rl_model is not None:
            try:
                obs_dummy = np.zeros(36, dtype=np.float32)
                action, _ = rl_model.predict(obs_dummy, deterministic=True)
                # Action indices 2..9 correspond to 8 servos
                base["neck_yaw"] += float(action[2] * 5.0)
                base["neck_pitch"] += float(action[3] * 5.0)
            except Exception:
                pass

        return base
