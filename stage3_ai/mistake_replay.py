from __future__ import annotations
import json
import os
import sqlite3
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

MISTAKE_TYPES = [
    "CLIFF_TRIGGER",
    "MUG_PROXIMITY_VIOLATION",
    "WHEEL_SKID_PAPER",
    "DROPPED_STYLUS",
    "USER_CORRECTION_STAY_BACK"
]

class HardSafetyShield:
    @staticmethod
    def filter_action(obs: np.ndarray, action: np.ndarray, mug_center: np.ndarray) -> Tuple[np.ndarray, bool]:
        safe_act = action.copy()
        intervened = False

        cliff_readings = obs[0:4]
        rx = float(obs[28]) if len(obs) > 28 else 0.0
        ry = float(obs[29]) if len(obs) > 29 else 0.0

        # Pre-cliff check (<2ms reflex)
        front_cliff_danger = any(c > 0.038 for c in cliff_readings[:2]) or abs(rx) > 0.52 or abs(ry) > 0.24
        if front_cliff_danger and (safe_act[0] + safe_act[1]) > 0.0:
            safe_act[0] = -6.0
            safe_act[1] = -6.0
            intervened = True

        # 12 cm Mug keep-out shield
        dist_mug = float(np.sqrt((rx - mug_center[0])**2 + (ry - mug_center[1])**2))
        if dist_mug < 0.13:
            mug_vec = np.array([mug_center[0] - rx, mug_center[1] - ry])
            cross_z = rx * mug_vec[1] - ry * mug_vec[0]
            steer = 6.0 if cross_z > 0 else -6.0
            safe_act[0] = -3.0 + steer
            safe_act[1] = -3.0 - steer
            intervened = True

        return safe_act, intervened

class MistakeReplayBuffer:
    def __init__(self, db_path: str = "stage3_ai/memory/mistakes.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.safety_shield = HardSafetyShield()
        self._init_db()

    def _init_db(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mistakes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                mistake_type TEXT NOT NULL,
                state_json TEXT NOT NULL,
                action_json TEXT NOT NULL,
                penalty REAL NOT NULL,
                priority REAL NOT NULL,
                resolution_state_json TEXT,
                corrected_count INTEGER DEFAULT 0
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_priority ON mistakes(priority DESC)
        """)
        self.conn.commit()

    def record_mistake(
        self,
        mistake_type: str,
        state: np.ndarray | List[float],
        action: np.ndarray | List[float],
        penalty: float,
        priority: float = 1.0,
        resolution_state: Optional[np.ndarray | List[float]] = None
    ) -> int:
        state_list = state.tolist() if isinstance(state, np.ndarray) else list(state)
        action_list = action.tolist() if isinstance(action, np.ndarray) else list(action)
        res_list = resolution_state.tolist() if isinstance(resolution_state, np.ndarray) else (list(resolution_state) if resolution_state is not None else None)

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO mistakes (timestamp, mistake_type, state_json, action_json, penalty, priority, resolution_state_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            mistake_type,
            json.dumps(state_list),
            json.dumps(action_list),
            penalty,
            priority,
            json.dumps(res_list) if res_list else None
        ))
        self.conn.commit()
        return cursor.lastrowid

    def sample_corrective_batch(self, batch_size: int = 16) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, mistake_type, state_json, action_json, penalty, priority, corrected_count
            FROM mistakes
            ORDER BY priority DESC, timestamp DESC
            LIMIT ?
        """, (batch_size,))
        rows = cursor.fetchall()
        batch = []
        for row in rows:
            mid, mtype, s_json, a_json, pen, prio, c_count = row
            batch.append({
                "id": mid,
                "mistake_type": mtype,
                "state": np.array(json.loads(s_json), dtype=np.float32),
                "action": np.array(json.loads(a_json), dtype=np.float32),
                "penalty": pen,
                "priority": prio,
                "corrected_count": c_count
            })
            cursor.execute("UPDATE mistakes SET corrected_count = corrected_count + 1 WHERE id = ?", (mid,))
        self.conn.commit()
        return batch

    def sample_balanced_batch(self, batch_size: int = 16) -> List[Dict]:
        mistake_k = max(1, int(batch_size * 0.30))
        mistakes = self.sample_corrective_batch(batch_size=mistake_k)
        return mistakes

    def decay_priority(self, mistake_id: int, decay_factor: float = 0.5) -> None:
        cursor = self.conn.cursor()
        cursor.execute("UPDATE mistakes SET priority = priority * ? WHERE id = ?", (decay_factor, mistake_id))
        self.conn.commit()

    def get_mistake_summary(self) -> Dict[str, int]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT mistake_type, COUNT(*) FROM mistakes GROUP BY mistake_type")
        return {mtype: count for mtype, count in cursor.fetchall()}

    def close(self) -> None:
        self.conn.close()
