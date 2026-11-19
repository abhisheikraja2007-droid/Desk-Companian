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

class MistakeReplayBuffer:
    def __init__(self, db_path: str = "stage3_ai/memory/mistakes.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
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

    def decay_priority(self, mistake_id: int, decay_factor: float = 0.5) -> None:
        cursor = self.conn.cursor()
        cursor.execute("UPDATE mistakes SET priority = priority * ? WHERE id = ?", (decay_factor, mistake_id))
        self.conn.commit()

    def get_mistake_counts(self) -> Dict[str, int]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT mistake_type, COUNT(*) FROM mistakes GROUP BY mistake_type")
        counts = {m: 0 for m in MISTAKE_TYPES}
        for mtype, count in cursor.fetchall():
            counts[mtype] = count
        return counts

    def get_total_mistakes(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM mistakes")
        row = cursor.fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self.conn.close()
