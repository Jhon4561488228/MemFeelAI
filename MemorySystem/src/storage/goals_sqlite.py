"""
SQLite storage for procedural goals as per architecture plan.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import os

@dataclass
class Goal:
    id: str
    user_id: str
    name: str
    description: str
    status: str
    progress: float
    next_run: Optional[datetime]

class GoalsSQLiteStorage:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_root = os.getenv("AIRI_DATA_DIR", "./data")
            db_path = os.getenv("SQLITE_DB", str(Path(data_root) / "memory_system.db"))
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    trigger_type TEXT,
                    trigger_value TEXT,
                    action TEXT,
                    status TEXT,
                    progress REAL DEFAULT 0.0,
                    next_run DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id)")

    def add_goal(self, goal_id: str, user_id: str, name: str, description: str, next_run: Optional[datetime]) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO goals(id, user_id, trigger_type, trigger_value, action, status, progress, next_run) VALUES(?,?,?,?,?,?,?,?)",
                (
                    goal_id,
                    user_id,
                    "manual",
                    name,
                    description,
                    "active",
                    0.0,
                    (next_run.isoformat() if next_run else None),
                ),
            )

    def get_active_goals(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT id, trigger_value, action, status, progress FROM goals WHERE user_id = ? AND status = 'active' ORDER BY next_run ASC LIMIT ?",
                (user_id, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "status": row[3],
                "progress": row[4],
            }
            for row in rows
        ]

    def update_goal(self, goal_id: str, progress: float, status: Optional[str]) -> None:
        with self._conn() as conn:
            if status is not None:
                conn.execute(
                    "UPDATE goals SET progress = ?, status = ?, next_run = ? WHERE id = ?",
                    (max(0.0, min(1.0, progress)), status, datetime.now().isoformat(), goal_id),
                )
            else:
                conn.execute(
                    "UPDATE goals SET progress = ?, next_run = ? WHERE id = ?",
                    (max(0.0, min(1.0, progress)), datetime.now().isoformat(), goal_id),
                )

    def delete_completed_goals(self, user_id: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(1) FROM goals WHERE user_id = ? AND status = 'completed'",
                (user_id,),
            )
            cnt = cur.fetchone()[0]
            conn.execute("DELETE FROM goals WHERE user_id = ? AND status = 'completed'", (user_id,))
        return int(cnt)

    def count(self) -> int:
        with self._conn() as conn:
            cur = conn.execute("SELECT COUNT(1) FROM goals")
            return int(cur.fetchone()[0])
