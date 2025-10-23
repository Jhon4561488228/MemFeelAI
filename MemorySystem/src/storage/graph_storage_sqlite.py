"""
SQLite storage for graph edges as per architecture plan.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

@dataclass
class Edge:
    from_id: str
    to_id: str
    relation_type: str
    weight: float
    user_id: Optional[str]

class GraphSQLiteStorage:
    def __init__(self, db_path: str = "./data/memory_system.db"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def add_edge(self, from_id: str, to_id: str, relation_type: str, weight: float = 1.0, user_id: Optional[str] = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO memory_edges(from_id, to_id, relation_type, weight, user_id) VALUES(?,?,?,?,?)",
                (from_id, to_id, relation_type, weight, user_id),
            )

    def get_edges_for_node(self, user_id: str, node_id: str, limit: int = 1000) -> List[Edge]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT from_id, to_id, relation_type, weight, user_id FROM memory_edges WHERE user_id = ? AND (from_id = ? OR to_id = ?) LIMIT ?",
                (user_id, node_id, node_id, limit),
            )
            rows = cur.fetchall()
        return [Edge(*row) for row in rows]

    def find_path_bfs(self, user_id: str, source_id: str, target_id: str, max_depth: int = 3) -> List[str]:
        if source_id == target_id:
            return [source_id]
        from collections import deque
        queue: deque[Tuple[str, List[str]]] = deque([(source_id, [source_id])])
        visited = {source_id}
        depth_map = {source_id: 0}

        while queue:
            current, path = queue.popleft()
            depth = depth_map[current]
            if depth >= max_depth:
                continue
            for edge in self.get_edges_for_node(user_id, current, limit=1000):
                neighbor = edge.to_id if edge.from_id == current else edge.from_id
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                new_path = path + [neighbor]
                if neighbor == target_id:
                    return new_path
                depth_map[neighbor] = depth + 1
                queue.append((neighbor, new_path))
        return []
