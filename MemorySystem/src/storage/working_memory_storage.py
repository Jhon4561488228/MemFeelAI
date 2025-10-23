from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from chromadb import PersistentClient
from chromadb.config import Settings
import os
try:
    from .base_chroma import BaseChromaStorage
except Exception:  # pragma: no cover
    from storage.base_chroma import BaseChromaStorage

class WorkingMemoryStorage(BaseChromaStorage):
    def __init__(self, chromadb_path: Optional[str] = None):
        super().__init__(chromadb_path)
        self.collection = self.get_or_create_collection(name="working_memory")

    async def add(self, item_id: str, content: str, metadata: Dict[str, Any]):
        try:
            self.collection.add(ids=[item_id], documents=[content], metadatas=[metadata])
        except Exception as e:
            raise

    async def recent(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        res = self.collection.get(where={"user_id": user_id}, limit=limit, include=["documents", "metadatas"])
        items: List[Dict[str, Any]] = []
        if res.get("ids"):
            for i, _id in enumerate(res["ids"]):
                items.append({
                    "id": _id,
                    "content": res["documents"][i],
                    "metadata": res["metadatas"][i],
                })
        # sort by timestamp desc
        items.sort(key=lambda x: x["metadata"].get("timestamp", x["metadata"].get("created_at", "")), reverse=True)
        return items[:limit]

    async def delete_many(self, ids: List[str]):
        if ids:
            self.collection.delete(ids=ids)

    async def older_than(self, user_id: str, ttl_hours: int) -> List[str]:
        cutoff = datetime.now() - timedelta(hours=ttl_hours)
        res = self.collection.get(where={"user_id": user_id}, include=["metadatas"])
        ids: List[str] = []
        if res.get("ids"):
            for i, _id in enumerate(res["ids"]):
                meta = res["metadatas"][i] or {}
                ts = meta.get("timestamp") or meta.get("created_at")
                if not ts:
                    continue
                try:
                    if datetime.fromisoformat(ts) < cutoff:
                        ids.append(_id)
                except Exception:
                    continue
        return ids
