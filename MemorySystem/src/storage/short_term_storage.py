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

class ShortTermMemoryStorage(BaseChromaStorage):
    def __init__(self, chromadb_path: Optional[str] = None):
        super().__init__(chromadb_path)
        self.collection = self.get_or_create_collection(name="short_term_memory")

    async def add(self, item_id: str, content: str, metadata: Dict[str, Any]):
        # ChromaDB операции синхронные, но метод async для совместимости
        self.collection.add(ids=[item_id], documents=[content], metadatas=[metadata])

    async def get_all(self, user_id: str, limit: int = 200) -> Dict[str, Any]:
        return self.collection.get(where={"user_id": user_id}, limit=limit, include=["documents", "metadatas"])  # raw

    async def recent(self, user_id: str, since: datetime, limit: int = 200) -> List[Dict[str, Any]]:
        res = self.collection.get(where={"user_id": user_id}, limit=limit, include=["documents", "metadatas"])
        items: List[Dict[str, Any]] = []
        if res.get("ids"):
            for i, _id in enumerate(res["ids"]):
                meta = res["metadatas"][i] or {}
                try:
                    ts = datetime.fromisoformat(meta.get("timestamp"))
                except Exception:
                    continue
                if ts >= since:
                    items.append({
                        "id": _id,
                        "content": res["documents"][i],
                        "metadata": meta,
                    })
        items.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)
        return items

    async def search(self, user_id: str, query: str, limit: int = 10, where_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        where_filter: Dict[str, Any] = {"user_id": user_id}
        if where_extra:
            where_filter.update(where_extra)
        return self.collection.query(query_texts=[query], where=where_filter, n_results=limit, include=["documents", "metadatas", "distances"])  # raw

    async def older_than(self, user_id: str, ttl_hours: int, limit: int = 1000) -> List[str]:
        cutoff = datetime.now() - timedelta(hours=ttl_hours)
        res = self.collection.get(where={"user_id": user_id}, limit=limit, include=["metadatas"])  # raw
        ids: List[str] = []
        if res.get("ids"):
            for i, _id in enumerate(res["ids"]):
                meta = res["metadatas"][i] or {}
                ts = meta.get("timestamp")
                if not ts:
                    continue
                try:
                    if datetime.fromisoformat(ts) < cutoff:
                        ids.append(_id)
                except Exception:
                    continue
        return ids

    async def delete_many(self, ids: List[str]):
        if ids:
            self.collection.delete(ids=ids)
