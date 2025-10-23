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

class SemanticMemoryStorage(BaseChromaStorage):
    def __init__(self, chromadb_path: Optional[str] = None):
        super().__init__(chromadb_path)
        self.collection = self.get_or_create_collection(name="semantic_memory")

    async def add(self, item_id: str, content: str, metadata: Dict[str, Any]):
        self.collection.add(ids=[item_id], documents=[content], metadatas=[metadata])

    async def search(self, user_id: str, query: str, limit: int = 10, knowledge_type: Optional[str] = None, category: Optional[str] = None) -> Dict[str, Any]:
        from loguru import logger
        logger.info(f"SemanticMemoryStorage.search: query='{query}', user_id='{user_id}', limit={limit}")
        logger.info(f"self.collection: {self.collection}")
        logger.info(f"self.collection.name: {self.collection.name}")
        logger.info(f"self.collection.count(): {self.collection.count()}")
        
        conditions: List[Dict[str, Any]] = [{"user_id": user_id}]
        if knowledge_type:
            conditions.append({"knowledge_type": knowledge_type})
        if category:
            conditions.append({"category": category})
        where_filter: Dict[str, Any] = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        logger.info(f"where_filter: {where_filter}")
        
        # Используем query_embeddings для правильного вычисления distances
        # Получаем embedding для запроса через embedding function
        try:
            # Пытаемся получить embedding function из коллекции
            logger.info(f"DEBUG: collection has _embedding_function: {hasattr(self.collection, '_embedding_function')}")
            if hasattr(self.collection, '_embedding_function'):
                logger.info(f"DEBUG: _embedding_function is not None: {self.collection._embedding_function is not None}")
            
            if hasattr(self.collection, '_embedding_function') and self.collection._embedding_function:
                logger.info("DEBUG: Using query_embeddings with collection embedding function")
                query_embedding = self.collection._embedding_function([query])[0]
                result = self.collection.query(query_embeddings=[query_embedding], where=where_filter, n_results=limit, include=["documents", "metadatas", "distances"])  # raw
            else:
                logger.info("DEBUG: Using query_texts fallback")
                # Fallback к query_texts если embedding function недоступен
                result = self.collection.query(query_texts=[query], where=where_filter, n_results=limit, include=["documents", "metadatas", "distances"])  # raw
        except Exception as e:
            logger.warning(f"Failed to use query_embeddings, falling back to query_texts: {e}")
            result = self.collection.query(query_texts=[query], where=where_filter, n_results=limit, include=["documents", "metadatas", "distances"])  # raw
        logger.info(f"Query result: {result}")
        return result

    async def get_by_category(self, user_id: str, category: str, limit: int = 50) -> Dict[str, Any]:
        return self.collection.get(where={"$and": [{"user_id": user_id}, {"category": category}]}, limit=limit, include=["documents", "metadatas"])  # raw

    async def get_all(self, user_id: str, limit: int = 2000) -> Dict[str, Any]:
        return self.collection.get(where={"user_id": user_id}, limit=limit, include=["documents", "metadatas"])  # raw

    async def older_than_days(self, user_id: str, ttl_days: int, importance_threshold: float = 0.3, min_access_count: int = 2, limit: int = 5000) -> List[str]:
        cutoff = datetime.now() - timedelta(days=ttl_days)
        res = self.collection.get(where={"user_id": user_id}, limit=limit, include=["metadatas"])  # raw
        ids: List[str] = []
        if res.get("ids"):
            for i, _id in enumerate(res["ids"]):
                meta = res["metadatas"][i] or {}
                ts = meta.get("timestamp")
                if not ts:
                    continue
                try:
                    old_enough = datetime.fromisoformat(ts) < cutoff
                except Exception:
                    continue
                importance = float(meta.get("importance", 0.5))
                access_count = int(meta.get("access_count", 0))
                if old_enough and importance < importance_threshold and access_count < min_access_count:
                    ids.append(_id)
        return ids

    async def delete_many(self, ids: List[str]):
        if ids:
            self.collection.delete(ids=ids)
