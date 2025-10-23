"""
Working Memory Manager для AIRI Memory System
Уровень 1: Рабочая память - активный контекст (последние 10 сообщений, TTL = 1 час)
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

try:
    from ..storage.working_memory_storage import WorkingMemoryStorage  # when imported as package
except Exception:  # pragma: no cover
    from storage.working_memory_storage import WorkingMemoryStorage  # when running tests with src on sys.path

logger = logging.getLogger(__name__)

@dataclass
class WorkingMemoryItem:
    """Элемент рабочей памяти"""
    id: str
    content: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    context: Optional[str] = None
    emotion_data: Optional[Dict[str, Any]] = None

class WorkingMemoryManager:
    """Менеджер рабочей памяти"""
    
    def __init__(self, chromadb_path: str = "./data/chroma_db"):
        self.chromadb_path = chromadb_path
        # Хранилище рабочего уровня
        self._storage = WorkingMemoryStorage(chromadb_path=chromadb_path)
        # Совместимость со старым кодом (поиск/обновление метаданных)
        self.collection = self._storage.collection
        self.max_items = 10  # Максимум элементов в рабочей памяти
        self.ttl_hours = 1   # Время жизни элемента
        
    async def add_memory(self, content: str, user_id: str, 
                        importance: float = 0.5, confidence: float = 0.5, context: Optional[str] = None,
                        emotion_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Добавить элемент в рабочую память
        
        Args:
            content: Содержимое памяти
            user_id: ID пользователя
            importance: Важность (0.0-1.0)
            context: Контекст
            emotion_data: Эмоциональные данные
            
        Returns:
            ID созданного элемента
        """
        try:
            memory_id = f"wm_{uuid.uuid4()}"
            timestamp = datetime.now()
            
            # Создаем метаданные
            metadata = {
                "user_id": user_id,
                "timestamp": timestamp.isoformat(),
                "importance": importance,
                "confidence": confidence,
                "context": context or "",
                "emotion_data": __import__("json").dumps(emotion_data) if emotion_data else "{}",
                "memory_type": "working"
            }
            
            # Добавляем через storage
            await self._storage.add(memory_id, content, metadata)
            
            # Очищаем старые элементы
            await self._cleanup_old_memories(user_id)
            
            logger.info(f"Added working memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error adding working memory: {e}")
            raise
    
    async def get_active_context(self, user_id: str, limit: int = 10) -> List[WorkingMemoryItem]:
        """
        Получить активный контекст пользователя
        
        Args:
            user_id: ID пользователя
            limit: Максимальное количество элементов
            
        Returns:
            Список элементов рабочей памяти
        """
        try:
            # Получаем последние элементы через storage
            raw_items = await self._storage.recent(user_id=user_id, limit=limit)
            items: List[WorkingMemoryItem] = []
            for it in raw_items:
                meta = it["metadata"]
                items.append(WorkingMemoryItem(
                    id=it["id"],
                    content=it["content"],
                    user_id=meta.get("user_id", user_id),
                    timestamp=datetime.fromisoformat(meta.get("timestamp")),
                    importance=meta.get("importance", 0.5),
                    context=meta.get("context"),
                    emotion_data=__import__("json").loads(meta.get("emotion_data", "{}")) if meta.get("emotion_data") else None
                ))
            return items
            
        except Exception as e:
            logger.error(f"Error getting active context: {e}")
            return []
    
    async def search_context(self, user_id: str, query: str, limit: int = 5) -> List[WorkingMemoryItem]:
        """
        Поиск в рабочей памяти
        
        Args:
            user_id: ID пользователя
            query: Поисковый запрос
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных элементов
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                where={"user_id": user_id},
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            items = []
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                # Фильтруем по релевантности
                if distance < 0.8:  # Порог релевантности
                    item = WorkingMemoryItem(
                        id=results["ids"][0][i],
                        content=doc,
                        user_id=metadata["user_id"],
                        timestamp=datetime.fromisoformat(metadata["timestamp"]),
                        importance=metadata.get("importance", 0.5),
                        context=metadata.get("context"),
                        emotion_data=__import__("json").loads(metadata.get("emotion_data", "{}")) if metadata.get("emotion_data") else None
                    )
                    items.append(item)
            
            return items
            
        except Exception as e:
            logger.error(f"Error searching context: {e}")
            return []
    
    async def update_importance(self, memory_id: str, importance: float) -> bool:
        """
        Обновить важность элемента
        
        Args:
            memory_id: ID элемента
            importance: Новая важность (0.0-1.0)
            
        Returns:
            True если успешно
        """
        try:
            # Получаем текущие метаданные
            results = self.collection.get(ids=[memory_id], include=["metadatas"])
            if not results["metadatas"]:
                return False
            
            metadata = results["metadatas"][0]
            metadata["importance"] = importance
            
            # Обновляем метаданные
            self.collection.update(
                ids=[memory_id],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated importance for {memory_id}: {importance}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating importance: {e}")
            return False
    
    async def remove_memory(self, memory_id: str) -> bool:
        """
        Удалить элемент из рабочей памяти
        
        Args:
            memory_id: ID элемента
            
        Returns:
            True если успешно
        """
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Removed working memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing memory: {e}")
            return False
    
    async def _cleanup_old_memories(self, user_id: str):
        """Очистка старых элементов рабочей памяти"""
        try:
            # Удаляем старые (TTL)
            old_ids = await self._storage.older_than(user_id=user_id, ttl_hours=self.ttl_hours)
            if old_ids:
                await self._storage.delete_many(old_ids)
                logger.info(f"Cleaned up {len(old_ids)} old working memories")

            # Ограничиваем количество элементов
            recent_items = await self._storage.recent(user_id=user_id, limit=1000)
            if len(recent_items) > self.max_items:
                # Самые старые к удалению
                excess = recent_items[self.max_items:]
                to_remove = [it["id"] for it in excess]
                if to_remove:
                    await self._storage.delete_many(to_remove)
                    logger.info(f"Removed {len(to_remove)} excess working memories")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")
    
    async def get_older_than(self, user_id: str, delta: timedelta, limit: int = 100) -> List[WorkingMemoryItem]:
        """Вернуть элементы старше указанного дельта-времени."""
        try:
            cutoff = datetime.now() - delta
            results = self.collection.get(
                where={"user_id": user_id},
                limit=limit,
                include=["documents", "metadatas"]
            )
            items: List[WorkingMemoryItem] = []
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i]
                ts = datetime.fromisoformat(metadata["timestamp"])
                if ts <= cutoff:
                    items.append(WorkingMemoryItem(
                        id=results["ids"][i],
                        content=doc,
                        user_id=metadata["user_id"],
                        timestamp=ts,
                        importance=metadata.get("importance", 0.5),
                        context=metadata.get("context"),
                        emotion_data=__import__("json").loads(metadata.get("emotion_data", "{}")) if metadata.get("emotion_data") else None
                    ))
            return items
        except Exception as e:
            logger.error(f"Error getting older working memories: {e}")
            return []
    
    async def bulk_delete(self, ids: List[str]) -> int:
        """Удалить элементы по списку ID."""
        if not ids:
            return 0
        try:
            self.collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            logger.error(f"Error bulk deleting working memories: {e}")
            return 0
    
    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Получить статистику рабочей памяти
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Словарь со статистикой
        """
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            if not results["metadatas"]:
                return {
                    "total_items": 0,
                    "avg_importance": 0.0,
                    "oldest_item": None,
                    "newest_item": None
                }
            
            importances = [m.get("importance", 0.5) for m in results["metadatas"]]
            timestamps = [datetime.fromisoformat(m["timestamp"]) for m in results["metadatas"]]
            
            return {
                "total_items": len(results["metadatas"]),
                "avg_importance": sum(importances) / len(importances),
                "oldest_item": min(timestamps).isoformat(),
                "newest_item": max(timestamps).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    async def get_memory_by_id(self, memory_id: str, user_id: str) -> Optional[WorkingMemoryItem]:
        """
        Получить элемент рабочей памяти по ID
        
        Args:
            memory_id: ID элемента памяти
            user_id: ID пользователя
            
        Returns:
            WorkingMemoryItem или None если не найден
        """
        try:
            results = self.collection.get(
                ids=[memory_id],
                where={"user_id": user_id},
                include=["documents", "metadatas"]
            )
            
            if not results["documents"] or not results["metadatas"]:
                return None
            
            doc = results["documents"][0]
            metadata = results["metadatas"][0]
            
            # Парсим timestamp
            ts = datetime.fromisoformat(metadata["timestamp"])
            
            return WorkingMemoryItem(
                id=memory_id,
                content=doc,
                user_id=metadata["user_id"],
                timestamp=ts,
                importance=metadata.get("importance", 0.5),
                context=metadata.get("context"),
                emotion_data=__import__("json").loads(metadata.get("emotion_data", "{}")) if metadata.get("emotion_data") else None
            )
            
        except Exception as e:
            logger.error(f"Error getting working memory by ID {memory_id}: {e}")
            return None
    
    async def update_metadata(self, memory_id: str, user_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Обновить метаданные элемента рабочей памяти
        
        Args:
            memory_id: ID элемента памяти
            user_id: ID пользователя
            metadata: Новые метаданные
            
        Returns:
            True если успешно, False если ошибка
        """
        try:
            # Получаем текущий элемент
            current_item = await self.get_memory_by_id(memory_id, user_id)
            if not current_item:
                logger.warning(f"Working memory item {memory_id} not found for user {user_id}")
                return False
            
            # Обновляем метаданные
            updated_metadata = {
                "user_id": user_id,
                "timestamp": current_item.timestamp.isoformat(),
                "importance": current_item.importance,
                "context": current_item.context,
                "emotion_data": __import__("json").dumps(current_item.emotion_data) if current_item.emotion_data else None
            }
            
            # Добавляем новые метаданные
            updated_metadata.update(metadata)
            
            # Обновляем в ChromaDB
            self.collection.update(
                ids=[memory_id],
                metadatas=[updated_metadata]
            )
            
            logger.debug(f"Updated metadata for working memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating working memory metadata {memory_id}: {e}")
            return False
    
    async def get_memories(self, user_id: str, limit: int = 20) -> List[WorkingMemoryItem]:
        """
        Получить все воспоминания пользователя из рабочей памяти
        
        Args:
            user_id: ID пользователя
            limit: Максимальное количество элементов
            
        Returns:
            Список элементов рабочей памяти
        """
        try:
            # Получаем все элементы пользователя
            results = self.collection.get(
                where={"user_id": user_id},
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            memories = []
            if results["ids"]:
                for i, memory_id in enumerate(results["ids"]):
                    content = results["documents"][i]
                    metadata = results["metadatas"][i]
                    
                    # Создаем объект WorkingMemoryItem
                    memory_item = WorkingMemoryItem(
                        id=memory_id,
                        content=content,
                        user_id=user_id,
                        timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                        importance=metadata.get("importance", 0.5),
                        context=metadata.get("context"),
                        emotion_data=__import__("json").loads(metadata.get("emotion_data", "{}"))
                    )
                    memories.append(memory_item)
            
            logger.debug(f"Retrieved {len(memories)} working memories for user {user_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Error getting working memories for user {user_id}: {e}")
            return []
