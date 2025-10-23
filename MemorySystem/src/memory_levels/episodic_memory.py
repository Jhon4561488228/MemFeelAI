"""
Episodic Memory Manager для AIRI Memory System
Уровень 3: Эпизодическая память - значимые события и переживания, TTL = 30 дней
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

try:
    from ..storage.episodic_storage import EpisodicMemoryStorage
except Exception:  # pragma: no cover
    from storage.episodic_storage import EpisodicMemoryStorage

logger = logging.getLogger(__name__)

@dataclass
class EpisodicMemoryItem:
    """Элемент эпизодической памяти"""
    id: str
    content: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    event_type: str = "experience"
    location: Optional[str] = None
    participants: List[str] = None
    emotion_data: Optional[Dict[str, Any]] = None
    related_memories: List[str] = None
    significance: float = 0.5
    vividness: float = 0.5
    context: Optional[str] = None

class EpisodicMemoryManager:
    """Менеджер эпизодической памяти"""
    
    def __init__(self, chromadb_path: str = "./data/chroma_db"):
        self.chromadb_path = chromadb_path
        self._storage = EpisodicMemoryStorage(chromadb_path=chromadb_path)
        self.collection = self._storage.collection
        self.max_items = 1000  # Максимум элементов в эпизодической памяти
        self.ttl_days = 30     # Время жизни элемента (30 дней)
        
    async def add_experience(self, content: str, user_id: str, 
                           importance: float = 0.5, confidence: float = 0.5, event_type: str = "experience",
                           location: Optional[str] = None, participants: List[str] = None,
                           emotion_data: Optional[Dict[str, Any]] = None,
                           related_memories: List[str] = None,
                           significance: float = 0.5, vividness: float = 0.5,
                           context: Optional[str] = None) -> str:
        """
        Добавить опыт в эпизодическую память
        
        Args:
            content: Содержимое опыта
            user_id: ID пользователя
            importance: Важность (0.0-1.0)
            event_type: Тип события (experience, achievement, relationship, etc.)
            location: Местоположение
            participants: Участники события
            emotion_data: Эмоциональные данные
            related_memories: Связанные воспоминания
            significance: Значимость события (0.0-1.0)
            vividness: Яркость воспоминания (0.0-1.0)
            context: Контекст события
            
        Returns:
            ID созданного элемента
        """
        try:
            memory_id = f"ep_{uuid.uuid4()}"
            timestamp = datetime.now()
            
            # Создаем метаданные
            metadata = {
                "user_id": user_id,
                "timestamp": timestamp.isoformat(),
                "importance": importance,
                "confidence": confidence,
                "event_type": event_type,
                "location": location or "",
                "participants": __import__("json").dumps(participants or []),
                "emotion_data": __import__("json").dumps(emotion_data or {}),
                "related_memories": __import__("json").dumps(related_memories or []),
                "significance": significance,
                "vividness": vividness,
                "context": context or "",
                "memory_type": "episodic"
            }
            
            # Добавляем через storage
            await self._storage.add(memory_id, content, metadata)
            
            # Очищаем старые элементы
            await self._cleanup_old_memories(user_id)
            
            logger.info(f"Added episodic memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error adding episodic memory: {e}")
            raise
    
    async def get_significant_events(self, user_id: str, days: int = 30, 
                                   min_importance: float = 0.3, limit: int = 50) -> List[EpisodicMemoryItem]:
        """
        Получить значимые события
        
        Args:
            user_id: ID пользователя
            days: Количество дней назад
            min_importance: Минимальная важность
            limit: Максимальное количество элементов
            
        Returns:
            Список значимых событий
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            raw = await self._storage.recent_since(user_id=user_id, since=cutoff_time, limit=limit)
            items: List[EpisodicMemoryItem] = []
            for it in raw:
                meta = it["metadata"]
                if meta.get("importance", 0.5) < min_importance:
                    continue
                items.append(EpisodicMemoryItem(
                    id=it["id"],
                    content=it["content"],
                    user_id=meta.get("user_id", user_id),
                    timestamp=datetime.fromisoformat(meta.get("timestamp")),
                    importance=meta.get("importance", 0.5),
                    event_type=meta.get("event_type", "experience"),
                    location=meta.get("location"),
                    participants=__import__("json").loads(meta.get("participants", "[]")) if meta.get("participants") else [],
                    emotion_data=__import__("json").loads(meta.get("emotion_data", "{}")) if meta.get("emotion_data") else None,
                    related_memories=__import__("json").loads(meta.get("related_memories", "[]")) if meta.get("related_memories") else [],
                    significance=meta.get("significance", 0.5),
                    vividness=meta.get("vividness", 0.5),
                    context=meta.get("context")
                ))
            return items
            
        except Exception as e:
            logger.error(f"Error getting significant events: {e}")
            return []
    
    async def search_experiences(self, user_id: str, query: str, 
                               event_type: Optional[str] = None, 
                               days: int = 30, limit: int = 10) -> List[EpisodicMemoryItem]:
        """
        Поиск опытов в эпизодической памяти
        
        Args:
            user_id: ID пользователя
            query: Поисковый запрос
            event_type: Тип события для фильтрации
            days: Количество дней назад
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных опытов
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Строим фильтр
            where_filter = {"user_id": user_id}
            if event_type:
                where_filter["event_type"] = event_type
            
            results = await self._storage.search(user_id=user_id, query=query, limit=limit, where_extra={"event_type": event_type} if event_type else None)
            
            items = []
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                timestamp = datetime.fromisoformat(metadata["timestamp"])
                
                # Фильтруем по времени и релевантности
                if timestamp >= cutoff_time and distance < 0.8:
                    item = EpisodicMemoryItem(
                        id=results["ids"][0][i],
                        content=doc,
                        user_id=metadata["user_id"],
                        timestamp=timestamp,
                        importance=metadata.get("importance", 0.5),
                        event_type=metadata.get("event_type", "experience"),
                        location=metadata.get("location"),
                        participants=__import__("json").loads(metadata.get("participants", "[]")) if metadata.get("participants") else [],
                        emotion_data=__import__("json").loads(metadata.get("emotion_data", "{}")) if metadata.get("emotion_data") else None,
                        related_memories=__import__("json").loads(metadata.get("related_memories", "[]")) if metadata.get("related_memories") else [],
                        significance=metadata.get("significance", 0.5),
                        vividness=metadata.get("vividness", 0.5),
                        context=metadata.get("context")
                    )
                    items.append(item)
            
            return items
            
        except Exception as e:
            logger.error(f"Error searching experiences: {e}")
            return []
    
    async def get_emotional_memories(self, user_id: str, emotion: str, 
                                   days: int = 30, limit: int = 20) -> List[EpisodicMemoryItem]:
        """
        Получить воспоминания с определенной эмоцией
        
        Args:
            user_id: ID пользователя
            emotion: Эмоция для поиска
            days: Количество дней назад
            limit: Максимальное количество элементов
            
        Returns:
            Список воспоминаний с указанной эмоцией
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            results = await self._storage.get_all(user_id=user_id, limit=limit)
            
            items = []
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i]
                timestamp = datetime.fromisoformat(metadata["timestamp"])
                
                # Фильтруем по времени
                if timestamp >= cutoff_time:
                    emotion_data = __import__("json").loads(metadata.get("emotion_data", "{}")) if metadata.get("emotion_data") else {}
                    
                    # Проверяем наличие эмоции
                    if emotion in str(emotion_data).lower():
                        item = EpisodicMemoryItem(
                            id=results["ids"][i],
                            content=doc,
                            user_id=metadata["user_id"],
                            timestamp=timestamp,
                            importance=metadata.get("importance", 0.5),
                            event_type=metadata.get("event_type", "experience"),
                            location=metadata.get("location"),
                            participants=__import__("json").loads(metadata.get("participants", "[]")) if metadata.get("participants") else [],
                            emotion_data=emotion_data,
                            related_memories=__import__("json").loads(metadata.get("related_memories", "[]")) if metadata.get("related_memories") else [],
                            significance=metadata.get("significance", 0.5),
                            vividness=metadata.get("vividness", 0.5),
                            context=metadata.get("context")
                        )
                        items.append(item)
            
            # Сортируем по важности
            items.sort(key=lambda x: x.importance, reverse=True)
            
            return items
            
        except Exception as e:
            logger.error(f"Error getting emotional memories: {e}")
            return []
    
    async def get_timeline(self, user_id: str, start_date: datetime, 
                          end_date: datetime, limit: int = 100) -> List[EpisodicMemoryItem]:
        """
        Получить временную линию событий
        
        Args:
            user_id: ID пользователя
            start_date: Начальная дата
            end_date: Конечная дата
            limit: Максимальное количество элементов
            
        Returns:
            Список событий в указанном временном диапазоне
        """
        try:
            raw = await self._storage.get_timeframe(user_id=user_id, start=start_date, end=end_date, limit=limit)
            items: List[EpisodicMemoryItem] = []
            for it in raw:
                meta = it["metadata"]
                ts = datetime.fromisoformat(meta.get("timestamp"))
                items.append(EpisodicMemoryItem(
                    id=it["id"],
                    content=it["content"],
                    user_id=meta.get("user_id", user_id),
                    timestamp=ts,
                    importance=meta.get("importance", 0.5),
                    event_type=meta.get("event_type", "experience"),
                    location=meta.get("location"),
                    participants=__import__("json").loads(meta.get("participants", "[]")) if meta.get("participants") else [],
                    emotion_data=__import__("json").loads(meta.get("emotion_data", "{}")) if meta.get("emotion_data") else None,
                    related_memories=__import__("json").loads(meta.get("related_memories", "[]")) if meta.get("related_memories") else [],
                    significance=meta.get("significance", 0.5),
                    vividness=meta.get("vividness", 0.5),
                    context=meta.get("context")
                ))
            items.sort(key=lambda x: x.timestamp)
            return items
            
        except Exception as e:
            logger.error(f"Error getting timeline: {e}")
            return []
    
    async def update_significance(self, memory_id: str, significance: float) -> bool:
        """
        Обновить значимость воспоминания
        
        Args:
            memory_id: ID элемента
            significance: Новая значимость (0.0-1.0)
            
        Returns:
            True если успешно
        """
        try:
            # Получаем текущие метаданные
            results = self.collection.get(ids=[memory_id], include=["metadatas"])
            if not results["metadatas"]:
                return False
            
            metadata = results["metadatas"][0]
            metadata["significance"] = significance
            
            # Обновляем метаданные
            self.collection.update(
                ids=[memory_id],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated significance for {memory_id}: {significance}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating significance: {e}")
            return False
    
    async def add_relation(self, memory_id: str, related_memory_id: str) -> bool:
        """
        Добавить связь между воспоминаниями
        
        Args:
            memory_id: ID основного элемента
            related_memory_id: ID связанного элемента
            
        Returns:
            True если успешно
        """
        try:
            # Получаем текущие метаданные
            results = self.collection.get(ids=[memory_id], include=["metadatas"])
            if not results["metadatas"]:
                return False
            
            metadata = results["metadatas"][0]
            related_memories = __import__("json").loads(metadata.get("related_memories", "[]"))
            
            if related_memory_id not in related_memories:
                related_memories.append(related_memory_id)
                metadata["related_memories"] = __import__("json").dumps(related_memories)
                
                # Обновляем метаданные
                self.collection.update(
                    ids=[memory_id],
                    metadatas=[metadata]
                )
                
                logger.info(f"Added relation {related_memory_id} to {memory_id}")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding relation: {e}")
            return False
    
    async def remove_memory(self, memory_id: str) -> bool:
        """
        Удалить воспоминание из эпизодической памяти
        
        Args:
            memory_id: ID элемента
            
        Returns:
            True если успешно
        """
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Removed episodic memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing memory: {e}")
            return False
    
    async def _cleanup_old_memories(self, user_id: str):
        """Очистка старых элементов эпизодической памяти"""
        try:
            # TTL очистка и ограничение количества через storage
            old_ids = await self._storage.older_than_days(user_id=user_id, ttl_days=self.ttl_days, significance_threshold=0.3)
            if old_ids:
                await self._storage.delete_many(old_ids)
                logger.info(f"Cleaned up {len(old_ids)} old episodic memories")

            all_res = await self._storage.get_all(user_id=user_id, limit=2000)
            ids = list(all_res.get("ids") or [])
            metas = list(all_res.get("metadatas") or [])
            if len(ids) > self.max_items:
                items_with_score = []
                for i, mid in enumerate(ids):
                    meta = metas[i] or {}
                    try:
                        ts = datetime.fromisoformat(meta.get("timestamp"))
                    except Exception:
                        continue
                    score = float(meta.get("significance", 0.5)) * 0.7 + float(meta.get("importance", 0.5)) * 0.3
                    items_with_score.append((mid, score, ts))
                items_with_score.sort(key=lambda x: (x[1], x[2]))
                excess = [mid for mid, _, _ in items_with_score[: max(0, len(items_with_score) - self.max_items)]]
                if excess:
                    await self._storage.delete_many(excess)
                    logger.info(f"Removed {len(excess)} excess episodic memories")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")
    
    async def get_older_than(self, user_id: str, delta: timedelta, limit: int = 500) -> List[EpisodicMemoryItem]:
        """Вернуть элементы старше указанного дельта-времени."""
        try:
            cutoff = datetime.now() - delta
            results = self.collection.get(
                where={"user_id": user_id},
                limit=limit,
                include=["documents", "metadatas"]
            )
            items: List[EpisodicMemoryItem] = []
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i]
                ts = datetime.fromisoformat(metadata["timestamp"])
                if ts <= cutoff:
                    items.append(EpisodicMemoryItem(
                        id=results["ids"][i],
                        content=doc,
                        user_id=metadata["user_id"],
                        timestamp=ts,
                        importance=metadata.get("importance", 0.5),
                        event_type=metadata.get("event_type", "experience"),
                        location=metadata.get("location"),
                        participants=__import__("json").loads(metadata.get("participants", "[]")) if metadata.get("participants") else [],
                        emotion_data=__import__("json").loads(metadata.get("emotion_data", "{}")) if metadata.get("emotion_data") else None,
                        related_memories=__import__("json").loads(metadata.get("related_memories", "[]")) if metadata.get("related_memories") else [],
                        significance=metadata.get("significance", 0.5),
                        vividness=metadata.get("vividness", 0.5),
                        context=metadata.get("context")
                    ))
            return items
        except Exception as e:
            logger.error(f"Error getting older episodic memories: {e}")
            return []
    
    async def bulk_delete(self, ids: List[str]) -> int:
        """Удалить элементы по списку ID."""
        if not ids:
            return 0
        try:
            self.collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            logger.error(f"Error bulk deleting episodic memories: {e}")
            return 0
    
    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Получить статистику эпизодической памяти
        
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
                    "avg_significance": 0.0,
                    "event_types": {},
                    "oldest_item": None,
                    "newest_item": None
                }
            
            importances = [m.get("importance", 0.5) for m in results["metadatas"]]
            significances = [m.get("significance", 0.5) for m in results["metadatas"]]
            timestamps = [datetime.fromisoformat(m["timestamp"]) for m in results["metadatas"]]
            event_types = {}
            
            for metadata in results["metadatas"]:
                event_type = metadata.get("event_type", "experience")
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            return {
                "total_items": len(results["metadatas"]),
                "avg_importance": sum(importances) / len(importances),
                "avg_significance": sum(significances) / len(significances),
                "event_types": event_types,
                "oldest_item": min(timestamps).isoformat(),
                "newest_item": max(timestamps).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    async def get_memory_by_id(self, memory_id: str, user_id: str) -> Optional[EpisodicMemoryItem]:
        """
        Получить элемент эпизодической памяти по ID
        
        Args:
            memory_id: ID элемента памяти
            user_id: ID пользователя
            
        Returns:
            EpisodicMemoryItem или None если не найден
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
            
            # Парсим participants и related_memories
            participants = __import__("json").loads(metadata.get("participants", "[]")) if metadata.get("participants") else []
            related_memories = __import__("json").loads(metadata.get("related_memories", "[]")) if metadata.get("related_memories") else []
            
            return EpisodicMemoryItem(
                id=memory_id,
                content=doc,
                user_id=metadata["user_id"],
                timestamp=ts,
                importance=metadata.get("importance", 0.5),
                event_type=metadata.get("event_type", "experience"),
                location=metadata.get("location"),
                participants=participants,
                emotion_data=__import__("json").loads(metadata.get("emotion_data", "{}")) if metadata.get("emotion_data") else None,
                related_memories=related_memories,
                significance=metadata.get("significance", 0.5),
                vividness=metadata.get("vividness", 0.5),
                context=metadata.get("context")
            )
            
        except Exception as e:
            logger.error(f"Error getting episodic memory by ID {memory_id}: {e}")
            return None
    
    async def update_metadata(self, memory_id: str, user_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Обновить метаданные элемента эпизодической памяти
        
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
                logger.warning(f"Episodic memory item {memory_id} not found for user {user_id}")
                return False
            
            # Обновляем метаданные
            updated_metadata = {
                "user_id": user_id,
                "timestamp": current_item.timestamp.isoformat(),
                "importance": current_item.importance,
                "event_type": current_item.event_type,
                "location": current_item.location,
                "participants": __import__("json").dumps(current_item.participants) if current_item.participants else None,
                "emotion_data": __import__("json").dumps(current_item.emotion_data) if current_item.emotion_data else None,
                "related_memories": __import__("json").dumps(current_item.related_memories) if current_item.related_memories else None,
                "significance": current_item.significance,
                "vividness": current_item.vividness,
                "context": current_item.context
            }
            
            # Добавляем новые метаданные
            updated_metadata.update(metadata)
            
            # Обновляем в ChromaDB
            self.collection.update(
                ids=[memory_id],
                metadatas=[updated_metadata]
            )
            
            logger.debug(f"Updated metadata for episodic memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating episodic memory metadata {memory_id}: {e}")
            return False
    
    async def get_memories(self, user_id: str, limit: int = 20) -> List[EpisodicMemoryItem]:
        """
        Получить все воспоминания пользователя из эпизодической памяти
        
        Args:
            user_id: ID пользователя
            limit: Максимальное количество элементов
            
        Returns:
            Список элементов эпизодической памяти
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
                    
                    # Создаем объект EpisodicMemoryItem
                    memory_item = EpisodicMemoryItem(
                        id=memory_id,
                        content=content,
                        user_id=user_id,
                        timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                        importance=metadata.get("importance", 0.5),
                        event_type=metadata.get("event_type", "experience"),
                        location=metadata.get("location"),
                        participants=__import__("json").loads(metadata.get("participants", "[]")) if metadata.get("participants") else [],
                        emotion_data=__import__("json").loads(metadata.get("emotion_data", "{}")) if metadata.get("emotion_data") else None,
                        related_memories=__import__("json").loads(metadata.get("related_memories", "[]")) if metadata.get("related_memories") else [],
                        sensory_details=__import__("json").loads(metadata.get("sensory_details", "{}")) if metadata.get("sensory_details") else None
                    )
                    memories.append(memory_item)
            
            logger.debug(f"Retrieved {len(memories)} episodic memories for user {user_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Error getting episodic memories for user {user_id}: {e}")
            return []
