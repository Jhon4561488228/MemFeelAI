"""
Short-term Memory Manager для AIRI Memory System
Уровень 2: Кратковременная память - события последних 24 часов, TTL = 24 часа
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from ..utils.metadata_serializer import MetadataSerializer
except ImportError:
    from utils.metadata_serializer import MetadataSerializer

try:
    from ..storage.short_term_storage import ShortTermMemoryStorage
except Exception:  # pragma: no cover
    from storage.short_term_storage import ShortTermMemoryStorage

logger = logging.getLogger(__name__)

@dataclass
class ShortTermMemoryItem:
    """Элемент кратковременной памяти"""
    id: str
    content: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    event_type: str = "general"
    location: Optional[str] = None
    participants: List[str] = None
    emotion_data: Optional[Dict[str, Any]] = None
    related_memories: List[str] = None

class ShortTermMemoryManager:
    """Менеджер кратковременной памяти"""
    
    def __init__(self, chromadb_path: str = "./data/chroma_db"):
        self.chromadb_path = chromadb_path
        self._storage = ShortTermMemoryStorage(chromadb_path=chromadb_path)
        self.collection = self._storage.collection
        self.max_items = 100  # Максимум элементов в кратковременной памяти
        self.ttl_hours = 24   # Время жизни элемента (24 часа)
        
    async def add_event(self, content: str, user_id: str, 
                       importance: float = 0.5, confidence: float = 0.5, event_type: str = "general",
                       location: Optional[str] = None, participants: List[str] = None,
                       emotion_data: Optional[Dict[str, Any]] = None,
                       related_memories: List[str] = None) -> str:
        """
        Добавить событие в кратковременную память
        
        Args:
            content: Содержимое события
            user_id: ID пользователя
            importance: Важность (0.0-1.0)
            event_type: Тип события (conversation, action, observation, etc.)
            location: Местоположение
            participants: Участники события
            emotion_data: Эмоциональные данные
            related_memories: Связанные воспоминания
            
        Returns:
            ID созданного элемента
        """
        try:
            memory_id = f"stm_{uuid.uuid4()}"
            timestamp = datetime.now()
            
            # Создаем метаданные
            metadata = {
                "user_id": user_id,
                "timestamp": timestamp.isoformat(),
                "importance": importance,
                "confidence": confidence,
                "event_type": event_type,
                "location": location or "",
                "participants": MetadataSerializer.safe_serialize_list(participants or []),
                "emotion_data": MetadataSerializer.safe_serialize_dict(emotion_data or {}),
                "related_memories": MetadataSerializer.safe_serialize_list(related_memories or []),
                "memory_type": "short_term"
            }
            
            # Добавляем через storage
            await self._storage.add(memory_id, content, metadata)
            
            # Очищаем старые элементы
            await self._cleanup_old_memories(user_id)
            
            logger.info(f"Added short-term memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error adding short-term memory: {e}")
            raise
    
    async def get_recent_events(self, user_id: str, hours: int = 24, limit: int = 50) -> List[ShortTermMemoryItem]:
        """
        Получить недавние события
        
        Args:
            user_id: ID пользователя
            hours: Количество часов назад
            limit: Максимальное количество элементов
            
        Returns:
            Список недавних событий
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            raw = await self._storage.recent(user_id=user_id, since=cutoff_time, limit=limit)
            items: List[ShortTermMemoryItem] = []
            for it in raw:
                meta = it["metadata"]
                items.append(ShortTermMemoryItem(
                    id=it["id"],
                    content=it["content"],
                    user_id=meta.get("user_id", user_id),
                    timestamp=datetime.fromisoformat(meta.get("timestamp")),
                    importance=meta.get("importance", 0.5),
                    event_type=meta.get("event_type", "general"),
                    location=meta.get("location"),
                    participants=MetadataSerializer.safe_deserialize_list(meta.get("participants", "[]")),
                    emotion_data=MetadataSerializer.safe_deserialize_dict(meta.get("emotion_data", "{}")),
                    related_memories=MetadataSerializer.safe_deserialize_list(meta.get("related_memories", "[]"))
                ))
            return items
            
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
    async def search_events(self, user_id: str, query: str, 
                           event_type: Optional[str] = None, 
                           hours: int = 24, limit: int = 10,
                           emotion_filter: Optional[str] = None) -> List[ShortTermMemoryItem]:
        """
        Поиск событий в кратковременной памяти
        
        Args:
            user_id: ID пользователя
            query: Поисковый запрос
            event_type: Тип события для фильтрации
            hours: Количество часов назад
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных событий
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Строим фильтр
            where_filter = {"user_id": user_id}
            if event_type:
                where_filter["event_type"] = event_type
            
            results = await self._storage.search(user_id=user_id, query=query, limit=limit, where_extra={"event_type": event_type} if event_type else None)
            
            # Собираем и ранжируем
            ranked: List[Tuple[float, ShortTermMemoryItem]] = []
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = float(results["distances"][0][i])
                timestamp = datetime.fromisoformat(metadata["timestamp"])
                if timestamp < cutoff_time or distance >= 0.8:
                    continue
                emotion_data = MetadataSerializer.safe_deserialize_dict(metadata.get("emotion_data", "{}"))
                # Эмоциональное совпадение
                emo_match = 0.0
                if emotion_filter:
                    emo_text = str(emotion_data).lower()
                    if emotion_filter.lower() in emo_text:
                        emo_match = 1.0
                importance = float(metadata.get("importance", 0.5))
                # Рекенси: чем ближе к now, тем выше (0..1)
                age_hours = max(0.0, (datetime.now() - timestamp).total_seconds() / 3600.0)
                recency = max(0.0, 1.0 - min(age_hours / max(1.0, float(hours)), 1.0))
                relevance = max(0.0, 1.0 - distance)
                # Веса: релевантность 0.5, важность 0.2, свежесть 0.2, эмоция 0.1
                score = relevance * 0.5 + importance * 0.2 + recency * 0.2 + emo_match * 0.1
                item = ShortTermMemoryItem(
                    id=results["ids"][0][i],
                    content=doc,
                    user_id=metadata["user_id"],
                    timestamp=timestamp,
                    importance=importance,
                    event_type=metadata.get("event_type", "general"),
                    location=metadata.get("location"),
                    participants=MetadataSerializer.safe_deserialize_list(metadata.get("participants", "[]")),
                    emotion_data=emotion_data if emotion_data else None,
                    related_memories=MetadataSerializer.safe_deserialize_list(metadata.get("related_memories", "[]"))
                )
                ranked.append((score, item))

            ranked.sort(key=lambda x: x[0], reverse=True)
            return [it for _, it in ranked[:limit]]
            
        except Exception as e:
            logger.error(f"Error searching events: {e}")
            return []
    
    async def get_events_by_type(self, user_id: str, event_type: str, 
                                hours: int = 24, limit: int = 20) -> List[ShortTermMemoryItem]:
        """
        Получить события по типу
        
        Args:
            user_id: ID пользователя
            event_type: Тип события
            hours: Количество часов назад
            limit: Максимальное количество элементов
            
        Returns:
            Список событий указанного типа
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            results = await self._storage.get_all(user_id=user_id, limit=limit)
            
            items = []
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i]
                timestamp = datetime.fromisoformat(metadata["timestamp"])
                
                # Фильтруем по времени
                if timestamp >= cutoff_time:
                    item = ShortTermMemoryItem(
                        id=results["ids"][i],
                        content=doc,
                        user_id=metadata["user_id"],
                        timestamp=timestamp,
                        importance=metadata.get("importance", 0.5),
                        event_type=metadata.get("event_type", "general"),
                        location=metadata.get("location"),
                         participants=MetadataSerializer.safe_deserialize_list(metadata.get("participants", "[]")),
                        emotion_data=MetadataSerializer.safe_deserialize_dict(metadata.get("emotion_data", "{}")),
                        related_memories=MetadataSerializer.safe_deserialize_list(metadata.get("related_memories", "[]"))
                    )
                    items.append(item)
            
            # Сортируем по времени (новые первые)
            items.sort(key=lambda x: x.timestamp, reverse=True)
            
            return items
            
        except Exception as e:
            logger.error(f"Error getting events by type: {e}")
            return []
    
    async def update_importance(self, memory_id: str, importance: float) -> bool:
        """
        Обновить важность события
        
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
            related_memories = MetadataSerializer.safe_deserialize_list(metadata.get("related_memories", "[]"))
            
            if related_memory_id not in related_memories:
                related_memories.append(related_memory_id)
                metadata["related_memories"] = MetadataSerializer.safe_serialize_list(related_memories)
                
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
        Удалить событие из кратковременной памяти
        
        Args:
            memory_id: ID элемента
            
        Returns:
            True если успешно
        """
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Removed short-term memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing memory: {e}")
            return False
    
    async def _cleanup_old_memories(self, user_id: str):
        """Очистка старых элементов кратковременной памяти"""
        try:
            # TTL очистка
            old_ids = await self._storage.older_than(user_id=user_id, ttl_hours=self.ttl_hours)
            if old_ids:
                await self._storage.delete_many(old_ids)
                logger.info(f"Cleaned up {len(old_ids)} old short-term memories")

            # Ограничение количества
            all_res = await self._storage.get_all(user_id=user_id, limit=1000)
            ids = list(all_res.get("ids") or [])
            metas = list(all_res.get("metadatas") or [])
            if len(ids) > self.max_items:
                items_with_time = []
                for i, mid in enumerate(ids):
                    try:
                        ts = datetime.fromisoformat((metas[i] or {}).get("timestamp"))
                    except Exception:
                        continue
                    items_with_time.append((mid, ts))
                items_with_time.sort(key=lambda x: x[1])
                excess = [mid for mid, _ in items_with_time[: max(0, len(items_with_time) - self.max_items)]]
                if excess:
                    await self._storage.delete_many(excess)
                    logger.info(f"Removed {len(excess)} excess short-term memories")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")
    
    async def get_older_than(self, user_id: str, delta: timedelta, limit: int = 200) -> List[ShortTermMemoryItem]:
        """Вернуть элементы старше указанного дельта-времени."""
        try:
            cutoff = datetime.now() - delta
            results = self.collection.get(
                where={"user_id": user_id},
                limit=limit,
                include=["documents", "metadatas"]
            )
            items: List[ShortTermMemoryItem] = []
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i]
                ts = datetime.fromisoformat(metadata["timestamp"])
                if ts <= cutoff:
                    items.append(ShortTermMemoryItem(
                        id=results["ids"][i],
                        content=doc,
                        user_id=metadata["user_id"],
                        timestamp=ts,
                        importance=metadata.get("importance", 0.5),
                        event_type=metadata.get("event_type", "general"),
                        location=metadata.get("location"),
                        participants=MetadataSerializer.safe_deserialize_list(metadata.get("participants", "[]")),
                        emotion_data=MetadataSerializer.safe_deserialize_dict(metadata.get("emotion_data", "{}")),
                        related_memories=MetadataSerializer.safe_deserialize_list(metadata.get("related_memories", "[]"))
                    ))
            return items
        except Exception as e:
            logger.error(f"Error getting older short-term memories: {e}")
            return []
    
    async def bulk_delete(self, ids: List[str]) -> int:
        """Удалить элементы по списку ID."""
        if not ids:
            return 0
        try:
            self.collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            logger.error(f"Error bulk deleting short-term memories: {e}")
            return 0
    
    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Получить статистику кратковременной памяти
        
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
                    "event_types": {},
                    "oldest_item": None,
                    "newest_item": None
                }
            
            importances = [m.get("importance", 0.5) for m in results["metadatas"]]
            timestamps = [datetime.fromisoformat(m["timestamp"]) for m in results["metadatas"]]
            event_types = {}
            
            for metadata in results["metadatas"]:
                event_type = metadata.get("event_type", "general")
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            return {
                "total_items": len(results["metadatas"]),
                "avg_importance": sum(importances) / len(importances),
                "event_types": event_types,
                "oldest_item": min(timestamps).isoformat(),
                "newest_item": max(timestamps).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    async def get_memory_by_id(self, memory_id: str, user_id: str) -> Optional[ShortTermMemoryItem]:
        """
        Получить элемент кратковременной памяти по ID
        
        Args:
            memory_id: ID элемента памяти
            user_id: ID пользователя
            
        Returns:
            ShortTermMemoryItem или None если не найден
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
            participants = MetadataSerializer.safe_deserialize_list(metadata.get("participants", "[]"))
            related_memories = MetadataSerializer.safe_deserialize_list(metadata.get("related_memories", "[]"))
            
            return ShortTermMemoryItem(
                id=memory_id,
                content=doc,
                user_id=metadata["user_id"],
                timestamp=ts,
                importance=metadata.get("importance", 0.5),
                event_type=metadata.get("event_type", "general"),
                location=metadata.get("location"),
                participants=participants,
                emotion_data=MetadataSerializer.safe_deserialize_dict(metadata.get("emotion_data", "{}")),
                related_memories=related_memories
            )
            
        except Exception as e:
            logger.error(f"Error getting short-term memory by ID {memory_id}: {e}")
            return None
    
    async def update_metadata(self, memory_id: str, user_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Обновить метаданные элемента кратковременной памяти
        
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
                logger.warning(f"Short-term memory item {memory_id} not found for user {user_id}")
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
                "related_memories": __import__("json").dumps(current_item.related_memories) if current_item.related_memories else None
            }
            
            # Добавляем новые метаданные
            updated_metadata.update(metadata)
            
            # Обновляем в ChromaDB
            self.collection.update(
                ids=[memory_id],
                metadatas=[updated_metadata]
            )
            
            logger.debug(f"Updated metadata for short-term memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating short-term memory metadata {memory_id}: {e}")
            return False
    
    async def get_memories(self, user_id: str, limit: int = 20) -> List[ShortTermMemoryItem]:
        """
        Получить все воспоминания пользователя из краткосрочной памяти
        
        Args:
            user_id: ID пользователя
            limit: Максимальное количество элементов
            
        Returns:
            Список элементов краткосрочной памяти
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
                    
                    # Создаем объект ShortTermMemoryItem
                    memory_item = ShortTermMemoryItem(
                        id=memory_id,
                        content=content,
                        user_id=user_id,
                        timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                        importance=metadata.get("importance", 0.5),
                        event_type=metadata.get("event_type", "general"),
                        location=metadata.get("location"),
                        participants=MetadataSerializer.safe_deserialize_list(metadata.get("participants", "[]")),
                        emotion_data=MetadataSerializer.safe_deserialize_dict(metadata.get("emotion_data", "{}")),
                        related_memories=MetadataSerializer.safe_deserialize_list(metadata.get("related_memories", "[]"))
                    )
                    memories.append(memory_item)
            
            logger.debug(f"Retrieved {len(memories)} short-term memories for user {user_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Error getting short-term memories for user {user_id}: {e}")
            return []
