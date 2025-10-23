"""
Semantic Memory Manager для AIRI Memory System
Уровень 4: Семантическая память - факты, знания, концепции, TTL = 365 дней
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

try:
    from ..utils.metadata_serializer import MetadataSerializer
except ImportError:
    from utils.metadata_serializer import MetadataSerializer

try:
    from ..storage.semantic_storage import SemanticMemoryStorage
except Exception:  # pragma: no cover
    from storage.semantic_storage import SemanticMemoryStorage

try:
    from ..classifiers.knowledge_classifier import KnowledgeClassifier
except Exception:  # pragma: no cover
    from classifiers.knowledge_classifier import KnowledgeClassifier

logger = logging.getLogger(__name__)

@dataclass
class SemanticMemoryItem:
    """Элемент семантической памяти"""
    id: str
    content: str
    user_id: str
    timestamp: datetime
    knowledge_type: str = "fact"
    category: str = "general"
    confidence: float = 0.5
    source: Optional[str] = None
    related_concepts: List[str] = None
    tags: List[str] = None
    importance: float = 0.5
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    distance: Optional[float] = None  # Добавляем поле distance для дедупликации

class SemanticMemoryManager:
    """Менеджер семантической памяти"""
    
    def __init__(self, chromadb_path: str = "./data/chroma_db"):
        self.chromadb_path = chromadb_path
        self._storage = SemanticMemoryStorage(chromadb_path=chromadb_path)
        self.collection = self._storage.collection
        self._classifier = KnowledgeClassifier()
        self.max_items = 10000  # Максимум элементов в семантической памяти
        self.ttl_days = 365     # Время жизни элемента (365 дней)
        
    async def add_knowledge(self, content: str, user_id: str, 
                           knowledge_type: str = "fact", category: str = "",
                           confidence: float = 0.5, source: Optional[str] = None,
                           related_concepts: List[str] = None, tags: List[str] = None,
                           importance: float = 0.5) -> str:
        """
        Добавить знание в семантическую память
        
        Args:
            content: Содержимое знания
            user_id: ID пользователя
            knowledge_type: Тип знания (fact, concept, rule, definition, etc.)
            category: Категория знания
            confidence: Уверенность в знании (0.0-1.0)
            source: Источник знания
            related_concepts: Связанные концепции
            tags: Теги для классификации
            importance: Важность знания (0.0-1.0)
            
        Returns:
            ID созданного элемента
        """
        try:
            memory_id = f"sm_{uuid.uuid4()}"
            timestamp = datetime.now()
            # Авто-категоризация при пустой категории
            auto_category = category.strip() if category else await self._classifier.classify(content)
            
            # Создаем метаданные
            metadata = {
                "user_id": user_id,
                "timestamp": timestamp.isoformat(),
                "knowledge_type": knowledge_type,
                "category": auto_category,
                "confidence": confidence,
                "source": source or "",
                "related_concepts": MetadataSerializer.safe_serialize_list(related_concepts or []),
                "tags": MetadataSerializer.safe_serialize_list(tags or []),
                "importance": importance,
                "last_accessed": timestamp.isoformat(),
                "access_count": 0,
                "memory_type": "semantic"
            }
            
            # Добавляем через storage
            await self._storage.add(memory_id, content, metadata)
            
            # Очищаем старые элементы
            await self._cleanup_old_memories(user_id)
            
            logger.info(f"Added semantic memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error adding semantic memory: {e}")
            raise
    
    async def search_knowledge(self, user_id: str, query: str, 
                             knowledge_type: Optional[str] = None,
                             category: Optional[str] = None,
                             min_confidence: float = 0.25, limit: int = 10) -> List[SemanticMemoryItem]:
        """
        Поиск знаний в семантической памяти
        
        Args:
            user_id: ID пользователя
            query: Поисковый запрос
            knowledge_type: Тип знания для фильтрации
            category: Категория для фильтрации
            min_confidence: Минимальная уверенность
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных знаний
        """
        try:
            logger.info(f"SemanticMemoryManager.search_knowledge: query='{query}', user_id='{user_id}', min_confidence={min_confidence}, limit={limit}")
            results = await self._storage.search(user_id=user_id, query=query, limit=limit, knowledge_type=knowledge_type, category=category)
            
            items = []
            logger.info(f"SemanticMemoryManager: ChromaDB returned {len(results['documents'][0])} documents")
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                confidence = metadata.get("confidence", 0.5)
                
                logger.info(f"SemanticMemoryManager: Document {i}: distance={distance}, confidence={confidence}, min_confidence={min_confidence}")
                
                # Фильтруем по релевантности и уверенности
                # Порог distance=0.8 корректен для cosine similarity (1-distance)
                if distance < 0.8 and confidence >= min_confidence:
                    logger.info(f"SemanticMemoryManager: Document {i} PASSED filter")
                    item = SemanticMemoryItem(
                        id=results["ids"][0][i],
                        content=doc,
                        user_id=metadata["user_id"],
                        timestamp=datetime.fromisoformat(metadata["timestamp"]),
                        knowledge_type=metadata.get("knowledge_type", "fact"),
                        category=metadata.get("category", "general"),
                        confidence=confidence,
                        source=metadata.get("source"),
                        related_concepts=metadata.get("related_concepts", []),
                        tags=metadata.get("tags", []),
                        importance=metadata.get("importance", 0.5),
                        last_accessed=datetime.fromisoformat(metadata.get("last_accessed", metadata["timestamp"])),
                        access_count=metadata.get("access_count", 0),
                        distance=distance  # Добавляем distance для дедупликации
                    )
                    items.append(item)
                else:
                    # Детальная диагностика причин отклонения
                    distance_fail = distance >= 0.8
                    confidence_fail = confidence < min_confidence
                    if distance_fail and confidence_fail:
                        reason = f"distance={distance:.3f} >= 0.8 (cosine_sim={1-distance:.3f}) AND confidence={confidence} < {min_confidence}"
                    elif distance_fail:
                        reason = f"distance={distance:.3f} >= 0.8 (cosine_sim={1-distance:.3f}) (confidence={confidence} >= {min_confidence} OK)"
                    elif confidence_fail:
                        reason = f"confidence={confidence} < {min_confidence} (distance={distance:.3f} < 0.8, cosine_sim={1-distance:.3f} OK)"
                    else:
                        reason = "unknown reason"
                    logger.info(f"SemanticMemoryManager: Document {i} FAILED filter - {reason}")
            
            logger.info(f"SemanticMemoryManager: Final result: {len(items)} items passed filter")
            # Обновляем статистику доступа (без увеличения счетчика)
            for item in items:
                await self._update_access_stats(item.id, increment_count=False)
            
            return items
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    async def get_knowledge_by_category(self, user_id: str, category: str, 
                                       limit: int = 50) -> List[SemanticMemoryItem]:
        """
        Получить знания по категории
        
        Args:
            user_id: ID пользователя
            category: Категория знаний
            limit: Максимальное количество элементов
            
        Returns:
            Список знаний указанной категории
        """
        try:
            results = await self._storage.get_by_category(user_id=user_id, category=category, limit=limit)
            
            items = []
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i]
                item = SemanticMemoryItem(
                    id=results["ids"][i],
                    content=doc,
                    user_id=metadata["user_id"],
                    timestamp=datetime.fromisoformat(metadata["timestamp"]),
                    knowledge_type=metadata.get("knowledge_type", "fact"),
                    category=metadata.get("category", "general"),
                    confidence=metadata.get("confidence", 0.5),
                    source=metadata.get("source"),
                    related_concepts=MetadataSerializer.safe_deserialize_list(metadata.get("related_concepts", "[]")),
                    tags=MetadataSerializer.safe_deserialize_list(metadata.get("tags", "[]")),
                    importance=metadata.get("importance", 0.5),
                    last_accessed=datetime.fromisoformat(metadata.get("last_accessed", metadata["timestamp"])),
                    access_count=metadata.get("access_count", 0)
                )
                items.append(item)
            
            # Сортируем по важности и уверенности
            items.sort(key=lambda x: (x.importance, x.confidence), reverse=True)
            
            return items
            
        except Exception as e:
            logger.error(f"Error getting knowledge by category: {e}")
            return []
    
    async def get_related_concepts(self, user_id: str, concept: str, 
                                 limit: int = 20) -> List[SemanticMemoryItem]:
        """
        Получить связанные концепции
        
        Args:
            user_id: ID пользователя
            concept: Концепция для поиска
            limit: Максимальное количество элементов
            
        Returns:
            Список связанных концепций
        """
        try:
            results = await self._storage.get_all(user_id=user_id, limit=limit)
            
            items = []
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i]
                related_concepts = MetadataSerializer.safe_deserialize_list(metadata.get("related_concepts", "[]"))
                
                # Проверяем наличие концепции в связанных
                if concept.lower() in [c.lower() for c in related_concepts]:
                    item = SemanticMemoryItem(
                        id=results["ids"][i],
                        content=doc,
                        user_id=metadata["user_id"],
                        timestamp=datetime.fromisoformat(metadata["timestamp"]),
                        knowledge_type=metadata.get("knowledge_type", "fact"),
                        category=metadata.get("category", "general"),
                        confidence=metadata.get("confidence", 0.5),
                        source=metadata.get("source"),
                        related_concepts=related_concepts,
                        tags=metadata.get("tags", []),
                        importance=metadata.get("importance", 0.5),
                        last_accessed=datetime.fromisoformat(metadata.get("last_accessed", metadata["timestamp"])),
                        access_count=metadata.get("access_count", 0)
                    )
                    items.append(item)
            
            # Сортируем по важности
            items.sort(key=lambda x: x.importance, reverse=True)
            
            return items
            
        except Exception as e:
            logger.error(f"Error getting related concepts: {e}")
            return []
    
    async def update_confidence(self, memory_id: str, confidence: float) -> bool:
        """
        Обновить уверенность в знании
        
        Args:
            memory_id: ID элемента
            confidence: Новая уверенность (0.0-1.0)
            
        Returns:
            True если успешно
        """
        try:
            # Получаем текущие метаданные
            results = self.collection.get(ids=[memory_id], include=["metadatas"])
            if not results["metadatas"]:
                return False
            
            metadata = results["metadatas"][0]
            metadata["confidence"] = confidence
            
            # Обновляем метаданные
            self.collection.update(
                ids=[memory_id],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated confidence for {memory_id}: {confidence}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating confidence: {e}")
            return False
    
    async def add_related_concept(self, memory_id: str, concept: str) -> bool:
        """
        Добавить связанную концепцию
        
        Args:
            memory_id: ID элемента
            concept: Концепция для добавления
            
        Returns:
            True если успешно
        """
        try:
            # Получаем текущие метаданные
            results = self.collection.get(ids=[memory_id], include=["metadatas"])
            if not results["metadatas"]:
                return False
            
            metadata = results["metadatas"][0]
            related_concepts = MetadataSerializer.safe_deserialize_list(metadata.get("related_concepts", "[]"))
            
            if concept not in related_concepts:
                related_concepts.append(concept)
                metadata["related_concepts"] = MetadataSerializer.safe_serialize_list(related_concepts)
                
                # Обновляем метаданные
                self.collection.update(
                    ids=[memory_id],
                    metadatas=[metadata]
                )
                
                logger.info(f"Added related concept {concept} to {memory_id}")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding related concept: {e}")
            return False
    
    async def add_tag(self, memory_id: str, tag: str) -> bool:
        """
        Добавить тег к знанию
        
        Args:
            memory_id: ID элемента
            tag: Тег для добавления
            
        Returns:
            True если успешно
        """
        try:
            # Получаем текущие метаданные
            results = self.collection.get(ids=[memory_id], include=["metadatas"])
            if not results["metadatas"]:
                return False
            
            metadata = results["metadatas"][0]
            tags = metadata.get("tags", [])
            
            if tag not in tags:
                tags.append(tag)
                metadata["tags"] = MetadataSerializer.safe_serialize_list(tags)
                
                # Обновляем метаданные
                self.collection.update(
                    ids=[memory_id],
                    metadatas=[metadata]
                )
                
                logger.info(f"Added tag {tag} to {memory_id}")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding tag: {e}")
            return False
    
    async def _update_access_stats(self, memory_id: str, increment_count: bool = False):
        """Обновить статистику доступа к знанию"""
        try:
            # Получаем текущие метаданные
            results = self.collection.get(ids=[memory_id], include=["metadatas"])
            if not results["metadatas"]:
                return
            
            metadata = results["metadatas"][0]
            metadata["last_accessed"] = datetime.now().isoformat()
            
            # Увеличиваем access_count только при реальном доступе, а не при поиске
            if increment_count:
                metadata["access_count"] = metadata.get("access_count", 0) + 1
            
            # Обновляем метаданные
            self.collection.update(
                ids=[memory_id],
                metadatas=[metadata]
            )
            
        except Exception as e:
            logger.error(f"Error updating access stats: {e}")
    
    async def remove_memory(self, memory_id: str) -> bool:
        """
        Удалить знание из семантической памяти
        
        Args:
            memory_id: ID элемента
            
        Returns:
            True если успешно
        """
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Removed semantic memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing memory: {e}")
            return False
    
    async def _cleanup_old_memories(self, user_id: str):
        """Очистка старых элементов семантической памяти"""
        try:
            # TTL и ограничение через storage
            old_ids = await self._storage.older_than_days(user_id=user_id, ttl_days=self.ttl_days, importance_threshold=0.3, min_access_count=2)
            if old_ids:
                await self._storage.delete_many(old_ids)
                logger.info(f"Cleaned up {len(old_ids)} old semantic memories")

            all_res = await self._storage.get_all(user_id=user_id, limit=5000)
            ids = list(all_res.get("ids") or [])
            metas = list(all_res.get("metadatas") or [])
            if len(ids) > self.max_items:
                items_with_score = []
                for i, mid in enumerate(ids):
                    meta = metas[i] or {}
                    score = float(meta.get("importance", 0.5)) * 0.6 + min(float(meta.get("access_count", 0)) / 10, 1.0) * 0.4
                    items_with_score.append((mid, score))
                items_with_score.sort(key=lambda x: x[1])
                excess = [mid for mid, _ in items_with_score[: max(0, len(items_with_score) - self.max_items)]]
                if excess:
                    await self._storage.delete_many(excess)
                    logger.info(f"Removed {len(excess)} excess semantic memories")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")
    
    async def bulk_delete(self, ids: List[str]) -> int:
        """Удалить элементы по списку ID."""
        if not ids:
            return 0
        try:
            self.collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            logger.error(f"Error bulk deleting semantic memories: {e}")
            return 0
    
    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Получить статистику семантической памяти
        
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
                    "avg_confidence": 0.0,
                    "avg_importance": 0.0,
                    "knowledge_types": {},
                    "categories": {},
                    "most_accessed": 0
                }
            
            confidences = [m.get("confidence", 0.5) for m in results["metadatas"]]
            importances = [m.get("importance", 0.5) for m in results["metadatas"]]
            access_counts = [m.get("access_count", 0) for m in results["metadatas"]]
            knowledge_types = {}
            categories = {}
            
            for metadata in results["metadatas"]:
                kt = metadata.get("knowledge_type", "fact")
                cat = metadata.get("category", "general")
                knowledge_types[kt] = knowledge_types.get(kt, 0) + 1
                categories[cat] = categories.get(cat, 0) + 1
            
            return {
                "total_items": len(results["metadatas"]),
                "avg_confidence": sum(confidences) / len(confidences),
                "avg_importance": sum(importances) / len(importances),
                "knowledge_types": knowledge_types,
                "categories": categories,
                "most_accessed": max(access_counts) if access_counts else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    async def get_memory_by_id(self, memory_id: str, user_id: str) -> Optional[SemanticMemoryItem]:
        """
        Получить элемент семантической памяти по ID
        
        Args:
            memory_id: ID элемента памяти
            user_id: ID пользователя
            
        Returns:
            SemanticMemoryItem или None если не найден
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
            
            # Парсим related_concepts и tags
            related_concepts = MetadataSerializer.safe_deserialize_list(metadata.get("related_concepts", "[]"))
            tags = MetadataSerializer.safe_deserialize_list(metadata.get("tags", "[]"))
            
            # Парсим last_accessed если есть
            last_accessed = None
            if metadata.get("last_accessed"):
                last_accessed = datetime.fromisoformat(metadata["last_accessed"])
            
            return SemanticMemoryItem(
                id=memory_id,
                content=doc,
                user_id=metadata["user_id"],
                timestamp=ts,
                knowledge_type=metadata.get("knowledge_type", "fact"),
                category=metadata.get("category", "general"),
                confidence=metadata.get("confidence", 0.5),
                source=metadata.get("source"),
                related_concepts=related_concepts,
                tags=tags,
                importance=metadata.get("importance", 0.5),
                last_accessed=last_accessed,
                access_count=metadata.get("access_count", 0)
            )
            
        except Exception as e:
            logger.error(f"Error getting semantic memory by ID {memory_id}: {e}")
            return None
    
    async def update_metadata(self, memory_id: str, user_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Обновить метаданные элемента семантической памяти
        
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
                logger.warning(f"Semantic memory item {memory_id} not found for user {user_id}")
                return False
            
            # Обновляем метаданные
            updated_metadata = {
                "user_id": user_id,
                "timestamp": current_item.timestamp.isoformat(),
                "knowledge_type": current_item.knowledge_type,
                "category": current_item.category,
                "confidence": current_item.confidence,
                "source": current_item.source,
                "related_concepts": MetadataSerializer.safe_serialize_list(current_item.related_concepts) if current_item.related_concepts else None,
                "tags": MetadataSerializer.safe_serialize_list(current_item.tags) if current_item.tags else None,
                "importance": current_item.importance,
                "last_accessed": current_item.last_accessed.isoformat() if current_item.last_accessed else None,
                "access_count": current_item.access_count
            }
            
            # Добавляем новые метаданные
            updated_metadata.update(metadata)
            
            # Обновляем в ChromaDB
            self.collection.update(
                ids=[memory_id],
                metadatas=[updated_metadata]
            )
            
            logger.debug(f"Updated metadata for semantic memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating semantic memory metadata {memory_id}: {e}")
            return False
