"""
Memory Consolidator для AIRI Memory System
Умная консолидация и очистка памяти с учетом важности и использования
"""

import asyncio
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from ..monitoring.metrics import inc, set_gauge, record_event
except Exception:
    from monitoring.metrics import inc, set_gauge, record_event

# LocalMem0 НЕ используется - это legacy система
# Текущая система использует MemoryOrchestrator с 6-уровневой памятью
LOCAL_MEM0_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MemoryItemScore:
    """Оценка элемента памяти для консолидации"""
    item_id: str
    item_type: str
    score: float
    importance: float
    access_count: int
    last_accessed: Optional[datetime]
    age_days: int
    recency_days: int
    emotional_weight: float
    consolidation_reason: str
    content: Optional[str] = None  # Добавляем поле для текстового содержимого

class MemoryConsolidator:
    """
    Умная консолидация памяти с учетом важности, использования и эмоциональной значимости
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        
        # LocalMem0 НЕ используется - это legacy система
        # Используем прямую работу с memory managers через orchestrator
        self.local_mem0 = None
        logger.info("MemoryConsolidator инициализирован для работы с 6-уровневой памятью")
        
        self.consolidation_thresholds = {
            "sensor": 1000,     # Максимум 1000 элементов в Sensor Buffer (автоочистка)
            "working": 10,      # Максимум 10 элементов
            "short_term": 100,  # Максимум 100 элементов
            "episodic": 500,    # Максимум 500 элементов
            "semantic": 10000,  # Максимум 10000 элементов
            "graph": 5000,      # Максимум 5000 узлов
            "procedural": 1000, # Максимум 1000 навыков
            "summary": 1000     # Максимум 1000 суммаризаций (TTL 180 дней)
        }
        
        # Веса для вычисления балла важности
        self.score_weights = {
            "importance": 0.4,      # 40% - важность контента
            "access_frequency": 0.3, # 30% - частота использования
            "recency": 0.2,         # 20% - свежесть
            "emotional": 0.1        # 10% - эмоциональная значимость
        }
        
        # Минимальные пороги для удаления
        self.removal_thresholds = {
            "working": 0.2,      # Низкий порог для рабочей памяти
            "short_term": 0.3,   # Средний порог для краткосрочной
            "episodic": 0.4,     # Высокий порог для эпизодической
            "semantic": 0.5,     # Очень высокий порог для семантической
            "graph": 0.3,        # Средний порог для графа
            "procedural": 0.4    # Высокий порог для процедурной
        }
        
        # Настройки дедупликации
        self.deduplication_config = {
            "enabled": True,
            "similarity_threshold": 0.6,   # Порог схожести для эмбеддингов (оптимизированный)
            "text_normalization": True,    # Нормализация текста
            "use_hybrid_approach": True,   # Гибридный подход (эмбеддинги + метаданные)
            "use_embeddings": True,        # Использовать эмбеддинги для семантического сравнения
            "fallback_threshold": 0.3,     # Порог для fallback к простому сравнению
            "cache_enabled": True,         # Кэширование результатов проверки дублей
            "cache_ttl_seconds": 3600,     # TTL кэша - 1 час
            "batch_processing": True,      # Батчевая обработка
            "batch_size": 10,              # Размер батча для обработки
            "low_importance_working_only": True,  # Для низкой важности проверять только Working Memory
            "low_importance_threshold": 0.3  # Порог низкой важности
        }
        
        # Кэш для результатов проверки дублей
        self._duplicate_cache = {}
        self._cache_timestamps = {}
        
        # Метрики дедупликации
        self._deduplication_metrics = {
            "total_checks": 0,
            "cache_hits": 0,
            "duplicates_detected": 0,
            "duplicates_removed": 0,
            "batch_operations": 0,
            "working_only_checks": 0,
            "embedding_calculations": 0,
            "fallback_calculations": 0
        }
        
        # Русские стоп-слова для дедупликации
        self.russian_stop_words = {
            "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда", "конечно", "всю", "между"
        }
    
    async def consolidate_all_memory(self, user_id: str) -> Dict[str, Any]:
        """
        Консолидация всех уровней памяти
        
        Args:
            user_id: ID пользователя для консолидации памяти
        """
        logger.info(f"Starting comprehensive memory consolidation for user {user_id}...")
        start_time = time.time()
        results = {}
        
        try:
            # Консолидируем каждый уровень
            for level_name in self.consolidation_thresholds.keys():
                logger.info(f"Consolidating {level_name} memory for user {user_id}...")
                level_result = await self.consolidate_memory_level(level_name, user_id)
                results[level_name] = level_result
                logger.info(f"Consolidation result for {level_name}: {level_result}")
                
                # Небольшая пауза между уровнями
                await asyncio.sleep(0.1)
            
            # Обновляем метрики
            total_removed = sum(r.get("removed_count", 0) for r in results.values())
            total_consolidated = sum(r.get("consolidated_count", 0) for r in results.values())
            
            try:
                inc("memory_consolidation_total")
                set_gauge("memory_consolidation_removed_items", total_removed)
                set_gauge("memory_consolidation_consolidated_items", total_consolidated)
            except Exception as e:
                logger.warning(f"Failed to update consolidation metrics: {e}")
            
            processing_time = time.time() - start_time
            logger.info(f"Memory consolidation completed in {processing_time:.2f}s")
            logger.info(f"Removed {total_removed} items, consolidated {total_consolidated} items")
            
            return {
                "success": True,
                "processing_time": processing_time,
                "total_removed": total_removed,
                "total_consolidated": total_consolidated,
                "level_results": results
            }
            
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def consolidate_memory_level(self, level_name: str, user_id: str) -> Dict[str, Any]:
        """
        Консолидация конкретного уровня памяти
        """
        try:
            # Получаем менеджер уровня памяти
            memory_manager = self._get_memory_manager(level_name)
            if not memory_manager:
                return {"error": f"Memory manager not found for level: {level_name}"}
            
            # Получаем все элементы уровня для указанного пользователя
            all_items = await self._get_all_items(memory_manager, level_name, user_id)
            logger.info(f"Consolidation for {level_name}: found {len(all_items)} items for user {user_id}")
            if not all_items:
                logger.info(f"No items found for consolidation in {level_name} for user {user_id}")
                return {"removed_count": 0, "consolidated_count": 0, "items_analyzed": 0}
            
            # Вычисляем баллы важности для всех элементов
            scored_items = []
            for item in all_items:
                score_data = self._calculate_memory_score(item, level_name)
                scored_items.append(score_data)
            
            # Сортируем по баллу (убывание)
            scored_items.sort(key=lambda x: x.score, reverse=True)
            
            # Определяем элементы для удаления и консолидации
            target_size = self.consolidation_thresholds[level_name]
            removal_threshold = self.removal_thresholds[level_name]
            
            items_to_keep = scored_items[:target_size]
            items_to_remove = []
            items_to_consolidate = []
            
            # Анализируем элементы сверх лимита
            for item_score in scored_items[target_size:]:
                if item_score.score < removal_threshold:
                    items_to_remove.append(item_score)
                else:
                    # Попробуем консолидировать с похожими элементами
                    items_to_consolidate.append(item_score)
            
            # Выполняем удаление (bulk операция)
            removed_count = 0
            if items_to_remove:
                try:
                    item_ids_to_remove = [item.item_id for item in items_to_remove]
                    removed_count = await self._remove_memory_items(memory_manager, level_name, item_ids_to_remove)
                    logger.debug(f"Removed {removed_count} {level_name} items (low scores: {[item.score for item in items_to_remove[:5]]})")
                except Exception as e:
                    logger.warning(f"Failed to remove {level_name} items: {e}")
            
            # Выполняем консолидацию
            consolidated_count = 0
            if items_to_consolidate:
                consolidated_count = await self._consolidate_similar_items(
                    memory_manager, level_name, items_to_consolidate, items_to_keep
                )
            
            # Записываем событие
            try:
                record_event("memory_consolidation_level", {
                    "level": level_name,
                    "items_analyzed": len(all_items),
                    "items_removed": removed_count,
                    "items_consolidated": consolidated_count,
                    "target_size": target_size
                })
            except Exception:
                pass
            
            return {
                "removed_count": removed_count,
                "consolidated_count": consolidated_count,
                "items_analyzed": len(all_items),
                "target_size": target_size,
                "removal_threshold": removal_threshold
            }
            
        except Exception as e:
            logger.error(f"Error consolidating {level_name} memory: {e}")
            return {"error": str(e)}
    
    def _calculate_memory_score(self, item: Any, level_name: str) -> MemoryItemScore:
        """
        Вычисляет балл важности для элемента памяти
        """
        now = datetime.now()
        
        # Базовые атрибуты
        importance = getattr(item, 'importance', 0.5)
        access_count = getattr(item, 'access_count', 0)
        last_accessed = getattr(item, 'last_accessed', None)
        created_at = getattr(item, 'created_at', getattr(item, 'timestamp', now))
        
        # Временные факторы
        age_days = (now - created_at).days
        recency_days = (now - last_accessed).days if last_accessed else age_days
        
        # Эмоциональные факторы
        emotional_weight = 1.0
        emotion_data = getattr(item, 'emotion_data', None)
        if emotion_data and isinstance(emotion_data, dict):
            confidence = emotion_data.get('primary_confidence', 0.5)
            emotional_weight = 0.5 + confidence  # 0.5-1.5
        
        # Вычисляем компоненты балла
        importance_score = importance * self.score_weights["importance"]
        access_score = min(1.0, access_count / 10) * self.score_weights["access_frequency"]
        recency_score = max(0.0, 1.0 - recency_days / 30) * self.score_weights["recency"]
        emotional_score = emotional_weight * self.score_weights["emotional"]
        
        # Итоговый балл
        total_score = importance_score + access_score + recency_score + emotional_score
        
        # Определяем причину консолидации
        consolidation_reason = self._get_consolidation_reason(
            total_score, importance, access_count, recency_days, emotional_weight
        )
        
        # Извлекаем содержимое для дедупликации
        content = getattr(item, 'content', None)
        if not content:
            # Пытаемся извлечь из метаданных
            metadata = getattr(item, 'metadata', {})
            if isinstance(metadata, dict):
                content = metadata.get('content', None)
        
        return MemoryItemScore(
            item_id=getattr(item, 'id', 'unknown'),
            item_type=level_name,
            score=total_score,
            importance=importance,
            access_count=access_count,
            last_accessed=last_accessed,
            age_days=age_days,
            recency_days=recency_days,
            emotional_weight=emotional_weight,
            consolidation_reason=consolidation_reason,
            content=content
        )
    
    def _get_consolidation_reason(self, score: float, importance: float, 
                                access_count: int, recency_days: int, 
                                emotional_weight: float) -> str:
        """Определяет причину консолидации элемента"""
        if score < 0.3:
            return "low_priority"
        elif access_count == 0:
            return "never_accessed"
        elif recency_days > 30:
            return "stale_data"
        elif importance < 0.3:
            return "low_importance"
        elif emotional_weight < 0.7:
            return "low_emotional_significance"
        else:
            return "capacity_limit"
    
    def _get_memory_manager(self, level_name: str):
        """Получает менеджер для уровня памяти"""
        managers = {
            "sensor": self.orchestrator.sensor_buffer,
            "working": self.orchestrator.working_memory,
            "short_term": self.orchestrator.short_term_memory,
            "episodic": self.orchestrator.episodic_memory,
            "semantic": self.orchestrator.semantic_memory,
            "graph": self.orchestrator.graph_memory,
            "procedural": self.orchestrator.procedural_memory,
            "summary": self.orchestrator.semantic_memory  # Суммаризации хранятся в semantic memory
        }
        return managers.get(level_name)
    
    async def _get_all_items(self, memory_manager, level_name: str, user_id: str) -> List[Any]:
        """Получает все элементы уровня памяти"""
        try:
            # Используем реальные методы memory managers
            from datetime import timedelta
            
            if level_name == "working":
                # Получаем ВСЕ элементы для консолидации
                all_items = await memory_manager.get_active_context(user_id, limit=1000)
                logger.info(f"Working memory consolidation: found {len(all_items)} total items for user {user_id}")
                return all_items
            elif level_name == "short_term":
                # Для short-term используем get_active_context если есть, иначе get_older_than с большим лимитом
                if hasattr(memory_manager, 'get_active_context'):
                    all_items = await memory_manager.get_active_context(user_id, limit=1000)
                else:
                    # Получаем все элементы (старше 0 дней = все)
                    all_items = await memory_manager.get_older_than(user_id, timedelta(days=0), limit=1000)
                logger.info(f"Short-term memory consolidation: found {len(all_items)} total items for user {user_id}")
                return all_items
            elif level_name == "episodic":
                # Для episodic используем get_older_than с большим лимитом
                all_items = await memory_manager.get_older_than(user_id, timedelta(days=0), limit=1000)
                logger.info(f"Episodic memory consolidation: found {len(all_items)} total items for user {user_id}")
                return all_items
            elif level_name == "semantic":
                # Для семантической памяти используем get_older_than с большим лимитом
                if hasattr(memory_manager, 'get_older_than'):
                    all_items = await memory_manager.get_older_than(user_id, timedelta(days=0), limit=1000)
                    logger.info(f"Semantic memory consolidation: found {len(all_items)} total items for user {user_id}")
                    return all_items
                else:
                    # Fallback - получаем через поиск
                    logger.warning(f"Semantic memory manager has no get_older_than method")
                    return []
            elif level_name == "graph":
                # Для графа получаем все узлы через поиск
                if hasattr(memory_manager, 'get_all_nodes'):
                    all_nodes = await memory_manager.get_all_nodes(user_id)
                    logger.info(f"Graph memory consolidation: found {len(all_nodes)} nodes for user {user_id}")
                    return all_nodes
                else:
                    logger.warning(f"Graph memory manager has no get_all_nodes method")
                    return []
            elif level_name == "procedural":
                # Для процедурной памяти получаем все навыки
                if hasattr(memory_manager, 'get_all_skills'):
                    all_skills = await memory_manager.get_all_skills(user_id)
                    logger.info(f"Procedural memory consolidation: found {len(all_skills)} skills for user {user_id}")
                    return all_skills
                else:
                    logger.warning(f"Procedural memory manager has no get_all_skills method")
                    return []
            elif level_name == "summary":
                # Для суммаризаций получаем только элементы с type="summary" и проверяем TTL
                if hasattr(memory_manager, 'get_older_than'):
                    all_items = await memory_manager.get_older_than(user_id, timedelta(days=0), limit=1000)
                    # Фильтруем только суммаризации
                    summary_items = []
                    for item in all_items:
                        metadata = getattr(item, 'metadata', {})
                        if isinstance(metadata, dict) and metadata.get('type') == 'summary':
                            # Проверяем TTL (180 дней)
                            created_at = metadata.get('created_at')
                            if created_at:
                                try:
                                    from datetime import datetime
                                    created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                    age_days = (datetime.now() - created_date).days
                                    if age_days > 180:  # TTL истек
                                        summary_items.append(item)
                                except Exception as e:
                                    logger.warning(f"Failed to parse created_at for summary item: {e}")
                            else:
                                # Если нет даты создания, считаем старым
                                summary_items.append(item)
                    
                    logger.info(f"Summary memory consolidation: found {len(summary_items)} expired summaries for user {user_id}")
                    return summary_items
                else:
                    logger.warning(f"Summary memory manager has no get_older_than method")
                    return []
            else:
                logger.warning(f"Unknown memory level: {level_name}")
                return []
        except Exception as e:
            logger.warning(f"Failed to get items for {level_name}: {e}")
            return []
    
    async def _remove_memory_items(self, memory_manager, level_name: str, item_ids: List[str]):
        """Удаляет элементы памяти используя bulk_delete"""
        try:
            if not item_ids:
                return
            
            # Используем bulk_delete для эффективного удаления
            if hasattr(memory_manager, 'bulk_delete'):
                deleted_count = await memory_manager.bulk_delete(item_ids)
                logger.debug(f"Successfully deleted {deleted_count} {level_name} items via bulk_delete")
                return deleted_count
            else:
                # Fallback к индивидуальному удалению
                deleted_count = 0
                for item_id in item_ids:
                    try:
                        if level_name == "working" and hasattr(memory_manager, 'delete_memory'):
                            await memory_manager.delete_memory(item_id)
                        elif level_name == "short_term" and hasattr(memory_manager, 'delete_event'):
                            await memory_manager.delete_event(item_id)
                        elif level_name == "episodic" and hasattr(memory_manager, 'delete_experience'):
                            await memory_manager.delete_experience(item_id)
                        elif level_name == "semantic" and hasattr(memory_manager, 'delete_knowledge'):
                            await memory_manager.delete_knowledge(item_id)
                        elif level_name == "graph" and hasattr(memory_manager, 'delete_node'):
                            await memory_manager.delete_node(item_id)
                        elif level_name == "procedural" and hasattr(memory_manager, 'delete_skill'):
                            await memory_manager.delete_skill(item_id)
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {level_name} item {item_id}: {e}")
                
                logger.debug(f"Successfully deleted {deleted_count} {level_name} items individually")
                return deleted_count
        except Exception as e:
            logger.warning(f"Failed to remove {level_name} items: {e}")
            raise
    
    async def _consolidate_similar_items(self, memory_manager, level_name: str, 
                                       items_to_consolidate: List[MemoryItemScore],
                                       items_to_keep: List[MemoryItemScore]) -> int:
        """
        Консолидирует похожие элементы памяти с отслеживанием метрик
        """
        consolidated_count = 0
        duplicates_found = 0
        duplicates_removed = 0
        
        try:
            # Группируем похожие элементы по содержимому
            similar_groups = await self._group_similar_items(items_to_consolidate)
            
            for group in similar_groups:
                if len(group) > 1:
                    duplicates_found += len(group)
                    
                    # Находим лучший элемент в группе для сохранения
                    best_item = max(group, key=lambda x: x.score)
                    
                    # Объединяем информацию из других элементов
                    consolidated_content = await self._merge_item_content(group, level_name)
                    
                    # Обновляем лучший элемент объединенным содержимым
                    await self._update_item_content(memory_manager, level_name, 
                                                  best_item.item_id, consolidated_content)
                    
                    # Удаляем остальные элементы группы
                    items_to_remove = [item for item in group if item.item_id != best_item.item_id]
                    if items_to_remove:
                        item_ids_to_remove = [item.item_id for item in items_to_remove]
                        removed_count = await self._remove_memory_items(memory_manager, level_name, item_ids_to_remove)
                        consolidated_count += removed_count
                        duplicates_removed += removed_count
                    
                    logger.debug(f"Consolidated {len(group)} {level_name} items into {best_item.item_id}")
            
            # Отслеживаем метрики дедупликации
            if duplicates_found > 0:
                await self._track_deduplication_metrics(duplicates_found, duplicates_removed)
        
        except Exception as e:
            logger.warning(f"Error during consolidation of {level_name}: {e}")
        
        return consolidated_count
    
    async def _group_similar_items(self, items: List[MemoryItemScore]) -> List[List[MemoryItemScore]]:
        """Группирует похожие элементы для консолидации с улучшенной дедупликацией"""
        if not self.deduplication_config["enabled"]:
            # Fallback к старой логике если дедупликация отключена
            return await self._group_similar_items_legacy(items)
        
        groups = []
        processed = set()
        
        for item in items:
            if item.item_id in processed:
                continue
            
            # Находим похожие элементы с гибридным подходом
            similar_items = [item]
            
            for other_item in items:
                if (other_item.item_id not in processed and 
                    other_item.item_id != item.item_id and
                    await self._is_similar_hybrid(item, other_item)):
                    similar_items.append(other_item)
                    processed.add(other_item.item_id)
            
            processed.add(item.item_id)
            if len(similar_items) > 1:
                groups.append(similar_items)
                logger.debug(f"Grouped {len(similar_items)} similar items for consolidation")
        
        return groups
    
    async def _group_similar_items_legacy(self, items: List[MemoryItemScore]) -> List[List[MemoryItemScore]]:
        """Старая логика группировки (fallback)"""
        groups = []
        processed = set()
        
        for item in items:
            if item.item_id in processed:
                continue
            
            # Находим похожие элементы по важности и времени
            similar_items = [item]
            
            for other_item in items:
                if (other_item.item_id not in processed and 
                    other_item.item_id != item.item_id and
                    abs(other_item.importance - item.importance) < 0.2 and
                    abs(other_item.age_days - item.age_days) < 7):  # Похожий возраст
                    similar_items.append(other_item)
                    processed.add(other_item.item_id)
            
            processed.add(item.item_id)
            if len(similar_items) > 1:
                groups.append(similar_items)
        
        return groups
    
    async def _is_similar_hybrid(self, item1: MemoryItemScore, item2: MemoryItemScore) -> bool:
        """Гибридное определение схожести элементов"""
        if not self.deduplication_config["use_hybrid_approach"]:
            # Fallback к метаданным только
            return self._is_similar_metadata(item1, item2)
        
        # 1. Текстовое сравнение (приоритет)
        if hasattr(item1, 'content') and hasattr(item2, 'content'):
            if await self._is_duplicate_text(item1.content, item2.content):
                return True
        
        # 2. Метаданные (fallback)
        return self._is_similar_metadata(item1, item2)
    
    async def _is_duplicate_text(self, text1: str, text2: str) -> bool:
        """Проверка на текстовые дубли с использованием эмбеддингов и кэширования"""
        if not text1 or not text2:
            return False
        
        # Отслеживаем общее количество проверок
        self._track_deduplication_metrics("total_checks")
        
        # Проверяем кэш
        cache_key = self._get_cache_key(text1, text2)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Нормализация текста
        if self.deduplication_config["text_normalization"]:
            text1_norm = self._normalize_text(text1)
            text2_norm = self._normalize_text(text2)
        else:
            text1_norm = text1.lower().strip()
            text2_norm = text2.lower().strip()
        
        # Точное совпадение
        if text1_norm == text2_norm:
            self._cache_result(cache_key, True)
            return True
        
        # Используем эмбеддинги для семантического сравнения
        is_duplicate = False
        if self.deduplication_config["use_embeddings"]:
            try:
                similarity = await self._calculate_embedding_similarity(text1_norm, text2_norm)
                self._track_deduplication_metrics("embedding_calculations")
                threshold = self.deduplication_config["similarity_threshold"]
                is_duplicate = similarity > threshold
            except Exception as e:
                logger.warning(f"Ошибка при вычислении схожести эмбеддингов: {e}")
                # Fallback к простому текстовому сравнению
                fallback_threshold = self.deduplication_config["fallback_threshold"]
                is_duplicate = self._calculate_simple_text_similarity(text1_norm, text2_norm) > fallback_threshold
                self._track_deduplication_metrics("fallback_calculations")
        else:
            # Используем только простое текстовое сравнение
            fallback_threshold = self.deduplication_config["fallback_threshold"]
            is_duplicate = self._calculate_simple_text_similarity(text1_norm, text2_norm) > fallback_threshold
            self._track_deduplication_metrics("fallback_calculations")
        
        # Кэшируем результат
        self._cache_result(cache_key, is_duplicate)
        
        return is_duplicate
    
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста для сравнения"""
        if not text:
            return ""
        
        # Убираем пунктуацию, приводим к нижнему регистру
        normalized = re.sub(r'[^\w\s]', ' ', text.lower())
        # Убираем лишние пробелы
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Расширенное сравнение текстов с учетом русских особенностей"""
        if not text1 or not text2:
            return 0.0
        
        # Разбиваем на слова и убираем стоп-слова
        words1 = set(text1.split()) - self.russian_stop_words
        words2 = set(text2.split()) - self.russian_stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        base_similarity = intersection / union if union > 0 else 0.0
        
        # Бонус за длину (короткие тексты должны быть более похожими)
        length_bonus = min(0.1, 0.1 * (1 - abs(len(text1) - len(text2)) / max(len(text1), len(text2))))
        
        # Бонус за семантические синонимы (морфологический анализ)
        synonym_bonus = self._calculate_synonym_similarity(words1, words2)
        
        # Бонус за контекстную схожесть (для коротких фраз)
        context_bonus = 0.0
        if self.deduplication_config["use_context_similarity"]:
            context_bonus = self._calculate_context_similarity(text1, text2)
        
        return min(1.0, base_similarity + length_bonus + synonym_bonus + context_bonus)
    
    def _get_cache_key(self, text1: str, text2: str) -> str:
        """Генерация ключа кэша для пары текстов"""
        # Сортируем тексты для обеспечения консистентности
        sorted_texts = tuple(sorted([text1, text2]))
        return f"dup_check:{hash(sorted_texts[0])}:{hash(sorted_texts[1])}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Проверка валидности кэша"""
        if not self.deduplication_config["cache_enabled"]:
            return False
        
        if cache_key not in self._cache_timestamps:
            return False
        
        current_time = time.time()
        cache_time = self._cache_timestamps[cache_key]
        ttl = self.deduplication_config["cache_ttl_seconds"]
        
        return (current_time - cache_time) < ttl
    
    def _get_cached_result(self, cache_key: str) -> Optional[bool]:
        """Получение результата из кэша"""
        if self._is_cache_valid(cache_key):
            self._deduplication_metrics["cache_hits"] += 1
            return self._duplicate_cache.get(cache_key)
        return None
    
    def _cache_result(self, cache_key: str, is_duplicate: bool):
        """Сохранение результата в кэш"""
        if not self.deduplication_config["cache_enabled"]:
            return
        
        self._duplicate_cache[cache_key] = is_duplicate
        self._cache_timestamps[cache_key] = time.time()
        
        # Очистка старых записей кэша
        self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Очистка устаревших записей кэша"""
        current_time = time.time()
        ttl = self.deduplication_config["cache_ttl_seconds"]
        
        expired_keys = []
        for key, timestamp in self._cache_timestamps.items():
            if (current_time - timestamp) >= ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._duplicate_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    def _should_check_working_only(self, importance: float) -> bool:
        """Определение, нужно ли проверять только Working Memory для низкой важности"""
        if not self.deduplication_config["low_importance_working_only"]:
            return False
        
        return importance <= self.deduplication_config["low_importance_threshold"]
    
    def _track_deduplication_metrics(self, operation: str, count: int = 1):
        """Отслеживание метрик дедупликации"""
        if operation in self._deduplication_metrics:
            self._deduplication_metrics[operation] += count
        
        # Обновляем базовые метрики
        try:
            inc(f"deduplication_{operation}", count)
        except Exception as e:
            logger.warning(f"Failed to update deduplication metrics: {e}")
    
    def get_deduplication_metrics(self) -> Dict[str, Any]:
        """Получение метрик дедупликации"""
        total_checks = self._deduplication_metrics["total_checks"]
        cache_hits = self._deduplication_metrics["cache_hits"]
        
        return {
            "metrics": self._deduplication_metrics.copy(),
            "cache_hit_rate": cache_hits / total_checks if total_checks > 0 else 0.0,
            "duplication_efficiency": self._deduplication_metrics["duplicates_removed"] / max(1, self._deduplication_metrics["duplicates_detected"]),
            "cache_size": len(self._duplicate_cache),
            "config": self.deduplication_config.copy()
        }
    
    async def batch_check_duplicates(self, items: List[MemoryItemScore], importance: float) -> List[Tuple[MemoryItemScore, List[MemoryItemScore]]]:
        """Батчевая проверка дублей с оптимизацией для низкой важности"""
        if not self.deduplication_config["batch_processing"]:
            # Fallback к обычной обработке
            return await self._group_similar_items(items)
        
        # Определяем уровни для проверки
        if self._should_check_working_only(importance):
            # Для низкой важности проверяем только Working Memory
            self._track_deduplication_metrics("working_only_checks")
            levels_to_check = ["working"]
        else:
            # Для высокой важности проверяем все уровни
            levels_to_check = ["working", "short_term", "episodic", "semantic"]
        
        # Группируем элементы по батчам
        batch_size = self.deduplication_config["batch_size"]
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        all_duplicates = []
        
        for batch in batches:
            batch_duplicates = await self._process_batch_duplicates(batch, levels_to_check)
            all_duplicates.extend(batch_duplicates)
            self._track_deduplication_metrics("batch_operations")
        
        return all_duplicates
    
    async def _process_batch_duplicates(self, batch: List[MemoryItemScore], levels_to_check: List[str]) -> List[Tuple[MemoryItemScore, List[MemoryItemScore]]]:
        """Обработка батча элементов на дубли"""
        duplicates = []
        processed = set()
        
        for i, item1 in enumerate(batch):
            if item1.item_id in processed:
                continue
            
            similar_items = [item1]
            
            for j, item2 in enumerate(batch[i+1:], i+1):
                if item2.item_id in processed:
                    continue
                
                # Проверяем дубли только для указанных уровней
                if item1.item_type in levels_to_check and item2.item_type in levels_to_check:
                    if await self._is_similar_hybrid(item1, item2):
                        similar_items.append(item2)
                        processed.add(item2.item_id)
                        self._track_deduplication_metrics("duplicates_detected")
            
            if len(similar_items) > 1:
                duplicates.append((item1, similar_items[1:]))
                processed.add(item1.item_id)
        
        return duplicates
    
    def _calculate_synonym_similarity(self, words1: set, words2: set) -> float:
        """Вычисляет схожесть на основе морфологического анализа"""
        # Вместо словаря синонимов используем более умные алгоритмы
        
        # 1. Морфологическое сходство (общие корни)
        morphological_bonus = self._calculate_morphological_similarity(words1, words2)
        
        # 2. Длина слов (короткие слова чаще похожи)
        length_bonus = self._calculate_length_similarity(words1, words2)
        
        # 3. Частичная схожесть (общие подстроки)
        substring_bonus = self._calculate_substring_similarity(words1, words2)
        
        return min(0.4, morphological_bonus + length_bonus + substring_bonus)
    
    def _calculate_morphological_similarity(self, words1: set, words2: set) -> float:
        """Морфологическое сходство - общие корни слов"""
        bonus = 0.0
        
        for word1 in words1:
            for word2 in words2:
                if word1 == word2:
                    continue
                
                # Ищем общие корни (первые 3-4 символа)
                min_len = min(len(word1), len(word2))
                if min_len >= 3:
                    common_prefix = 0
                    for i in range(min(4, min_len)):
                        if word1[i] == word2[i]:
                            common_prefix += 1
                        else:
                            break
                    
                    if common_prefix >= 3:
                        bonus += 0.1  # Бонус за общий корень
        
        return min(0.2, bonus)
    
    def _calculate_length_similarity(self, words1: set, words2: set) -> float:
        """Бонус за схожую длину слов"""
        if not words1 or not words2:
            return 0.0
        
        avg_len1 = sum(len(word) for word in words1) / len(words1)
        avg_len2 = sum(len(word) for word in words2) / len(words2)
        
        # Бонус если средняя длина слов схожа
        length_diff = abs(avg_len1 - avg_len2)
        if length_diff <= 1:
            return 0.1
        elif length_diff <= 2:
            return 0.05
        
        return 0.0
    
    def _calculate_substring_similarity(self, words1: set, words2: set) -> float:
        """Схожесть на основе общих подстрок"""
        bonus = 0.0
        
        for word1 in words1:
            for word2 in words2:
                if word1 == word2:
                    continue
                
                # Ищем общие подстроки длиной 3+ символов
                max_common = 0
                for i in range(len(word1) - 2):
                    for j in range(len(word2) - 2):
                        common_len = 0
                        while (i + common_len < len(word1) and 
                               j + common_len < len(word2) and 
                               word1[i + common_len] == word2[j + common_len]):
                            common_len += 1
                        
                        if common_len >= 3:
                            max_common = max(max_common, common_len)
                
                if max_common >= 3:
                    bonus += 0.05 * (max_common / max(len(word1), len(word2)))
        
        return min(0.1, bonus)
    
    def _calculate_context_similarity(self, text1: str, text2: str) -> float:
        """Контекстная схожесть для коротких фраз"""
        # Для коротких фраз (1-3 слова) используем специальную логику
        words1 = text1.split()
        words2 = text2.split()
        
        if len(words1) > 3 or len(words2) > 3:
            return 0.0  # Только для коротких фраз
        
        # Специальные случаи для приветствий и вопросов
        greeting_patterns = [
            (["привет"], ["здравствуй", "приветик", "салют"]),
            (["здравствуй"], ["привет", "приветик", "салют"]),
            (["как", "дела"], ["как", "поживаешь"]),
            (["как", "поживаешь"], ["как", "дела"]),
            (["что", "нового"], ["что", "слышно"]),
            (["что", "слышно"], ["что", "нового"])
        ]
        
        for pattern1, pattern2 in greeting_patterns:
            if (all(word in words1 for word in pattern1) and 
                all(word in words2 for word in pattern2)):
                return 0.4  # Высокий бонус за контекстную схожесть
            
            if (all(word in words2 for word in pattern1) and 
                all(word in words1 for word in pattern2)):
                return 0.4  # Высокий бонус за контекстную схожесть
        
        return 0.0
    
    async def _calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """Вычисление схожести с использованием эмбеддингов"""
        try:
            # Получаем провайдер эмбеддингов из orchestrator
            if not hasattr(self.orchestrator, 'embedder') or not self.orchestrator.embedder:
                logger.warning("Embedder не доступен, используем fallback")
                return self._calculate_simple_text_similarity(text1, text2)
            
            # Генерируем эмбеддинги
            embedding1 = await self.orchestrator.embedder.generate_embedding(text1)
            embedding2 = await self.orchestrator.embedder.generate_embedding(text2)
            
            # Вычисляем косинусное сходство
            similarity = self._cosine_similarity(embedding1, embedding2)
            
            logger.debug(f"Embedding similarity: '{text1[:30]}...' vs '{text2[:30]}...' = {similarity:.3f}")
            return similarity
            
        except Exception as e:
            logger.warning(f"Ошибка при вычислении схожести эмбеддингов: {e}")
            return self._calculate_simple_text_similarity(text1, text2)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Вычисление косинусного сходства между векторами"""
        import numpy as np
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Нормализуем векторы
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Косинусное сходство
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)
    
    def _calculate_simple_text_similarity(self, text1: str, text2: str) -> float:
        """Простое текстовое сравнение как fallback"""
        if not text1 or not text2:
            return 0.0
        
        # Разбиваем на слова и убираем стоп-слова
        words1 = set(text1.split()) - self.russian_stop_words
        words2 = set(text2.split()) - self.russian_stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    
    async def _track_deduplication_metrics(self, duplicates_found: int, duplicates_removed: int):
        """Отслеживание эффективности дедупликации"""
        try:
            inc("duplicates_detected_total", duplicates_found)
            inc("duplicates_removed_total", duplicates_removed)
            if duplicates_found > 0:
                efficiency = duplicates_removed / duplicates_found
                set_gauge("duplication_efficiency", efficiency)
                logger.debug(f"Deduplication metrics: found={duplicates_found}, removed={duplicates_removed}, efficiency={efficiency:.2f}")
        except Exception as e:
            logger.warning(f"Failed to update deduplication metrics: {e}")
    
    def _is_similar_metadata(self, item1: MemoryItemScore, item2: MemoryItemScore) -> bool:
        """Сравнение по метаданным (старая логика)"""
        return (abs(item1.importance - item2.importance) < 0.2 and
                abs(item1.age_days - item2.age_days) < 7)
    
    async def _merge_item_content(self, items: List[MemoryItemScore], level_name: str) -> Dict[str, Any]:
        """Объединяет содержимое похожих элементов"""
        try:
            # Простое объединение метаданных без LLM
            # В будущем можно добавить семантическое объединение через orchestrator
            return {
                "importance": max(item.importance for item in items),
                "access_count": sum(item.access_count for item in items),
                "emotional_weight": max(item.emotional_weight for item in items),
                "source_count": len(items),
                "consolidated": True,
                "consolidation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Content merging failed: {e}")
            # Базовый fallback
            return {
                "importance": max(item.importance for item in items),
                "access_count": sum(item.access_count for item in items),
                "emotional_weight": max(item.emotional_weight for item in items)
            }
    
    async def _update_item_content(self, memory_manager, level_name: str, 
                                 item_id: str, content: Dict[str, Any]):
        """Обновляет содержимое элемента памяти"""
        try:
            # Пока что только логируем обновление
            # В будущем можно добавить реальное обновление через memory managers
            logger.debug(f"Updated {level_name} item {item_id} with consolidated content: {content}")
                
        except Exception as e:
            logger.warning(f"Failed to update {level_name} item {item_id}: {e}")
    
    async def get_consolidation_stats(self) -> Dict[str, Any]:
        """Получает статистику консолидации"""
        stats = {}
        
        for level_name in self.consolidation_thresholds.keys():
            try:
                memory_manager = self._get_memory_manager(level_name)
                if memory_manager:
                    items = await self._get_all_items(memory_manager, level_name)
                    stats[level_name] = {
                        "current_size": len(items),
                        "target_size": self.consolidation_thresholds[level_name],
                        "removal_threshold": self.removal_thresholds[level_name],
                        "needs_consolidation": len(items) > self.consolidation_thresholds[level_name]
                    }
            except Exception as e:
                stats[level_name] = {"error": str(e)}
        
        return stats
    
    async def cleanup_old_memories(self, user_id: str, days: int = 365) -> Dict[str, Any]:
        """
        Очистка старых воспоминаний через memory managers
        
        Args:
            days: Количество дней для удержания воспоминаний
            user_id: ID пользователя для очистки воспоминаний
            
        Returns:
            Результаты очистки
        """
        try:
            logger.info(f"Starting cleanup of memories older than {days} days for user {user_id}")
            
            total_deleted = 0
            from datetime import timedelta
            
            # Очищаем каждый уровень памяти
            for level_name in self.consolidation_thresholds.keys():
                try:
                    memory_manager = self._get_memory_manager(level_name)
                    if memory_manager and hasattr(memory_manager, 'get_older_than'):
                        # Получаем старые элементы для указанного пользователя
                        old_items = await memory_manager.get_older_than(
                            user_id=user_id,
                            delta=timedelta(days=days),
                            limit=1000
                        )
                        
                        if old_items:
                            # Удаляем старые элементы
                            item_ids = [item.id for item in old_items]
                            deleted_count = await self._remove_memory_items(memory_manager, level_name, item_ids)
                            total_deleted += deleted_count
                            logger.info(f"Cleaned {deleted_count} old items from {level_name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to cleanup {level_name}: {e}")
            
            # Обновляем метрики
            try:
                inc("memory_cleanup_total")
                set_gauge("memory_cleanup_deleted_items", total_deleted)
            except Exception as e:
                logger.warning(f"Failed to update cleanup metrics: {e}")
            
            logger.info(f"Cleanup completed: {total_deleted} old memories deleted")
            
            return {
                "success": True,
                "deleted_count": total_deleted,
                "retention_days": days
            }
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            return {
                "success": False,
                "error": str(e)
            }
