"""
Contextual Search Engine для AIRI Memory System
Контекстный поиск с фильтрацией по эмоциям, времени и важности
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from loguru import logger

from .fts5_search import get_fts5_engine
from .hybrid_search import HybridSearchEngine
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..memory_levels.memory_orchestrator import MemoryOrchestrator

@dataclass
class EmotionFilter:
    """Фильтр по эмоциональным данным"""
    primary_emotions: Optional[List[str]] = None
    secondary_emotions: Optional[List[str]] = None
    min_confidence: float = 0.25
    max_confidence: float = 1.0
    sentiment: Optional[str] = None  # "positive", "negative", "neutral"
    consistency: Optional[str] = None  # "high", "medium", "low"

@dataclass
class TimeFilter:
    """Фильтр по временным меткам"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    days_ago: Optional[int] = None
    hours_ago: Optional[int] = None

@dataclass
class ImportanceFilter:
    """Фильтр по важности"""
    min_importance: float = 0.0
    max_importance: float = 1.0
    importance_range: Optional[Tuple[float, float]] = None

@dataclass
class ContextualSearchQuery:
    """Контекстный поисковый запрос"""
    query: str
    user_id: str
    emotion_filter: Optional[EmotionFilter] = None
    time_filter: Optional[TimeFilter] = None
    importance_filter: Optional[ImportanceFilter] = None
    memory_types: Optional[List[str]] = None
    limit: int = 10
    offset: int = 0
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

class ContextualSearchEngine:
    """Контекстный поисковый движок с фильтрацией"""
    
    def __init__(self, memory_orchestrator: Optional["MemoryOrchestrator"] = None):
        """
        Initialize contextual search engine
        
        Args:
            memory_orchestrator: MemoryOrchestrator instance for semantic search
        """
        self.memory_orchestrator = memory_orchestrator
        self._lock = asyncio.Lock()
        logger.info("ContextualSearchEngine initialized")
    
    async def search(self, search_query: ContextualSearchQuery) -> List[Dict[str, Any]]:
        """
        Выполнить контекстный поиск с фильтрацией
        
        Args:
            search_query: Контекстный поисковый запрос
            
        Returns:
            Список отфильтрованных результатов
        """
        try:
            async with self._lock:
                start_time = time.time()
                logger.info(f"Contextual search: query='{search_query.query}', user_id='{search_query.user_id}'")
                
                # 1. Получаем базовые результаты через гибридный поиск
                from . import get_hybrid_engine_lazy
                get_hybrid_engine = get_hybrid_engine_lazy()
                hybrid_engine = await get_hybrid_engine(self.memory_orchestrator)
                
                # Выполняем гибридный поиск
                base_results = await hybrid_engine.search(
                    query=search_query.query,
                    user_id=search_query.user_id,
                    limit=search_query.limit * 3,  # Получаем больше результатов для фильтрации
                    memory_types=search_query.memory_types,
                    semantic_weight=search_query.semantic_weight,
                    keyword_weight=search_query.keyword_weight
                )
                
                # 2. Применяем контекстные фильтры
                filtered_results = await self._apply_contextual_filters(
                    base_results, search_query
                )
                
                # 3. Применяем контекстное ранжирование
                ranked_results = await self._apply_contextual_ranking(
                    filtered_results, search_query
                )
                
                # 4. Ограничиваем результаты
                final_results = ranked_results[search_query.offset:search_query.offset + search_query.limit]
                
                processing_time = time.time() - start_time
                logger.info(f"Contextual search completed: {len(final_results)} results in {processing_time:.3f}s")
                
                return final_results
                
        except Exception as e:
            logger.error(f"ContextualSearchEngine.search failed: {e}")
            return []
    
    async def _apply_contextual_filters(self, results: List[Dict[str, Any]], 
                                      search_query: ContextualSearchQuery) -> List[Dict[str, Any]]:
        """Применить контекстные фильтры к результатам"""
        filtered_results = []
        
        for result in results:
            # Проверяем эмоциональный фильтр
            if search_query.emotion_filter and not self._matches_emotion_filter(result, search_query.emotion_filter):
                continue
            
            # Проверяем временной фильтр
            if search_query.time_filter and not self._matches_time_filter(result, search_query.time_filter):
                continue
            
            # Проверяем фильтр важности
            if search_query.importance_filter and not self._matches_importance_filter(result, search_query.importance_filter):
                continue
            
            filtered_results.append(result)
        
        logger.debug(f"Contextual filtering: {len(results)} -> {len(filtered_results)} results")
        return filtered_results
    
    def _matches_emotion_filter(self, result: Dict[str, Any], emotion_filter: EmotionFilter) -> bool:
        """Проверить соответствие эмоциональному фильтру"""
        try:
            emotion_data = result.get("emotion_data")
            if not emotion_data or not isinstance(emotion_data, dict):
                return True  # Если нет эмоциональных данных, пропускаем фильтр
            
            # Проверяем основную эмоцию
            if emotion_filter.primary_emotions:
                primary_emotion = emotion_data.get("primary_emotion", "")
                if primary_emotion not in emotion_filter.primary_emotions:
                    return False
            
            # Проверяем вторичную эмоцию
            if emotion_filter.secondary_emotions:
                secondary_emotion = emotion_data.get("secondary_emotion", "")
                if secondary_emotion not in emotion_filter.secondary_emotions:
                    return False
            
            # Проверяем уверенность
            confidence = emotion_data.get("primary_confidence", 0.0)
            if confidence < emotion_filter.min_confidence or confidence > emotion_filter.max_confidence:
                return False
            
            # Проверяем тональность
            if emotion_filter.sentiment:
                sentiment = self._determine_sentiment(emotion_data)
                if sentiment != emotion_filter.sentiment:
                    return False
            
            # Проверяем консистентность
            if emotion_filter.consistency:
                consistency = emotion_data.get("consistency", "medium")
                if consistency != emotion_filter.consistency:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error in emotion filter: {e}")
            return True  # В случае ошибки пропускаем фильтр
    
    def _matches_time_filter(self, result: Dict[str, Any], time_filter: TimeFilter) -> bool:
        """Проверить соответствие временному фильтру"""
        try:
            created_at = result.get("created_at")
            if not created_at:
                return True  # Если нет временной метки, пропускаем фильтр
            
            # Преобразуем в datetime если нужно
            if isinstance(created_at, (int, float)):
                result_time = datetime.fromtimestamp(created_at)
            elif isinstance(created_at, str):
                result_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                result_time = created_at
            
            now = datetime.now()
            
            # Проверяем диапазон дат
            if time_filter.start_date and result_time < time_filter.start_date:
                return False
            
            if time_filter.end_date and result_time > time_filter.end_date:
                return False
            
            # Проверяем дни назад
            if time_filter.days_ago:
                days_ago_date = now - timedelta(days=time_filter.days_ago)
                if result_time < days_ago_date:
                    return False
            
            # Проверяем часы назад
            if time_filter.hours_ago:
                hours_ago_date = now - timedelta(hours=time_filter.hours_ago)
                if result_time < hours_ago_date:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error in time filter: {e}")
            return True  # В случае ошибки пропускаем фильтр
    
    def _matches_importance_filter(self, result: Dict[str, Any], importance_filter: ImportanceFilter) -> bool:
        """Проверить соответствие фильтру важности"""
        try:
            importance = result.get("importance", 0.5)
            
            # Проверяем минимальную важность
            if importance < importance_filter.min_importance:
                return False
            
            # Проверяем максимальную важность
            if importance > importance_filter.max_importance:
                return False
            
            # Проверяем диапазон важности
            if importance_filter.importance_range:
                min_imp, max_imp = importance_filter.importance_range
                if importance < min_imp or importance > max_imp:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error in importance filter: {e}")
            return True  # В случае ошибки пропускаем фильтр
    
    async def _apply_contextual_ranking(self, results: List[Dict[str, Any]], 
                                      search_query: ContextualSearchQuery) -> List[Dict[str, Any]]:
        """Применить контекстное ранжирование к результатам"""
        try:
            # Вычисляем контекстные баллы для каждого результата
            for result in results:
                context_score = await self._calculate_context_score(result, search_query)
                result["context_score"] = context_score
            
            # Сортируем по контекстному баллу (убывание)
            ranked_results = sorted(results, key=lambda x: x.get("context_score", 0.0), reverse=True)
            
            logger.debug(f"Contextual ranking applied to {len(ranked_results)} results")
            return ranked_results
            
        except Exception as e:
            logger.warning(f"Error in contextual ranking: {e}")
            return results  # В случае ошибки возвращаем исходные результаты
    
    async def _calculate_context_score(self, result: Dict[str, Any], 
                                     search_query: ContextualSearchQuery) -> float:
        """Вычислить контекстный балл для результата"""
        try:
            base_score = result.get("hybrid_score", 0.0)
            context_bonus = 0.0
            
            # Бонус за эмоциональную релевантность
            if search_query.emotion_filter:
                emotion_bonus = self._calculate_emotion_bonus(result, search_query.emotion_filter)
                context_bonus += emotion_bonus * 0.3
            
            # Бонус за временную релевантность
            if search_query.time_filter:
                time_bonus = self._calculate_time_bonus(result, search_query.time_filter)
                context_bonus += time_bonus * 0.2
            
            # Бонус за важность
            if search_query.importance_filter:
                importance_bonus = self._calculate_importance_bonus(result, search_query.importance_filter)
                context_bonus += importance_bonus * 0.2
            
            # Бонус за тип памяти
            memory_type_bonus = self._calculate_memory_type_bonus(result)
            context_bonus += memory_type_bonus * 0.1
            
            final_score = base_score + context_bonus
            return min(1.0, max(0.0, final_score))  # Ограничиваем диапазон [0, 1]
            
        except Exception as e:
            logger.warning(f"Error calculating context score: {e}")
            return result.get("hybrid_score", 0.0)
    
    def _calculate_emotion_bonus(self, result: Dict[str, Any], emotion_filter: EmotionFilter) -> float:
        """Вычислить бонус за эмоциональную релевантность"""
        try:
            emotion_data = result.get("emotion_data")
            if not emotion_data:
                return 0.0
            
            bonus = 0.0
            
            # Бонус за соответствие основной эмоции
            if emotion_filter.primary_emotions:
                primary_emotion = emotion_data.get("primary_emotion", "")
                if primary_emotion in emotion_filter.primary_emotions:
                    confidence = emotion_data.get("primary_confidence", 0.0)
                    bonus += confidence * 0.5
            
            # Бонус за высокую уверенность
            confidence = emotion_data.get("primary_confidence", 0.0)
            if confidence > 0.8:
                bonus += 0.2
            
            return bonus
            
        except Exception as e:
            logger.warning(f"Error calculating emotion bonus: {e}")
            return 0.0
    
    def _calculate_time_bonus(self, result: Dict[str, Any], time_filter: TimeFilter) -> float:
        """Вычислить бонус за временную релевантность"""
        try:
            created_at = result.get("created_at")
            if not created_at:
                return 0.0
            
            # Преобразуем в datetime если нужно
            if isinstance(created_at, (int, float)):
                result_time = datetime.fromtimestamp(created_at)
            elif isinstance(created_at, str):
                result_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                result_time = created_at
            
            now = datetime.now()
            age_hours = (now - result_time).total_seconds() / 3600
            
            # Бонус за свежесть (больше бонус за более свежие результаты)
            if age_hours < 1:
                return 0.3  # Очень свежие
            elif age_hours < 24:
                return 0.2  # Свежие
            elif age_hours < 168:  # 7 дней
                return 0.1  # Недавние
            else:
                return 0.0  # Старые
            
        except Exception as e:
            logger.warning(f"Error calculating time bonus: {e}")
            return 0.0
    
    def _calculate_importance_bonus(self, result: Dict[str, Any], importance_filter: ImportanceFilter) -> float:
        """Вычислить бонус за важность"""
        try:
            importance = result.get("importance", 0.5)
            
            # Бонус за высокую важность
            if importance > 0.8:
                return 0.3
            elif importance > 0.6:
                return 0.2
            elif importance > 0.4:
                return 0.1
            else:
                return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating importance bonus: {e}")
            return 0.0
    
    def _calculate_memory_type_bonus(self, result: Dict[str, Any]) -> float:
        """Вычислить бонус за тип памяти"""
        try:
            memory_type = result.get("memory_type", "general")
            
            # Приоритеты типов памяти
            type_priorities = {
                "working": 0.3,
                "short_term": 0.2,
                "episodic": 0.2,
                "semantic": 0.1,
                "graph": 0.1,
                "procedural": 0.1,
                "general": 0.0
            }
            
            return type_priorities.get(memory_type, 0.0)
            
        except Exception as e:
            logger.warning(f"Error calculating memory type bonus: {e}")
            return 0.0
    
    def _determine_sentiment(self, emotion_data: Dict[str, Any]) -> str:
        """Определить тональность по эмоциональным данным"""
        try:
            primary_emotion = emotion_data.get("primary_emotion", "").lower()
            
            positive_emotions = ["радость", "joy", "happiness", "excitement", "excited", "happy", "pleased", "satisfied"]
            negative_emotions = ["грусть", "sadness", "anger", "fear", "disgust", "sad", "angry", "afraid", "worried"]
            
            if any(pos in primary_emotion for pos in positive_emotions):
                return "positive"
            elif any(neg in primary_emotion for neg in negative_emotions):
                return "negative"
            else:
                return "neutral"
                
        except Exception as e:
            logger.warning(f"Error determining sentiment: {e}")
            return "neutral"

# Global contextual search engine instance
_contextual_engine: Optional[ContextualSearchEngine] = None

async def get_contextual_engine(memory_orchestrator: Optional["MemoryOrchestrator"] = None) -> ContextualSearchEngine:
    """Get global contextual search engine instance"""
    global _contextual_engine
    if _contextual_engine is None:
        _contextual_engine = ContextualSearchEngine(memory_orchestrator)
    return _contextual_engine

def get_contextual_engine_lazy():
    """Get lazy contextual search engine factory"""
    return get_contextual_engine
