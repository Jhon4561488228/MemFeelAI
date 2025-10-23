"""
Semantic Search Engine for AIRI Memory System.
Provides semantic search capabilities using SemanticMemoryManager.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger

from ..memory_levels.semantic_memory import SemanticMemoryManager, SemanticMemoryItem

try:
    from ..monitoring.search_metrics import record_search_metrics
except ImportError:
    from monitoring.search_metrics import record_search_metrics

class SemanticSearchEngine:
    """Семантический поисковый движок"""
    
    def __init__(self, semantic_manager: Optional[SemanticMemoryManager] = None):
        """
        Инициализация семантического поискового движка
        
        Args:
            semantic_manager: Менеджер семантической памяти
        """
        self.semantic_manager = semantic_manager
        logger.info("SemanticSearchEngine initialized")
    
    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
        min_confidence: float = 0.25
    ) -> List[SemanticMemoryItem]:
        """
        Выполнить семантический поиск
        
        Args:
            query: Поисковый запрос
            user_id: ID пользователя
            limit: Максимальное количество результатов
            offset: Смещение для пагинации
            min_confidence: Минимальная уверенность
            
        Returns:
            Список результатов семантического поиска
        """
        start_time = time.time()
        try:
            if not self.semantic_manager:
                logger.warning("Semantic manager not available")
                return []
            
            logger.info(f"Semantic search: query='{query}', user_id='{user_id}', limit={limit}")
            logger.info(f"Semantic manager type: {type(self.semantic_manager)}")
            logger.info(f"Semantic manager: {self.semantic_manager}")
            
            # Выполняем семантический поиск
            logger.info("About to call semantic_manager.search_knowledge")
            results = await self.semantic_manager.search_knowledge(
                user_id=user_id,
                query=query,
                knowledge_type=None,
                category=None,
                min_confidence=min_confidence,
                limit=limit
            )
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Semantic search completed: {len(results)} results in {duration_ms:.2f}ms")
            
            # Записываем метрики поиска
            try:
                record_search_metrics(
                    query=query,
                    search_type="semantic",
                    user_id=user_id,
                    duration_ms=duration_ms,
                    results=results,
                    cache_hit=False,
                    error=None
                )
            except Exception as e:
                logger.warning(f"Failed to record semantic search metrics: {e}")
            logger.info(f"Results type: {type(results)}")
            if results:
                logger.info(f"First result type: {type(results[0])}")
            return results
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Semantic search failed: {e}")
            
            # Записываем метрики ошибки
            try:
                record_search_metrics(
                    query=query,
                    search_type="semantic",
                    user_id=user_id,
                    duration_ms=duration_ms,
                    results=[],
                    cache_hit=False,
                    error=str(e)
                )
            except Exception as metrics_error:
                logger.warning(f"Failed to record semantic search error metrics: {metrics_error}")
            
            return []
    
    async def list_memories(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[SemanticMemoryItem]:
        """
        Получить список семантической памяти
        
        Args:
            user_id: ID пользователя
            limit: Максимальное количество результатов
            offset: Смещение для пагинации
            
        Returns:
            Список записей семантической памяти
        """
        try:
            if not self.semantic_manager:
                logger.warning("Semantic manager not available")
                return []
            
            logger.info(f"List semantic memories: user_id='{user_id}', limit={limit}")
            
            # Получаем список семантической памяти
            results = await self.semantic_manager.get_knowledge(
                user_id=user_id,
                limit=limit,
                offset=offset
            )
            
            logger.info(f"List semantic memories completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"List semantic memories failed: {e}")
            return []

# Глобальный экземпляр семантического поискового движка
_semantic_engine: Optional[SemanticSearchEngine] = None

async def get_semantic_engine(semantic_manager: Optional[SemanticMemoryManager] = None) -> SemanticSearchEngine:
    """
    Получить экземпляр семантического поискового движка
    
    Args:
        semantic_manager: Менеджер семантической памяти
        
    Returns:
        Экземпляр SemanticSearchEngine
    """
    global _semantic_engine
    
    if _semantic_engine is None or (semantic_manager and _semantic_engine.semantic_manager != semantic_manager):
        _semantic_engine = SemanticSearchEngine(semantic_manager)
        logger.info("SemanticSearchEngine created/updated")
    elif semantic_manager and _semantic_engine.semantic_manager is None:
        # Обновляем существующий экземпляр с новым менеджером
        _semantic_engine.semantic_manager = semantic_manager
        logger.info("SemanticSearchEngine updated with new manager")
    else:
        # Если semantic_manager не передан, но _semantic_engine существует
        logger.info("SemanticSearchEngine exists but no semantic_manager provided")
    
    return _semantic_engine
