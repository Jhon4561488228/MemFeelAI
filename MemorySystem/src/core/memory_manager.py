"""
Memory Manager для AIRI Memory System
Основной менеджер, координирующий работу всех компонентов
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
import yaml
from pathlib import Path

from .local_mem0 import LocalMem0
from .performance_optimizer import PerformanceOptimizer
from ..search import get_fts5_engine

class MemoryManager:
    """Основной менеджер системы памяти"""
    
    def __init__(self, config_path: str = "config/mem0_config.yaml"):
        """Инициализация менеджера памяти"""
        self.config_path = config_path
        self.mem0 = None
        self.config = None
        self.optimizer = PerformanceOptimizer("config/performance_config.yaml")
        self._initialize()
        logger.info("Memory Manager инициализирован с оптимизатором производительности")
    
    def _initialize(self):
        """Инициализация компонентов"""
        try:
            # Загружаем конфигурацию
            self.config = self._load_config()
            
            # Инициализируем Local Mem0
            self.mem0 = LocalMem0(self.config_path)
            
            logger.info("Memory Manager готов к работе")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации Memory Manager: {e}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Конфигурационный файл {self.config_path} не найден")
            return {}
    
    async def add_memory(
        self, 
        content: str, 
        user_id: str = "default_user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Добавление воспоминания"""
        try:
            if not content or not content.strip():
                raise ValueError("Содержимое воспоминания не может быть пустым")
            
            # Проверяем лимиты
            await self._check_memory_limits(user_id)
            
            # Добавляем воспоминание через Mem0
            result = await self.mem0.add_memory(
                content=content.strip(),
                user_id=user_id,
                metadata=metadata
            )
            
            # Индексируем в FTS5 для ключевого поиска
            try:
                fts5_engine = await get_fts5_engine()
                memory_type = metadata.get("memory_type", "general") if metadata else "general"
                importance = metadata.get("importance", 0.5) if metadata else 0.5
                
                await fts5_engine.index_memory(
                    memory_id=result['id'],
                    content=content.strip(),
                    user_id=user_id,
                    memory_type=memory_type,
                    importance=importance
                )
                logger.debug(f"Memory {result['id']} indexed in FTS5")
            except Exception as e:
                logger.warning(f"Failed to index memory {result['id']} in FTS5: {e}")
            
            logger.info(f"Воспоминание добавлено для пользователя {user_id}: {result['id']}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка добавления воспоминания: {e}")
            raise
    
    async def search_memories(
        self, 
        query: str, 
        user_id: str = "default_user",
        limit: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Поиск воспоминаний с оптимизацией"""
        try:
            if not query or not query.strip():
                return {"results": [], "query": query, "total": 0}
            
            # Оптимизируем запрос
            optimized_request = await self.optimizer.optimize_search_request(
                query=query.strip(),
                user_id=user_id,
                limit=limit,
                similarity_threshold=similarity_threshold or 0.2
            )
            
            # Ищем воспоминания через Mem0
            results = await self.mem0.search_memories(
                query=optimized_request["query"],
                user_id=optimized_request["user_id"],
                limit=optimized_request["limit"],
                similarity_threshold=optimized_request["similarity_threshold"]
            )
            
            logger.info(f"Найдено {results['total']} воспоминаний для запроса: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска воспоминаний: {e}")
            return {"results": [], "query": query, "total": 0, "error": str(e)}
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Получение конкретного воспоминания"""
        try:
            memory = await self.mem0.get_memory(memory_id)
            if memory:
                logger.info(f"Воспоминание получено: {memory_id}")
            return memory
        except Exception as e:
            logger.error(f"Ошибка получения воспоминания {memory_id}: {e}")
            return None
    
    async def update_memory(
        self, 
        memory_id: str, 
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Обновление воспоминания"""
        try:
            success = await self.mem0.update_memory(
                memory_id=memory_id,
                content=content,
                metadata=metadata
            )
            
            if success:
                logger.info(f"Воспоминание обновлено: {memory_id}")
            return success
            
        except Exception as e:
            logger.error(f"Ошибка обновления воспоминания {memory_id}: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Удаление воспоминания"""
        try:
            success = await self.mem0.delete_memory(memory_id)
            if success:
                logger.info(f"Воспоминание удалено: {memory_id}")
            return success
        except Exception as e:
            logger.error(f"Ошибка удаления воспоминания {memory_id}: {e}")
            return False
    
    async def get_user_memories(
        self, 
        user_id: str = "default_user",
        limit: int = 100,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Получение всех воспоминаний пользователя"""
        try:
            memories = await self.mem0.get_user_memories(
                user_id=user_id,
                limit=limit,
                memory_type=memory_type
            )
            
            logger.info(f"Получено {len(memories)} воспоминаний для пользователя {user_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Ошибка получения воспоминаний пользователя {user_id}: {e}")
            return []
    
    async def get_memory_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Получение статистики памяти"""
        try:
            stats = await self.mem0.get_stats()
            
            # Добавляем статистику по пользователям если указан user_id
            if user_id:
                user_memories = await self.get_user_memories(user_id, limit=10000)
                user_stats = {
                    "total_memories": len(user_memories),
                    "memory_types": {},
                    "recent_memories": len([m for m in user_memories 
                                          if self._is_recent(m.get("created_at", ""))])
                }
                
                # Подсчитываем типы воспоминаний
                for memory in user_memories:
                    memory_type = memory.get("metadata", {}).get("type", "unknown")
                    user_stats["memory_types"][memory_type] = user_stats["memory_types"].get(memory_type, 0) + 1
                
                stats["user_stats"] = user_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}
    
    async def cleanup_old_memories(self, days: Optional[int] = None) -> int:
        """Очистка старых воспоминаний"""
        try:
            deleted_count = await self.mem0.cleanup_old_memories(days)
            logger.info(f"Очищено {deleted_count} старых воспоминаний")
            return deleted_count
        except Exception as e:
            logger.error(f"Ошибка очистки старых воспоминаний: {e}")
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья системы"""
        try:
            health = await self.mem0.health_check()
            
            # Добавляем общую информацию
            health["service"] = "airi-memory"
            health["version"] = "1.0.0"
            health["timestamp"] = datetime.now().isoformat()
            
            return health
            
        except Exception as e:
            logger.error(f"Ошибка проверки здоровья: {e}")
            return {
                "overall": False,
                "service": "airi-memory",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _check_memory_limits(self, user_id: str):
        """Проверка лимитов памяти"""
        try:
            max_memories = self.config.get("memory_settings", {}).get("max_memories", 10000)
            cleanup_threshold = self.config.get("memory_settings", {}).get("cleanup_threshold", 0.8)
            
            # Получаем текущее количество воспоминаний
            user_memories = await self.get_user_memories(user_id, limit=max_memories)
            current_count = len(user_memories)
            
            # Если превышен порог очистки, запускаем очистку
            if current_count >= max_memories * cleanup_threshold:
                logger.info(f"Превышен порог памяти для пользователя {user_id}, запускаем очистку")
                await self.cleanup_old_memories()
            
        except Exception as e:
            logger.error(f"Ошибка проверки лимитов памяти: {e}")
    
    def _is_recent(self, created_at: str, days: int = 7) -> bool:
        """Проверка, является ли воспоминание недавним"""
        try:
            if not created_at:
                return False
            
            created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            now = datetime.now(created.tzinfo) if created.tzinfo else datetime.now()
            
            return (now - created).days <= days
        except Exception:
            return False
    
    async def batch_add_memories(
        self, 
        memories: List[Dict[str, Any]], 
        user_id: str = "default_user"
    ) -> List[Dict[str, Any]]:
        """Пакетное добавление воспоминаний"""
        try:
            results = []
            
            for memory_data in memories:
                content = memory_data.get("content", "")
                metadata = memory_data.get("metadata", {})
                
                if content:
                    result = await self.add_memory(
                        content=content,
                        user_id=user_id,
                        metadata=metadata
                    )
                    results.append(result)
            
            logger.info(f"Пакетно добавлено {len(results)} воспоминаний")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка пакетного добавления воспоминаний: {e}")
            return []
    
    async def search_similar_memories(
        self, 
        memory_id: str, 
        user_id: str = "default_user",
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Поиск похожих воспоминаний"""
        try:
            # Получаем исходное воспоминание
            memory = await self.get_memory(memory_id)
            if not memory:
                return []
            
            # Ищем похожие воспоминания по содержимому
            results = await self.search_memories(
                query=memory["content"],
                user_id=user_id,
                limit=limit + 1,  # +1 чтобы исключить само воспоминание
                similarity_threshold=0.2  # Используем оптимальный порог для русских текстов
            )
            
            # Фильтруем исходное воспоминание
            similar_memories = [
                m for m in results["results"] 
                if m["id"] != memory_id
            ]
            
            return similar_memories[:limit]
            
        except Exception as e:
            logger.error(f"Ошибка поиска похожих воспоминаний: {e}")
            return []
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности"""
        try:
            metrics = self.optimizer.get_performance_metrics()
            return metrics
        except Exception as e:
            logger.error(f"Ошибка получения метрик производительности: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Закрытие менеджера памяти"""
        try:
            if self.optimizer:
                await self.optimizer.cleanup()
            if self.mem0:
                await self.mem0.close()
            logger.info("Memory Manager закрыт")
        except Exception as e:
            logger.error(f"Ошибка закрытия Memory Manager: {e}")
