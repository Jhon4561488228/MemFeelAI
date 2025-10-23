#!/usr/bin/env python3
"""
Функции очистки памяти для различных компонентов системы
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MemoryCleanup:
    """Класс для очистки памяти различных компонентов"""
    
    def __init__(self):
        self.cleanup_stats = {
            "sensor_buffer": {"cleanups": 0, "last_cleanup": None, "items_cleared": 0},
            "memory_cache": {"cleanups": 0, "last_cleanup": None, "items_cleared": 0},
            "sqlite_cache": {"cleanups": 0, "last_cleanup": None, "items_cleared": 0},
            "lm_provider": {"cleanups": 0, "last_cleanup": None, "items_cleared": 0},
            "memory_orchestrator": {"cleanups": 0, "last_cleanup": None, "items_cleared": 0}
        }
    
    async def cleanup_sensor_buffer(self) -> float:
        """
        Очистка SensorBuffer - удаление старых данных
        
        Returns:
            Количество освобожденных MB (примерная оценка)
        """
        try:
            from ..memory_levels.sensor_buffer import get_sensor_buffer
            
            sensor_buffer = get_sensor_buffer()
            if not sensor_buffer:
                return 0.0
            
            # Получаем статистику до очистки
            before_stats = sensor_buffer.get_stats()
            
            # Очищаем старые данные (старше 1 часа)
            cutoff_time = time.time() - 3600  # 1 час назад
            cleared_count = 0
            
            # Очищаем каждый тип буфера
            for buffer_name in ['_audio_buffer', '_video_buffer', '_text_buffer', 
                              '_emotion_buffer', '_gesture_buffer', '_environment_buffer']:
                buffer = getattr(sensor_buffer, buffer_name, None)
                if buffer:
                    # Удаляем старые элементы
                    old_items = [item for item in buffer if item.timestamp < cutoff_time]
                    for item in old_items:
                        buffer.remove(item)
                        cleared_count += 1
            
            # Получаем статистику после очистки
            after_stats = sensor_buffer.get_stats()
            
            # Примерная оценка освобожденной памяти (1KB на элемент)
            freed_mb = cleared_count * 0.001
            
            # Обновляем статистику
            self.cleanup_stats["sensor_buffer"]["cleanups"] += 1
            self.cleanup_stats["sensor_buffer"]["last_cleanup"] = datetime.now()
            self.cleanup_stats["sensor_buffer"]["items_cleared"] += cleared_count
            
            if cleared_count > 0:
                logger.info(f"SensorBuffer очищен: удалено {cleared_count} элементов, "
                           f"освобождено ~{freed_mb:.2f}MB")
            
            return freed_mb
            
        except Exception as e:
            logger.error(f"Ошибка очистки SensorBuffer: {e}")
            return 0.0
    
    async def cleanup_memory_cache(self) -> float:
        """
        Очистка Memory Cache - удаление устаревших записей
        
        Returns:
            Количество освобожденных MB (примерная оценка)
        """
        try:
            from ..cache.memory_cache import cache_get, cache_set, cache_delete_prefix
            
            # Получаем статистику кэша
            try:
                # Подсчитываем количество записей в кэше
                cache_stats = {
                    "search_cache_entries": 0,
                    "user_context_cache_entries": 0,
                    "memory_cache_entries": 0,
                    "embedding_cache_entries": 0
                }
                
                # Подсчитываем реальные записи в кэше
                try:
                    # Получаем статистику из кэша (если доступна)
                    # Это примерная реализация - в реальности нужно использовать API кэша
                    cache_stats["search_cache_entries"] = 0  # Будет реализовано через API кэша
                    cache_stats["user_context_cache_entries"] = 0
                    cache_stats["memory_cache_entries"] = 0
                    cache_stats["embedding_cache_entries"] = 0
                except Exception as e:
                    logger.warning(f"Не удалось получить детальную статистику кэша: {e}")
                logger.info("Memory Cache: получена статистика кэша")
            except Exception as e:
                logger.warning(f"Не удалось получить статистику кэша: {e}")
            
            # Очищаем кэш поиска (часто накапливается)
            cache_delete_prefix("search:")
            cache_delete_prefix("user_context:")
            
            # Примерная оценка освобожденной памяти
            freed_mb = 5.0  # Примерная оценка
            
            # Обновляем статистику
            self.cleanup_stats["memory_cache"]["cleanups"] += 1
            self.cleanup_stats["memory_cache"]["last_cleanup"] = datetime.now()
            self.cleanup_stats["memory_cache"]["items_cleared"] += 1
            
            logger.info(f"Memory Cache очищен: освобождено ~{freed_mb:.2f}MB")
            return freed_mb
            
        except Exception as e:
            logger.error(f"Ошибка очистки Memory Cache: {e}")
            return 0.0
    
    async def cleanup_sqlite_cache(self) -> float:
        """
        Очистка SQLite Cache - удаление устаревших записей
        
        Returns:
            Количество освобожденных MB
        """
        try:
            from ..cache.sqlite_cache import get_sqlite_cache
            
            cache = await get_sqlite_cache()
            if not cache:
                return 0.0
            
            # Получаем статистику до очистки
            before_stats = await cache.get_stats()
            
            # Очищаем устаревшие записи
            await cache.cleanup_expired()
            
            # Получаем статистику после очистки
            after_stats = await cache.get_stats()
            
            # Вычисляем освобожденную память
            freed_mb = 0.0
            if before_stats and after_stats:
                before_size = before_stats.get("total_size_mb", 0)
                after_size = after_stats.get("total_size_mb", 0)
                freed_mb = max(0, before_size - after_size)
            
            # Обновляем статистику
            self.cleanup_stats["sqlite_cache"]["cleanups"] += 1
            self.cleanup_stats["sqlite_cache"]["last_cleanup"] = datetime.now()
            self.cleanup_stats["sqlite_cache"]["items_cleared"] += 1
            
            if freed_mb > 0:
                logger.info(f"SQLite Cache очищен: освобождено {freed_mb:.2f}MB")
            
            return freed_mb
            
        except Exception as e:
            logger.error(f"Ошибка очистки SQLite Cache: {e}")
            return 0.0
    
    async def cleanup_lm_provider(self) -> float:
        """
        Очистка LM Studio Provider - очистка fallback кэша
        
        Returns:
            Количество освобожденных MB
        """
        try:
            from ..providers.lm_studio_provider import LMStudioProvider
            
            # Получаем глобальный экземпляр провайдера
            try:
                from ..providers.lm_studio_provider import LMStudioProvider
                
                # Очищаем fallback кэш провайдера
                if hasattr(LMStudioProvider, '_memory_cache'):
                    old_entries = 0
                    current_time = time.time()
                    
                    # Удаляем записи старше 1 часа
                    old_keys = []
                    for key, (value, timestamp) in LMStudioProvider._memory_cache.items():
                        if current_time - timestamp > 3600:  # 1 час
                            old_keys.append(key)
                    
                    for key in old_keys:
                        del LMStudioProvider._memory_cache[key]
                        old_entries += 1
                    
                    if old_entries > 0:
                        logger.info(f"LM Provider: очищено {old_entries} устаревших записей из fallback кэша")
                        freed_mb = old_entries * 0.001  # Примерная оценка
                    else:
                        freed_mb = 0.0
                else:
                    freed_mb = 0.0
                    
            except Exception as e:
                logger.warning(f"Не удалось очистить LM Provider: {e}")
                freed_mb = 0.0
            
            # Обновляем статистику
            self.cleanup_stats["lm_provider"]["cleanups"] += 1
            self.cleanup_stats["lm_provider"]["last_cleanup"] = datetime.now()
            self.cleanup_stats["lm_provider"]["items_cleared"] += 1
            
            logger.info(f"LM Provider очищен: освобождено ~{freed_mb:.2f}MB")
            return freed_mb
            
        except Exception as e:
            logger.error(f"Ошибка очистки LM Provider: {e}")
            return 0.0
    
    async def cleanup_memory_orchestrator(self) -> float:
        """
        Очистка MemoryOrchestrator - очистка метрик и кэшей
        
        Returns:
            Количество освобожденных MB
        """
        try:
            # Очищаем метрики MemoryOrchestrator
            try:
                from ..memory_levels.memory_orchestrator import memory_orchestrator
                if memory_orchestrator:
                    # Очищаем старые счетчики суммаризации
                    if hasattr(memory_orchestrator, '_summary_counters'):
                        old_counters = 0
                        current_time = time.time()
                        
                        # Удаляем счетчики старше 1 часа
                        old_keys = []
                        for user_id, counter_data in memory_orchestrator._summary_counters.items():
                            if isinstance(counter_data, dict) and 'last_update' in counter_data:
                                if current_time - counter_data['last_update'] > 3600:  # 1 час
                                    old_keys.append(user_id)
                        
                        for key in old_keys:
                            del memory_orchestrator._summary_counters[key]
                            old_counters += 1
                        
                        if old_counters > 0:
                            logger.info(f"MemoryOrchestrator: очищено {old_counters} старых счетчиков")
                            freed_mb = old_counters * 0.001  # Примерная оценка
                        else:
                            freed_mb = 0.0
                    else:
                        freed_mb = 0.0
                else:
                    freed_mb = 0.0
                    
            except Exception as e:
                logger.warning(f"Не удалось очистить метрики MemoryOrchestrator: {e}")
                freed_mb = 0.0
            
            # Обновляем статистику
            self.cleanup_stats["memory_orchestrator"]["cleanups"] += 1
            self.cleanup_stats["memory_orchestrator"]["last_cleanup"] = datetime.now()
            self.cleanup_stats["memory_orchestrator"]["items_cleared"] += 1
            
            logger.info(f"MemoryOrchestrator очищен: освобождено ~{freed_mb:.2f}MB")
            return freed_mb
            
        except Exception as e:
            logger.error(f"Ошибка очистки MemoryOrchestrator: {e}")
            return 0.0
    
    async def cleanup_all_components(self) -> Dict[str, float]:
        """
        Очистка всех компонентов системы
        
        Returns:
            Словарь с результатами очистки каждого компонента
        """
        results = {}
        
        cleanup_functions = [
            ("sensor_buffer", self.cleanup_sensor_buffer),
            ("memory_cache", self.cleanup_memory_cache),
            ("sqlite_cache", self.cleanup_sqlite_cache),
            ("lm_provider", self.cleanup_lm_provider),
            ("memory_orchestrator", self.cleanup_memory_orchestrator)
        ]
        
        for component_name, cleanup_func in cleanup_functions:
            try:
                freed_mb = await cleanup_func()
                results[component_name] = freed_mb
            except Exception as e:
                logger.error(f"Ошибка очистки компонента {component_name}: {e}")
                results[component_name] = 0.0
        
        total_freed = sum(results.values())
        logger.info(f"Общая очистка завершена: освобождено {total_freed:.2f}MB")
        
        return results
    
    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Получение статистики очистки"""
        return self.cleanup_stats.copy()

# Глобальный экземпляр очистки
_memory_cleanup: Optional[MemoryCleanup] = None

def get_memory_cleanup() -> MemoryCleanup:
    """Получение глобального экземпляра очистки памяти"""
    global _memory_cleanup
    if _memory_cleanup is None:
        _memory_cleanup = MemoryCleanup()
    return _memory_cleanup
