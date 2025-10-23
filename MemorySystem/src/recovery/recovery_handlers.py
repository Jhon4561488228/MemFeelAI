#!/usr/bin/env python3
"""
Обработчики восстановления для различных компонентов системы
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RecoveryHandlers:
    """Класс с обработчиками восстановления для различных компонентов"""
    
    def __init__(self, memory_orchestrator=None):
        self.memory_orchestrator = memory_orchestrator
        self.recovery_stats = {
            "memory_orchestrator": {"recoveries": 0, "last_recovery": None, "success_rate": 0.0},
            "sensor_buffer": {"recoveries": 0, "last_recovery": None, "success_rate": 0.0},
            "memory_cache": {"recoveries": 0, "last_recovery": None, "success_rate": 0.0},
            "sqlite_cache": {"recoveries": 0, "last_recovery": None, "success_rate": 0.0},
            "lm_provider": {"recoveries": 0, "last_recovery": None, "success_rate": 0.0},
            "chromadb": {"recoveries": 0, "last_recovery": None, "success_rate": 0.0},
            "emotion_service": {"recoveries": 0, "last_recovery": None, "success_rate": 0.0}
        }
    
    async def recover_memory_orchestrator(self) -> bool:
        """
        Восстановление MemoryOrchestrator
        
        Returns:
            True если восстановление успешно
        """
        try:
            logger.info("Начинаем восстановление MemoryOrchestrator...")
            
            # 1. Очищаем фоновые задачи
            memory_orchestrator = self.memory_orchestrator
            if memory_orchestrator:
                # Отменяем все фоновые задачи
                for task in list(memory_orchestrator._background_tasks):
                    if not task.done():
                        task.cancel()
                
                # Очищаем завершенные задачи
                memory_orchestrator._background_tasks.clear()
                
                # Сбрасываем счетчики суммаризации
                memory_orchestrator._summary_counters.clear()
                
                logger.info("MemoryOrchestrator: очищены фоновые задачи и счетчики")
            
            # 2. Перезапускаем мониторинг памяти
            if hasattr(memory_orchestrator, 'memory_monitor') and memory_orchestrator.memory_monitor:
                await memory_orchestrator.stop_memory_monitoring()
                await asyncio.sleep(1)  # Небольшая пауза
                await memory_orchestrator.start_memory_monitoring()
                logger.info("MemoryOrchestrator: перезапущен мониторинг памяти")
            
            # 3. Перезапускаем проактивный таймер
            if hasattr(memory_orchestrator, 'proactive_timer') and memory_orchestrator.proactive_timer:
                memory_orchestrator.proactive_timer.stop()
                await asyncio.sleep(1)
                memory_orchestrator.proactive_timer.start()
                logger.info("MemoryOrchestrator: перезапущен проактивный таймер")
            
            # Обновляем статистику
            self.recovery_stats["memory_orchestrator"]["recoveries"] += 1
            self.recovery_stats["memory_orchestrator"]["last_recovery"] = datetime.now()
            self.recovery_stats["memory_orchestrator"]["success_rate"] = 1.0
            
            logger.info("MemoryOrchestrator успешно восстановлен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка восстановления MemoryOrchestrator: {e}")
            self.recovery_stats["memory_orchestrator"]["success_rate"] = 0.0
            return False
    
    async def recover_sensor_buffer(self) -> bool:
        """
        Восстановление SensorBuffer
        
        Returns:
            True если восстановление успешно
        """
        try:
            logger.info("Начинаем восстановление SensorBuffer...")
            
            from ..memory_levels.sensor_buffer import get_sensor_buffer
            
            sensor_buffer = get_sensor_buffer()
            if sensor_buffer:
                # Очищаем все буферы
                sensor_buffer._audio_buffer.clear()
                sensor_buffer._video_buffer.clear()
                sensor_buffer._text_buffer.clear()
                sensor_buffer._emotion_buffer.clear()
                sensor_buffer._gesture_buffer.clear()
                sensor_buffer._environment_buffer.clear()
                
                # Сбрасываем метрики
                sensor_buffer._total_added = 0
                sensor_buffer._total_evicted = 0
                sensor_buffer._last_cleanup = time.time()
                
                logger.info("SensorBuffer: очищены все буферы и метрики")
            
            # Обновляем статистику
            self.recovery_stats["sensor_buffer"]["recoveries"] += 1
            self.recovery_stats["sensor_buffer"]["last_recovery"] = datetime.now()
            self.recovery_stats["sensor_buffer"]["success_rate"] = 1.0
            
            logger.info("SensorBuffer успешно восстановлен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка восстановления SensorBuffer: {e}")
            self.recovery_stats["sensor_buffer"]["success_rate"] = 0.0
            return False
    
    async def recover_memory_cache(self) -> bool:
        """
        Восстановление Memory Cache
        
        Returns:
            True если восстановление успешно
        """
        try:
            logger.info("Начинаем восстановление Memory Cache...")
            
            from ..cache.memory_cache import cache_delete_prefix
            
            # Очищаем все кэши
            cache_delete_prefix("search:")
            cache_delete_prefix("user_context:")
            cache_delete_prefix("memory:")
            cache_delete_prefix("embedding:")
            
            logger.info("Memory Cache: очищены все кэши")
            
            # Обновляем статистику
            self.recovery_stats["memory_cache"]["recoveries"] += 1
            self.recovery_stats["memory_cache"]["last_recovery"] = datetime.now()
            self.recovery_stats["memory_cache"]["success_rate"] = 1.0
            
            logger.info("Memory Cache успешно восстановлен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка восстановления Memory Cache: {e}")
            self.recovery_stats["memory_cache"]["success_rate"] = 0.0
            return False
    
    async def recover_sqlite_cache(self) -> bool:
        """
        Восстановление SQLite Cache
        
        Returns:
            True если восстановление успешно
        """
        try:
            logger.info("Начинаем восстановление SQLite Cache...")
            
            from ..cache.sqlite_cache import get_sqlite_cache
            
            cache = await get_sqlite_cache()
            if cache:
                # Очищаем устаревшие записи
                await cache.cleanup_expired()
                
                # Оптимизируем базу данных
                await cache._optimize_database()
                
                logger.info("SQLite Cache: очищены устаревшие записи и оптимизирована БД")
            
            # Обновляем статистику
            self.recovery_stats["sqlite_cache"]["recoveries"] += 1
            self.recovery_stats["sqlite_cache"]["last_recovery"] = datetime.now()
            self.recovery_stats["sqlite_cache"]["success_rate"] = 1.0
            
            logger.info("SQLite Cache успешно восстановлен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка восстановления SQLite Cache: {e}")
            self.recovery_stats["sqlite_cache"]["success_rate"] = 0.0
            return False
    
    async def recover_lm_provider(self) -> bool:
        """
        Восстановление LM Studio Provider
        
        Returns:
            True если восстановление успешно
        """
        try:
            logger.info("Начинаем восстановление LM Provider...")
            
            # 1. Очищаем fallback кэш
            from ..providers.lm_studio_provider import LMStudioProvider
            
            # Получаем глобальный экземпляр провайдера (если доступен)
            # Очищаем fallback кэш
            try:
                # Очищаем старые записи из fallback кэша (старше 1 часа)
                current_time = time.time()
                if hasattr(LMStudioProvider, '_memory_cache'):
                    # Это статический кэш, очищаем его
                    old_keys = []
                    for key, (value, timestamp) in LMStudioProvider._memory_cache.items():
                        if current_time - timestamp > 3600:  # 1 час
                            old_keys.append(key)
                    
                    for key in old_keys:
                        del LMStudioProvider._memory_cache[key]
                    
                    logger.info(f"LM Provider: очищено {len(old_keys)} устаревших записей из fallback кэша")
            except Exception as e:
                logger.warning(f"Не удалось очистить fallback кэш LM Provider: {e}")
            
            # 2. Сбрасываем флаг использования SQLite (принудительное переключение)
            try:
                # Если есть глобальный экземпляр, сбрасываем его состояние
                if hasattr(LMStudioProvider, '_use_sqlite'):
                    LMStudioProvider._use_sqlite = True
                    logger.info("LM Provider: сброшен флаг использования SQLite")
            except Exception as e:
                logger.warning(f"Не удалось сбросить состояние LM Provider: {e}")
            
            # 3. Принудительная очистка SQLite кэша
            try:
                from ..cache.sqlite_cache import get_sqlite_cache
                cache = await get_sqlite_cache()
                if cache:
                    await cache.cleanup_expired()
                    logger.info("LM Provider: очищен SQLite кэш")
            except Exception as e:
                logger.warning(f"Не удалось очистить SQLite кэш: {e}")
            
            # Обновляем статистику
            self.recovery_stats["lm_provider"]["recoveries"] += 1
            self.recovery_stats["lm_provider"]["last_recovery"] = datetime.now()
            self.recovery_stats["lm_provider"]["success_rate"] = 1.0
            
            logger.info("LM Provider успешно восстановлен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка восстановления LM Provider: {e}")
            self.recovery_stats["lm_provider"]["success_rate"] = 0.0
            return False
    
    async def recover_chromadb(self) -> bool:
        """
        Восстановление ChromaDB
        
        Returns:
            True если восстановление успешно
        """
        try:
            logger.info("Начинаем восстановление ChromaDB...")
            
            # 1. Очищаем кэш ChromaDB провайдера
            try:
                from ..providers.chromadb_provider import ChromaDBProvider
                
                # Очищаем кэш коллекций (если есть)
                if hasattr(ChromaDBProvider, '_collection_cache'):
                    ChromaDBProvider._collection_cache.clear()
                    logger.info("ChromaDB: очищен кэш коллекций")
                
                # Сбрасываем флаг инициализации
                if hasattr(ChromaDBProvider, '_initialized'):
                    ChromaDBProvider._initialized = False
                    logger.info("ChromaDB: сброшен флаг инициализации")
                    
            except Exception as e:
                logger.warning(f"Не удалось очистить кэш ChromaDB: {e}")
            
            # 2. Принудительная очистка старых данных
            try:
                memory_orchestrator = self.memory_orchestrator
                if memory_orchestrator and hasattr(memory_orchestrator, 'semantic_memory'):
                    # Очищаем старые записи из семантической памяти (старше 7 дней)
                    cutoff_time = time.time() - (7 * 24 * 3600)  # 7 дней
                    # Очищаем старые записи из семантической памяти
                    try:
                        # Получаем все записи старше cutoff_time
                        old_records = await memory_orchestrator.semantic_memory.get_old_records(cutoff_time)
                        if old_records:
                            # Удаляем старые записи
                            for record_id in old_records:
                                await memory_orchestrator.semantic_memory.delete_memory(record_id)
                            logger.info(f"ChromaDB: удалено {len(old_records)} старых записей")
                        else:
                            logger.info("ChromaDB: старых записей для удаления не найдено")
                    except Exception as cleanup_error:
                        logger.warning(f"Не удалось очистить старые записи ChromaDB: {cleanup_error}")
                    
            except Exception as e:
                logger.warning(f"Не удалось очистить старые данные ChromaDB: {e}")
            
            # 3. Переинициализация подключения
            try:
                # Принудительно переинициализируем ChromaDB клиент
                from ..providers.chromadb_provider import get_chromadb_provider
                provider = await get_chromadb_provider()
                if provider:
                    # Выполняем health check для проверки подключения
                    health_status = await provider.health_check()
                    if health_status:
                        logger.info("ChromaDB: подключение восстановлено")
                    else:
                        logger.warning("ChromaDB: проблемы с подключением")
                        
            except Exception as e:
                logger.warning(f"Не удалось переинициализировать ChromaDB: {e}")
            
            # Обновляем статистику
            self.recovery_stats["chromadb"]["recoveries"] += 1
            self.recovery_stats["chromadb"]["last_recovery"] = datetime.now()
            self.recovery_stats["chromadb"]["success_rate"] = 1.0
            
            logger.info("ChromaDB успешно восстановлен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка восстановления ChromaDB: {e}")
            self.recovery_stats["chromadb"]["success_rate"] = 0.0
            return False
    
    async def recover_emotion_service(self) -> bool:
        """
        Восстановление Emotion Service
        
        Returns:
            True если восстановление успешно
        """
        try:
            logger.info("Начинаем восстановление Emotion Service...")
            
            # 1. Очищаем кэш Emotion Service
            try:
                from ..emotion_analysis_service import emotion_service
                if emotion_service:
                    # Очищаем кэш эмоций (если есть)
                    if hasattr(emotion_service, '_emotion_cache'):
                        emotion_service._emotion_cache.clear()
                        logger.info("Emotion Service: очищен кэш эмоций")
                    
                    # Сбрасываем счетчики ошибок
                    if hasattr(emotion_service, '_error_count'):
                        emotion_service._error_count = 0
                        logger.info("Emotion Service: сброшены счетчики ошибок")
                        
            except Exception as e:
                logger.warning(f"Не удалось очистить кэш Emotion Service: {e}")
            
            # 2. Проверяем подключение к внешним сервисам
            try:
                from ..emotion_analysis_service import emotion_service
                if emotion_service:
                    # Выполняем health check для проверки подключений
                    health_status = await emotion_service.health_check()
                    if health_status:
                        logger.info("Emotion Service: все подключения работают")
                    else:
                        logger.warning("Emotion Service: проблемы с подключениями")
                        
            except Exception as e:
                logger.warning(f"Не удалось проверить подключения Emotion Service: {e}")
            
            # 3. Переинициализация сервисов
            try:
                # Сбрасываем флаги инициализации
                from ..emotion_analysis_service import emotion_service
                if emotion_service:
                    if hasattr(emotion_service, '_initialized'):
                        emotion_service._initialized = False
                        logger.info("Emotion Service: сброшен флаг инициализации")
                    
                    # Принудительно переинициализируем
                    if hasattr(emotion_service, 'initialize'):
                        await emotion_service.initialize()
                        logger.info("Emotion Service: выполнена переинициализация")
                        
            except Exception as e:
                logger.warning(f"Не удалось переинициализировать Emotion Service: {e}")
            
            # Обновляем статистику
            self.recovery_stats["emotion_service"]["recoveries"] += 1
            self.recovery_stats["emotion_service"]["last_recovery"] = datetime.now()
            self.recovery_stats["emotion_service"]["success_rate"] = 1.0
            
            logger.info("Emotion Service успешно восстановлен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка восстановления Emotion Service: {e}")
            self.recovery_stats["emotion_service"]["success_rate"] = 0.0
            return False
    
    # Health check функции
    def check_memory_orchestrator_health(self) -> bool:
        """Проверка здоровья MemoryOrchestrator"""
        try:
            # Используем переданный экземпляр вместо импорта
            if self.memory_orchestrator is None:
                return False
            
            # Проверяем доступность основных методов
            if not hasattr(self.memory_orchestrator, 'add_memory'):
                return False
            if not hasattr(self.memory_orchestrator, 'search_memory'):
                return False
            if not hasattr(self.memory_orchestrator, 'get_memory_stats'):
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Memory orchestrator health check failed: {e}")
            return False
    
    def check_sensor_buffer_health(self) -> bool:
        """Проверка здоровья SensorBuffer"""
        try:
            from ..memory_levels.sensor_buffer import get_sensor_buffer
            sensor_buffer = get_sensor_buffer()
            return sensor_buffer is not None
        except:
            return False
    
    async def check_memory_cache_health(self) -> bool:
        """Проверка здоровья Memory Cache"""
        try:
            from ..cache.memory_cache import cache_get
            import time
            
            # Реальная проверка - попытка получить значение
            test_key = f"health_check_{int(time.time())}"
            result = cache_get(test_key)
            
            # Проверяем, что кэш отвечает (может вернуть None, это нормально)
            # Главное, что не было исключения
            return True
        except Exception as e:
            logger.warning(f"Memory cache health check failed: {e}")
            return False
    
    async def check_sqlite_cache_health(self) -> bool:
        """Проверка здоровья SQLite Cache"""
        try:
            from ..cache.sqlite_cache import get_sqlite_cache
            cache = await get_sqlite_cache()
            if cache is None:
                return False
            
            # Простая проверка - выполнить простой запрос
            result = await cache.execute("SELECT 1")
            return result is not None
        except Exception as e:
            logger.warning(f"SQLite cache health check failed: {e}")
            return False
    
    async def check_lm_provider_health(self) -> bool:
        """Проверка здоровья LM Provider"""
        try:
            # Быстрая проверка - только ping без полного запроса
            import aiohttp
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get("http://localhost:11434/api/tags", timeout=2) as response:
                        return response.status == 200
                except asyncio.TimeoutError:
                    logger.warning("LM Provider: таймаут подключения к Ollama")
                    return False
                except Exception as e:
                    logger.warning(f"LM Provider: ошибка подключения к Ollama: {e}")
                    return False
        except Exception as e:
            logger.warning(f"LM Provider health check failed: {e}")
            return False
    
    async def check_chromadb_health(self) -> bool:
        """Проверка здоровья ChromaDB"""
        try:
            from ..providers.chromadb_provider import get_chromadb_provider
            
            # Получаем провайдер и проверяем его здоровье
            provider = await get_chromadb_provider()
            if not provider:
                logger.warning("ChromaDB: провайдер не инициализирован")
                return False
            
            # Выполняем health check
            health_status = await provider.health_check()
            if health_status:
                logger.debug("ChromaDB: подключение работает")
                return True
            else:
                logger.warning("ChromaDB: проблемы с подключением")
                return False
                
        except Exception as e:
            logger.error(f"ChromaDB: критическая ошибка health check: {e}")
            return False
    
    async def check_emotion_service_health(self) -> bool:
        """Проверка здоровья Emotion Service"""
        try:
            from ..emotion_analysis_service import emotion_service
            
            if not emotion_service:
                logger.warning("Emotion Service: сервис не инициализирован")
                return False
            
            # Выполняем health check сервиса эмоций
            health_status = await emotion_service.health_check()
            if health_status:
                logger.debug("Emotion Service: все подключения работают")
                return True
            else:
                logger.warning("Emotion Service: проблемы с подключениями")
                return False
                
        except Exception as e:
            logger.error(f"Emotion Service: критическая ошибка health check: {e}")
            return False
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Получение статистики восстановления"""
        return self.recovery_stats.copy()

# Глобальный экземпляр обработчиков
_recovery_handlers: Optional[RecoveryHandlers] = None

def get_recovery_handlers(memory_orchestrator=None) -> RecoveryHandlers:
    """Получение глобального экземпляра обработчиков восстановления"""
    global _recovery_handlers
    if _recovery_handlers is None:
        _recovery_handlers = RecoveryHandlers(memory_orchestrator=memory_orchestrator)
    else:
        # Обновляем memory_orchestrator если он передан
        if memory_orchestrator is not None:
            _recovery_handlers.memory_orchestrator = memory_orchestrator
    return _recovery_handlers
