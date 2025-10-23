"""
Performance Optimizer для AIRI Memory System
Обеспечивает оптимизацию производительности системы памяти
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from loguru import logger
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque

class PerformanceOptimizer:
    """Оптимизатор производительности системы памяти"""
    
    def __init__(self, config_path: str = "config/performance_config.yaml"):
        """Инициализация оптимизатора"""
        self.config = self._load_config(config_path)
        self.metrics = defaultdict(list)
        self.request_queue = deque()
        self.batch_cache = {}
        self.cleanup_task = None
        self.monitoring_task = None
        self._start_background_tasks()
        logger.info("Performance Optimizer инициализирован")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Конфигурационный файл {config_path} не найден, используются настройки по умолчанию")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Настройки по умолчанию"""
        return {
            "cache_settings": {
                "max_size": 10000,
                "ttl_hours": 24,
                "cleanup_interval": 3600
            },
            "batch_settings": {
                "embedding_batch_size": 32,
                "search_batch_size": 16,
                "memory_batch_size": 50,
                "max_concurrent_requests": 5
            },
            "search_settings": {
                "default_limit": 10,
                "max_limit": 100,
                "similarity_threshold": 0.1,
                "search_timeout": 5.0,
                "prefilter_multiplier": 3
            },
            "optimization": {
                "enable_indexing": True,
                "enable_compression": True,
                "enable_caching": True,
                "enable_batching": True,
                "enable_prefetching": True
            }
        }
    
    def _start_background_tasks(self):
        """Запуск фоновых задач"""
        if self.config["optimization"]["enable_caching"]:
            self.cleanup_task = asyncio.create_task(self._cleanup_cache_periodically())
        
        if self.config.get("monitoring", {}).get("enable_metrics", False):
            self.monitoring_task = asyncio.create_task(self._monitor_performance())
    
    async def _cleanup_cache_periodically(self):
        """Периодическая очистка кэша"""
        cleanup_interval = self.config["cache_settings"]["cleanup_interval"]
        while True:
            try:
                await asyncio.sleep(cleanup_interval)
                await self._cleanup_expired_cache()
            except Exception as e:
                logger.error(f"Ошибка очистки кэша: {e}")
    
    async def _monitor_performance(self):
        """Мониторинг производительности"""
        metrics_interval = self.config.get("monitoring", {}).get("metrics_interval", 60)
        while True:
            try:
                await asyncio.sleep(metrics_interval)
                await self._collect_metrics()
            except Exception as e:
                logger.error(f"Ошибка сбора метрик: {e}")
    
    async def _cleanup_expired_cache(self):
        """Очистка устаревшего кэша"""
        try:
            ttl_hours = self.config["cache_settings"]["ttl_hours"]
            cutoff_time = datetime.now() - timedelta(hours=ttl_hours)
            
            # Очищаем устаревшие записи из кэша
            expired_keys = []
            for key, value in self.batch_cache.items():
                if isinstance(value, dict) and "timestamp" in value:
                    if value["timestamp"] < cutoff_time:
                        expired_keys.append(key)
            
            for key in expired_keys:
                del self.batch_cache[key]
            
            if expired_keys:
                logger.info(f"Очищено {len(expired_keys)} устаревших записей из кэша")
                
        except Exception as e:
            logger.error(f"Ошибка очистки устаревшего кэша: {e}")
    
    async def _collect_metrics(self):
        """Сбор метрик производительности"""
        try:
            current_time = time.time()
            
            # Собираем метрики
            metrics = {
                "timestamp": current_time,
                "cache_size": len(self.batch_cache),
                "request_queue_size": len(self.request_queue),
                "memory_usage": self._get_memory_usage()
            }
            
            # Сохраняем метрики
            self.metrics["performance"].append(metrics)
            
            # Ограничиваем размер истории метрик
            max_metrics = 1000
            if len(self.metrics["performance"]) > max_metrics:
                self.metrics["performance"] = self.metrics["performance"][-max_metrics:]
            
            logger.debug(f"Метрики собраны: {metrics}")
            
        except Exception as e:
            logger.error(f"Ошибка сбора метрик: {e}")
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Получение информации об использовании памяти"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    async def optimize_search_request(
        self, 
        query: str, 
        user_id: str, 
        limit: int,
        similarity_threshold: float
    ) -> Dict[str, Any]:
        """Оптимизация запроса поиска"""
        try:
            start_time = time.time()
            
            # Применяем оптимизации
            optimized_limit = min(limit, self.config["search_settings"]["max_limit"])
            optimized_threshold = max(similarity_threshold, self.config["search_settings"]["similarity_threshold"])
            
            # Проверяем кэш
            cache_key = f"search:{user_id}:{query}:{optimized_limit}:{optimized_threshold}"
            if cache_key in self.batch_cache:
                logger.debug("Результат поиска найден в кэше")
                return self.batch_cache[cache_key]["result"]
            
            # Создаем оптимизированный запрос
            optimized_request = {
                "query": query,
                "user_id": user_id,
                "limit": optimized_limit,
                "similarity_threshold": optimized_threshold,
                "prefilter_multiplier": self.config["search_settings"]["prefilter_multiplier"]
            }
            
            # Засекаем время выполнения
            execution_time = time.time() - start_time
            
            # Сохраняем в кэш
            if len(self.batch_cache) < self.config["cache_settings"]["max_size"]:
                self.batch_cache[cache_key] = {
                    "result": optimized_request,
                    "timestamp": datetime.now()
                }
            
            # Логируем производительность
            if self.config.get("monitoring", {}).get("log_performance", False):
                logger.info(f"Поиск оптимизирован за {execution_time:.3f}s")
            
            return optimized_request
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации запроса поиска: {e}")
            return {
                "query": query,
                "user_id": user_id,
                "limit": limit,
                "similarity_threshold": similarity_threshold
            }
    
    async def optimize_batch_operations(
        self, 
        operations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Оптимизация батчевых операций"""
        try:
            if not self.config["optimization"]["enable_batching"]:
                return operations
            
            # Группируем операции по типу
            grouped_ops = defaultdict(list)
            for op in operations:
                op_type = op.get("type", "unknown")
                grouped_ops[op_type].append(op)
            
            # Оптимизируем каждую группу
            optimized_ops = []
            for op_type, ops in grouped_ops.items():
                batch_size = self.config["batch_settings"].get(f"{op_type}_batch_size", 32)
                
                # Разбиваем на батчи
                for i in range(0, len(ops), batch_size):
                    batch = ops[i:i + batch_size]
                    optimized_ops.append({
                        "type": op_type,
                        "batch": batch,
                        "size": len(batch)
                    })
            
            logger.info(f"Оптимизировано {len(operations)} операций в {len(optimized_ops)} батчей")
            return optimized_ops
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации батчевых операций: {e}")
            return operations
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности"""
        try:
            current_time = time.time()
            
            # Анализируем метрики
            if "performance" in self.metrics and self.metrics["performance"]:
                recent_metrics = self.metrics["performance"][-10:]  # Последние 10 записей
                
                avg_cache_size = sum(m["cache_size"] for m in recent_metrics) / len(recent_metrics)
                avg_queue_size = sum(m["request_queue_size"] for m in recent_metrics) / len(recent_metrics)
                
                return {
                    "current_cache_size": len(self.batch_cache),
                    "current_queue_size": len(self.request_queue),
                    "average_cache_size": avg_cache_size,
                    "average_queue_size": avg_queue_size,
                    "total_metrics_collected": len(self.metrics["performance"]),
                    "optimization_enabled": self.config["optimization"],
                    "last_updated": current_time
                }
            else:
                return {
                    "current_cache_size": len(self.batch_cache),
                    "current_queue_size": len(self.request_queue),
                    "total_metrics_collected": 0,
                    "optimization_enabled": self.config["optimization"],
                    "last_updated": current_time
                }
                
        except Exception as e:
            logger.error(f"Ошибка получения метрик производительности: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Очистка ресурсов"""
        try:
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            # Очищаем кэш
            self.batch_cache.clear()
            self.request_queue.clear()
            
            logger.info("Performance Optimizer очищен")
            
        except Exception as e:
            logger.error(f"Ошибка очистки Performance Optimizer: {e}")
