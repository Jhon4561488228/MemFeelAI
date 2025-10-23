#!/usr/bin/env python3
"""
Мониторинг и предотвращение утечек памяти
"""

import asyncio
import gc
import logging
import os
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import weakref

logger = logging.getLogger(__name__)

class MemoryAlertLevel(Enum):
    """Уровень предупреждения о памяти"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class MemoryAlert:
    """Предупреждение о памяти"""
    timestamp: datetime
    level: MemoryAlertLevel
    component: str
    message: str
    memory_usage_mb: float
    threshold_mb: float
    action_taken: Optional[str] = None

@dataclass
class MemoryStats:
    """Статистика использования памяти"""
    timestamp: datetime
    total_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    process_memory_mb: float
    process_memory_percent: float
    gc_objects: int
    gc_collections: Dict[str, int]

class MemoryMonitor:
    """
    Мониторинг и управление памятью системы
    """
    
    def __init__(self, 
                 check_interval: int = 60,  # Проверка каждые 60 секунд
                 warning_threshold: float = 80.0,  # 80% использования памяти
                 critical_threshold: float = 90.0,  # 90% использования памяти
                 cleanup_threshold: float = 75.0):  # 75% - начало очистки
        """
        Инициализация монитора памяти
        
        Args:
            check_interval: Интервал проверки в секундах
            warning_threshold: Порог предупреждения (% использования памяти)
            critical_threshold: Критический порог (% использования памяти)
            cleanup_threshold: Порог начала очистки (% использования памяти)
        """
        self.check_interval = check_interval
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.cleanup_threshold = cleanup_threshold
        
        # Состояние мониторинга
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
        # Статистика и алерты
        self.memory_stats: List[MemoryStats] = []
        self.memory_alerts: List[MemoryAlert] = []
        self.max_stats_history = 1000  # Максимум записей статистики
        self.max_alerts_history = 500  # Максимум алертов
        
        # Компоненты для мониторинга
        self._monitored_components: Dict[str, Callable] = {}
        self._cleanup_handlers: Dict[str, Callable] = {}
        
        # Метрики
        self._total_cleanups = 0
        self._total_memory_freed_mb = 0.0
        self._last_cleanup = None
        
        logger.info(f"MemoryMonitor инициализирован: check_interval={check_interval}s, "
                   f"warning={warning_threshold}%, critical={critical_threshold}%, "
                   f"cleanup={cleanup_threshold}%")
    
    def register_component(self, name: str, 
                          monitor_func: Callable[[], Dict[str, Any]],
                          cleanup_func: Optional[Callable[[], int]] = None):
        """
        Регистрация компонента для мониторинга
        
        Args:
            name: Имя компонента
            monitor_func: Функция для получения статистики компонента
            cleanup_func: Функция для очистки компонента (возвращает количество освобожденных MB)
        """
        with self._lock:
            self._monitored_components[name] = monitor_func
            if cleanup_func:
                self._cleanup_handlers[name] = cleanup_func
            logger.info(f"Зарегистрирован компонент для мониторинга: {name}")
    
    async def start_monitoring(self):
        """Запуск мониторинга памяти"""
        if self.is_running:
            logger.warning("MemoryMonitor уже запущен")
            return
        
        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("MemoryMonitor запущен")
    
    async def stop_monitoring(self):
        """Остановка мониторинга памяти"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("MemoryMonitor остановлен")
    
    async def _monitoring_loop(self):
        """Основной цикл мониторинга"""
        while self.is_running:
            try:
                await self._check_memory_usage()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в цикле мониторинга памяти: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_memory_usage(self):
        """Проверка использования памяти"""
        try:
            # Получаем системную статистику памяти
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Статистика сборщика мусора
            gc_stats = gc.get_stats()
            gc_collections = {f"gen_{i}": stat["collections"] for i, stat in enumerate(gc_stats)}
            
            # Создаем запись статистики
            stats = MemoryStats(
                timestamp=datetime.now(),
                total_memory_mb=memory.total / (1024 * 1024),
                available_memory_mb=memory.available / (1024 * 1024),
                memory_percent=memory.percent,
                process_memory_mb=process_memory.rss / (1024 * 1024),
                process_memory_percent=(process_memory.rss / memory.total) * 100,
                gc_objects=len(gc.get_objects()),
                gc_collections=gc_collections
            )
            
            # Сохраняем статистику
            with self._lock:
                self.memory_stats.append(stats)
                if len(self.memory_stats) > self.max_stats_history:
                    self.memory_stats.pop(0)
            
            # Проверяем пороги и создаем алерты
            await self._check_memory_thresholds(stats)
            
            # Логируем статистику (раз в 10 проверок)
            if len(self.memory_stats) % 10 == 0:
                logger.info(f"Memory stats: {stats.memory_percent:.1f}% system, "
                           f"{stats.process_memory_mb:.1f}MB process, "
                           f"{stats.gc_objects} GC objects")
            
        except Exception as e:
            logger.error(f"Ошибка проверки памяти: {e}")
    
    async def _check_memory_thresholds(self, stats: MemoryStats):
        """Проверка порогов памяти и создание алертов"""
        memory_percent = stats.memory_percent
        
        # Определяем уровень предупреждения
        alert_level = None
        if memory_percent >= self.critical_threshold:
            alert_level = MemoryAlertLevel.CRITICAL
        elif memory_percent >= self.warning_threshold:
            alert_level = MemoryAlertLevel.WARNING
        elif memory_percent >= self.cleanup_threshold:
            alert_level = MemoryAlertLevel.INFO
        
        if alert_level:
            # Создаем алерт
            alert = MemoryAlert(
                timestamp=datetime.now(),
                level=alert_level,
                component="system",
                message=f"Memory usage: {memory_percent:.1f}%",
                memory_usage_mb=stats.total_memory_mb - stats.available_memory_mb,
                threshold_mb=stats.total_memory_mb * (self.critical_threshold / 100)
            )
            
            # Сохраняем алерт
            with self._lock:
                self.memory_alerts.append(alert)
                if len(self.memory_alerts) > self.max_alerts_history:
                    self.memory_alerts.pop(0)
            
            # Логируем алерт
            if alert_level == MemoryAlertLevel.CRITICAL:
                logger.critical(f"CRITICAL: {alert.message}")
            elif alert_level == MemoryAlertLevel.WARNING:
                logger.warning(f"WARNING: {alert.message}")
            else:
                logger.info(f"INFO: {alert.message}")
            
            # Выполняем очистку при необходимости
            if memory_percent >= self.cleanup_threshold:
                await self._perform_cleanup(alert)
    
    async def _perform_cleanup(self, alert: MemoryAlert):
        """Выполнение очистки памяти"""
        try:
            logger.info("Начинаем очистку памяти...")
            
            total_freed_mb = 0.0
            cleanup_actions = []
            
            # 1. Принудительная сборка мусора
            before_objects = len(gc.get_objects())
            collected = gc.collect()
            after_objects = len(gc.get_objects())
            
            if collected > 0:
                cleanup_actions.append(f"GC collected {collected} objects")
                logger.info(f"Garbage collection: {before_objects} -> {after_objects} objects")
            
            # 2. Очистка зарегистрированных компонентов
            for component_name, cleanup_func in self._cleanup_handlers.items():
                try:
                    freed_mb = await self._safe_cleanup(component_name, cleanup_func)
                    if freed_mb > 0:
                        total_freed_mb += freed_mb
                        cleanup_actions.append(f"{component_name}: {freed_mb:.1f}MB")
                except Exception as e:
                    logger.warning(f"Ошибка очистки компонента {component_name}: {e}")
            
            # 3. Очистка старых записей статистики
            with self._lock:
                old_stats_count = len(self.memory_stats)
                old_alerts_count = len(self.memory_alerts)
                
                # Оставляем только последние 500 записей
                if old_stats_count > 500:
                    self.memory_stats = self.memory_stats[-500:]
                    cleanup_actions.append(f"Cleaned {old_stats_count - 500} old stats")
                
                if old_alerts_count > 250:
                    self.memory_alerts = self.memory_alerts[-250:]
                    cleanup_actions.append(f"Cleaned {old_alerts_count - 250} old alerts")
            
            # Обновляем метрики
            self._total_cleanups += 1
            self._total_memory_freed_mb += total_freed_mb
            self._last_cleanup = datetime.now()
            
            # Обновляем алерт
            alert.action_taken = "; ".join(cleanup_actions) if cleanup_actions else "No cleanup needed"
            
            logger.info(f"Очистка памяти завершена: освобождено {total_freed_mb:.1f}MB, "
                       f"действия: {alert.action_taken}")
            
        except Exception as e:
            logger.error(f"Ошибка при очистке памяти: {e}")
    
    async def _safe_cleanup(self, component_name: str, cleanup_func: Callable) -> float:
        """Безопасное выполнение очистки компонента"""
        try:
            if asyncio.iscoroutinefunction(cleanup_func):
                result = await cleanup_func()
            else:
                result = cleanup_func()
            
            # Ожидаем, что функция возвращает количество освобожденных MB
            if isinstance(result, (int, float)):
                return float(result)
            else:
                logger.warning(f"Компонент {component_name} вернул неожиданный результат: {result}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Ошибка очистки компонента {component_name}: {e}")
            return 0.0
    
    def get_memory_stats(self, limit: Optional[int] = None) -> List[MemoryStats]:
        """Получение статистики памяти"""
        with self._lock:
            if limit:
                return self.memory_stats[-limit:]
            return self.memory_stats.copy()
    
    def get_memory_alerts(self, limit: Optional[int] = None) -> List[MemoryAlert]:
        """Получение алертов памяти"""
        with self._lock:
            if limit:
                return self.memory_alerts[-limit:]
            return self.memory_alerts.copy()
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Получение статуса мониторинга"""
        with self._lock:
            latest_stats = self.memory_stats[-1] if self.memory_stats else None
            
            return {
                "is_running": self.is_running,
                "check_interval": self.check_interval,
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold,
                "cleanup_threshold": self.cleanup_threshold,
                "monitored_components": len(self._monitored_components),
                "cleanup_handlers": len(self._cleanup_handlers),
                "total_cleanups": self._total_cleanups,
                "total_memory_freed_mb": self._total_memory_freed_mb,
                "last_cleanup": self._last_cleanup.isoformat() if self._last_cleanup else None,
                "stats_history_size": len(self.memory_stats),
                "alerts_history_size": len(self.memory_alerts),
                "latest_memory_percent": latest_stats.memory_percent if latest_stats else None,
                "latest_process_memory_mb": latest_stats.process_memory_mb if latest_stats else None
            }
    
    async def force_cleanup(self) -> Dict[str, Any]:
        """Принудительная очистка памяти"""
        logger.info("Выполняется принудительная очистка памяти...")
        
        # Создаем фиктивный алерт для запуска очистки
        fake_alert = MemoryAlert(
            timestamp=datetime.now(),
            level=MemoryAlertLevel.INFO,
            component="manual",
            message="Manual cleanup requested",
            memory_usage_mb=0.0,
            threshold_mb=0.0
        )
        
        await self._perform_cleanup(fake_alert)
        
        return {
            "cleanup_completed": True,
            "total_cleanups": self._total_cleanups,
            "total_memory_freed_mb": self._total_memory_freed_mb,
            "last_cleanup": self._last_cleanup.isoformat() if self._last_cleanup else None
        }

# Глобальный экземпляр монитора
_memory_monitor: Optional[MemoryMonitor] = None

def get_memory_monitor() -> MemoryMonitor:
    """Получение глобального экземпляра монитора памяти"""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor

async def start_memory_monitoring():
    """Запуск глобального мониторинга памяти"""
    monitor = get_memory_monitor()
    await monitor.start_monitoring()

async def stop_memory_monitoring():
    """Остановка глобального мониторинга памяти"""
    global _memory_monitor
    if _memory_monitor:
        await _memory_monitor.stop_monitoring()
        _memory_monitor = None
