#!/usr/bin/env python3
"""
Система автоматического восстановления компонентов
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import weakref

logger = logging.getLogger(__name__)

class RecoveryStatus(Enum):
    """Статус восстановления"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"

class RecoveryAction(Enum):
    """Действие восстановления"""
    RESTART = "restart"
    RECONNECT = "reconnect"
    FALLBACK = "fallback"
    SKIP = "skip"
    ALERT = "alert"

@dataclass
class ComponentHealth:
    """Состояние здоровья компонента"""
    name: str
    status: RecoveryStatus
    last_check: datetime
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    recovery_attempts: int = 0
    last_recovery: Optional[datetime] = None
    is_critical: bool = True
    check_interval: int = 30  # секунды
    max_failures: int = 3
    recovery_cooldown: int = 60  # секунды

@dataclass
class RecoveryResult:
    """Результат восстановления"""
    component_name: str
    action: RecoveryAction
    success: bool
    message: str
    timestamp: datetime
    duration_seconds: float
    error: Optional[str] = None

class AutoRecovery:
    """
    Система автоматического восстановления компонентов
    """
    
    def __init__(self, 
                 check_interval: int = 30,  # Проверка каждые 30 секунд
                 max_recovery_attempts: int = 3,  # Максимум попыток восстановления
                 recovery_cooldown: int = 60):  # Кулдаун между попытками
        """
        Инициализация системы восстановления
        
        Args:
            check_interval: Интервал проверки здоровья в секундах
            max_recovery_attempts: Максимум попыток восстановления компонента
            recovery_cooldown: Кулдаун между попытками восстановления в секундах
        """
        self.check_interval = check_interval
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_cooldown = recovery_cooldown
        
        # Состояние системы
        self.is_running = False
        self._recovery_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
        # Компоненты для мониторинга
        self._components: Dict[str, ComponentHealth] = {}
        self._health_checkers: Dict[str, Callable] = {}
        self._recovery_handlers: Dict[str, Callable] = {}
        self._fallback_handlers: Dict[str, Callable] = {}
        
        # История восстановления
        self._recovery_history: List[RecoveryResult] = []
        self._max_history_size = 1000
        
        # Метрики
        self._total_checks = 0
        self._total_failures = 0
        self._total_recoveries = 0
        self._successful_recoveries = 0
        
        logger.info(f"AutoRecovery инициализирован: check_interval={check_interval}s, "
                   f"max_attempts={max_recovery_attempts}, cooldown={recovery_cooldown}s")
    
    def register_component(self, 
                          name: str,
                          health_checker: Callable[[], bool],
                          recovery_handler: Optional[Callable[[], bool]] = None,
                          fallback_handler: Optional[Callable[[], bool]] = None,
                          is_critical: bool = True,
                          check_interval: int = 30,
                          max_failures: int = 3):
        """
        Регистрация компонента для мониторинга
        
        Args:
            name: Имя компонента
            health_checker: Функция проверки здоровья (возвращает bool)
            recovery_handler: Функция восстановления (возвращает bool)
            fallback_handler: Функция fallback (возвращает bool)
            is_critical: Критичность компонента
            check_interval: Интервал проверки в секундах
            max_failures: Максимум последовательных сбоев
        """
        with self._lock:
            self._components[name] = ComponentHealth(
                name=name,
                status=RecoveryStatus.UNKNOWN,
                last_check=datetime.now(),
                is_critical=is_critical,
                check_interval=check_interval,
                max_failures=max_failures
            )
            
            self._health_checkers[name] = health_checker
            if recovery_handler:
                self._recovery_handlers[name] = recovery_handler
            if fallback_handler:
                self._fallback_handlers[name] = fallback_handler
            
            logger.info(f"Зарегистрирован компонент для восстановления: {name} "
                       f"(critical={is_critical}, interval={check_interval}s)")
    
    async def start_monitoring(self):
        """Запуск мониторинга и восстановления"""
        if self.is_running:
            logger.warning("AutoRecovery уже запущен")
            return
        
        self.is_running = True
        self._recovery_task = asyncio.create_task(self._monitoring_loop())
        logger.info("AutoRecovery запущен")
    
    async def stop_monitoring(self):
        """Остановка мониторинга"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass
        logger.info("AutoRecovery остановлен")
    
    async def _monitoring_loop(self):
        """Основной цикл мониторинга"""
        while self.is_running:
            try:
                await self._check_all_components()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в цикле мониторинга восстановления: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_components(self):
        """Проверка всех зарегистрированных компонентов"""
        with self._lock:
            components_to_check = list(self._components.keys())
        
        for component_name in components_to_check:
            try:
                await self._check_component_health(component_name)
            except Exception as e:
                logger.error(f"Ошибка проверки компонента {component_name}: {e}")
    
    async def _check_component_health(self, component_name: str):
        """Проверка здоровья конкретного компонента"""
        with self._lock:
            component = self._components.get(component_name)
            if not component:
                return
            
            # Проверяем, нужно ли проверять компонент сейчас
            time_since_last_check = (datetime.now() - component.last_check).total_seconds()
            if time_since_last_check < component.check_interval:
                return
        
        # Выполняем проверку здоровья
        health_checker = self._health_checkers.get(component_name)
        if not health_checker:
            logger.warning(f"Health checker не найден для компонента {component_name}")
            return
        
        try:
            # Выполняем проверку (может быть async или sync)
            if asyncio.iscoroutinefunction(health_checker):
                is_healthy = await health_checker()
            else:
                is_healthy = health_checker()
            
            # Обновляем состояние компонента
            with self._lock:
                component = self._components[component_name]
                component.last_check = datetime.now()
                self._total_checks += 1
                
                if is_healthy:
                    if component.status != RecoveryStatus.HEALTHY:
                        logger.info(f"Компонент {component_name} восстановлен")
                    component.status = RecoveryStatus.HEALTHY
                    component.consecutive_failures = 0
                    component.last_error = None
                else:
                    component.consecutive_failures += 1
                    self._total_failures += 1
                    
                    if component.consecutive_failures >= component.max_failures:
                        component.status = RecoveryStatus.FAILED
                        logger.warning(f"Компонент {component_name} не работает "
                                     f"({component.consecutive_failures} сбоев подряд)")
                        
                        # Запускаем восстановление
                        await self._attempt_recovery(component_name)
                    else:
                        component.status = RecoveryStatus.DEGRADED
                        logger.warning(f"Компонент {component_name} деградировал "
                                     f"({component.consecutive_failures}/{component.max_failures} сбоев)")
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Ошибка проверки здоровья компонента {component_name}: {error_msg}")
            
            with self._lock:
                component = self._components[component_name]
                component.last_check = datetime.now()
                component.consecutive_failures += 1
                component.last_error = error_msg
                self._total_failures += 1
                
                if component.consecutive_failures >= component.max_failures:
                    component.status = RecoveryStatus.FAILED
                    await self._attempt_recovery(component_name)
                else:
                    component.status = RecoveryStatus.DEGRADED
    
    async def _attempt_recovery(self, component_name: str):
        """Попытка восстановления компонента"""
        with self._lock:
            component = self._components.get(component_name)
            if not component:
                return
            
            # Проверяем кулдаун
            if component.last_recovery:
                time_since_recovery = (datetime.now() - component.last_recovery).total_seconds()
                if time_since_recovery < component.recovery_cooldown:
                    logger.info(f"Компонент {component_name} в кулдауне восстановления "
                               f"({time_since_recovery:.1f}s/{component.recovery_cooldown}s)")
                    return
            
            # Проверяем максимальное количество попыток
            if component.recovery_attempts >= self.max_recovery_attempts:
                logger.error(f"Компонент {component_name} превысил максимальное количество "
                           f"попыток восстановления ({component.recovery_attempts})")
                return
        
        logger.info(f"Попытка восстановления компонента {component_name}")
        
        component.status = RecoveryStatus.RECOVERING
        component.recovery_attempts += 1
        component.last_recovery = datetime.now()
        
        # Определяем действие восстановления
        action = self._determine_recovery_action(component_name)
        
        # Выполняем восстановление
        result = await self._execute_recovery(component_name, action)
        
        # Сохраняем результат
        with self._lock:
            self._recovery_history.append(result)
            if len(self._recovery_history) > self._max_history_size:
                self._recovery_history.pop(0)
            
            self._total_recoveries += 1
            if result.success:
                self._successful_recoveries += 1
                component.status = RecoveryStatus.HEALTHY
                component.consecutive_failures = 0
                component.last_error = None
                logger.info(f"Компонент {component_name} успешно восстановлен")
            else:
                component.status = RecoveryStatus.FAILED
                component.last_error = result.error
                logger.error(f"Не удалось восстановить компонент {component_name}: {result.message}")
    
    def _determine_recovery_action(self, component_name: str) -> RecoveryAction:
        """Определение действия восстановления"""
        # Проверяем наличие обработчиков
        if component_name in self._recovery_handlers:
            return RecoveryAction.RESTART
        elif component_name in self._fallback_handlers:
            return RecoveryAction.FALLBACK
        else:
            return RecoveryAction.ALERT
    
    async def _execute_recovery(self, component_name: str, action: RecoveryAction) -> RecoveryResult:
        """Выполнение восстановления"""
        start_time = time.time()
        
        try:
            if action == RecoveryAction.RESTART:
                handler = self._recovery_handlers.get(component_name)
                if handler:
                    if asyncio.iscoroutinefunction(handler):
                        success = await handler()
                    else:
                        success = handler()
                    
                    return RecoveryResult(
                        component_name=component_name,
                        action=action,
                        success=success,
                        message=f"Restart {'successful' if success else 'failed'}",
                        timestamp=datetime.now(),
                        duration_seconds=time.time() - start_time
                    )
            
            elif action == RecoveryAction.FALLBACK:
                handler = self._fallback_handlers.get(component_name)
                if handler:
                    if asyncio.iscoroutinefunction(handler):
                        success = await handler()
                    else:
                        success = handler()
                    
                    return RecoveryResult(
                        component_name=component_name,
                        action=action,
                        success=success,
                        message=f"Fallback {'activated' if success else 'failed'}",
                        timestamp=datetime.now(),
                        duration_seconds=time.time() - start_time
                    )
            
            elif action == RecoveryAction.ALERT:
                # Отправляем алерт (пока что только логируем)
                logger.critical(f"КРИТИЧЕСКИЙ СБОЙ: Компонент {component_name} не работает "
                               f"и не может быть восстановлен автоматически")
                
                return RecoveryResult(
                    component_name=component_name,
                    action=action,
                    success=False,
                    message="Alert sent - manual intervention required",
                    timestamp=datetime.now(),
                    duration_seconds=time.time() - start_time
                )
            
            else:
                return RecoveryResult(
                    component_name=component_name,
                    action=action,
                    success=False,
                    message=f"Unknown recovery action: {action}",
                    timestamp=datetime.now(),
                    duration_seconds=time.time() - start_time,
                    error="Unknown action"
                )
        
        except Exception as e:
            return RecoveryResult(
                component_name=component_name,
                action=action,
                success=False,
                message=f"Recovery failed with exception: {str(e)}",
                timestamp=datetime.now(),
                duration_seconds=time.time() - start_time,
                error=str(e)
            )
    
    def get_component_status(self, component_name: str) -> Optional[ComponentHealth]:
        """Получение статуса компонента"""
        with self._lock:
            return self._components.get(component_name)
    
    def get_all_components_status(self) -> Dict[str, ComponentHealth]:
        """Получение статуса всех компонентов"""
        with self._lock:
            return self._components.copy()
    
    def get_recovery_history(self, limit: Optional[int] = None) -> List[RecoveryResult]:
        """Получение истории восстановления"""
        with self._lock:
            if limit:
                return self._recovery_history[-limit:]
            return self._recovery_history.copy()
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Получение статистики восстановления"""
        with self._lock:
            total_components = len(self._components)
            healthy_components = sum(1 for c in self._components.values() 
                                   if c.status == RecoveryStatus.HEALTHY)
            failed_components = sum(1 for c in self._components.values() 
                                  if c.status == RecoveryStatus.FAILED)
            recovering_components = sum(1 for c in self._components.values() 
                                      if c.status == RecoveryStatus.RECOVERING)
            
            success_rate = (self._successful_recoveries / self._total_recoveries * 100 
                          if self._total_recoveries > 0 else 0)
            
            return {
                "is_running": self.is_running,
                "check_interval": self.check_interval,
                "max_recovery_attempts": self.max_recovery_attempts,
                "recovery_cooldown": self.recovery_cooldown,
                "total_components": total_components,
                "healthy_components": healthy_components,
                "failed_components": failed_components,
                "recovering_components": recovering_components,
                "total_checks": self._total_checks,
                "total_failures": self._total_failures,
                "total_recoveries": self._total_recoveries,
                "successful_recoveries": self._successful_recoveries,
                "recovery_success_rate": success_rate,
                "history_size": len(self._recovery_history)
            }
    
    async def force_recovery(self, component_name: str) -> RecoveryResult:
        """Принудительное восстановление компонента"""
        logger.info(f"Принудительное восстановление компонента {component_name}")
        
        with self._lock:
            component = self._components.get(component_name)
            if not component:
                return RecoveryResult(
                    component_name=component_name,
                    action=RecoveryAction.SKIP,
                    success=False,
                    message="Component not found",
                    timestamp=datetime.now(),
                    duration_seconds=0.0,
                    error="Component not registered"
                )
        
        # Сбрасываем счетчики
        component.recovery_attempts = 0
        component.consecutive_failures = 0
        
        # Выполняем восстановление
        action = self._determine_recovery_action(component_name)
        result = await self._execute_recovery(component_name, action)
        
        # Сохраняем результат
        with self._lock:
            self._recovery_history.append(result)
            if len(self._recovery_history) > self._max_history_size:
                self._recovery_history.pop(0)
        
        return result

# Глобальный экземпляр системы восстановления
_auto_recovery: Optional[AutoRecovery] = None

def get_auto_recovery() -> AutoRecovery:
    """Получение глобального экземпляра системы восстановления"""
    global _auto_recovery
    if _auto_recovery is None:
        _auto_recovery = AutoRecovery()
    return _auto_recovery

async def start_auto_recovery():
    """Запуск глобальной системы восстановления"""
    recovery = get_auto_recovery()
    await recovery.start_monitoring()

async def stop_auto_recovery():
    """Остановка глобальной системы восстановления"""
    global _auto_recovery
    if _auto_recovery:
        await _auto_recovery.stop_monitoring()
        _auto_recovery = None
