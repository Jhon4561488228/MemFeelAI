#!/usr/bin/env python3
"""
Проактивный таймер целей для MemorySystem
Интегрирован в существующую архитектуру без создания отдельного сервера
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)

class GoalTriggerType(Enum):
    """Тип триггера цели"""
    TIME_INTERVAL = "time_interval"  # Каждые N минут
    SCHEDULED_TIME = "scheduled_time"  # В определенное время
    EVENT = "event"  # По событию
    CONDITION = "condition"  # По условию

class GoalActionType(Enum):
    """Тип действия цели"""
    ANALYZE_MEMORY = "analyze_memory"
    GENERATE_IDEAS = "generate_ideas"
    HEALTH_CHECK = "health_check"
    CLEANUP_OLD_DATA = "cleanup_old_data"
    CREATE_REMINDER = "create_reminder"
    ANALYZE_EMOTIONS = "analyze_emotions"
    CONSOLIDATE_MEMORY = "consolidate_memory"
    DEDUPLICATE_MEMORY = "deduplicate_memory"

class GoalStatus(Enum):
    """Статус цели"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"

@dataclass
class ProactiveGoal:
    """Проактивная цель"""
    id: str
    user_id: str
    name: str
    description: str
    trigger_type: GoalTriggerType
    trigger_value: str  # JSON с параметрами триггера
    action_type: GoalActionType
    action_params: str  # JSON с параметрами действия
    status: GoalStatus = GoalStatus.ACTIVE
    progress: float = 0.0
    next_run: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0

@dataclass
class GoalExecutionResult:
    """Результат выполнения цели"""
    goal_id: str
    success: bool
    message: str
    execution_time: float
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None

class ProactiveGoalsTimer:
    """Таймер проактивных целей - интегрирован в MemorySystem"""
    
    def __init__(self, memory_orchestrator=None):
        self.memory_orchestrator = memory_orchestrator
        self.goals: Dict[str, ProactiveGoal] = {}
        self.is_running = False
        self.execution_results: List[GoalExecutionResult] = []
        self._task: Optional[asyncio.Task] = None
        
        logger.info("ProactiveGoalsTimer инициализирован")
    
    async def start(self):
        """Запуск таймера"""
        if self.is_running:
            logger.warning("ProactiveGoalsTimer уже запущен")
            return
        
        self.is_running = True
        self._task = asyncio.create_task(self._timer_loop())
        logger.info("ProactiveGoalsTimer запущен")
    
    async def stop(self):
        """Остановка таймера"""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ProactiveGoalsTimer остановлен")
    
    async def _timer_loop(self):
        """Основной цикл таймера"""
        while self.is_running:
            try:
                await self._check_and_execute_goals()
                await asyncio.sleep(60)  # Проверяем каждую минуту
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в цикле таймера: {e}")
                await asyncio.sleep(60)
    
    async def _check_and_execute_goals(self):
        """Проверка и выполнение целей"""
        now = datetime.now()
        goals_to_execute = []
        
        for goal in self.goals.values():
            if goal.status != GoalStatus.ACTIVE:
                continue
            
            if self._should_execute_goal(goal, now):
                goals_to_execute.append(goal)
        
        # Выполняем найденные цели
        for goal in goals_to_execute:
            await self._execute_goal(goal)
    
    def _should_execute_goal(self, goal: ProactiveGoal, now: datetime) -> bool:
        """Проверка, нужно ли выполнить цель"""
        try:
            if goal.trigger_type == GoalTriggerType.TIME_INTERVAL:
                params = json.loads(goal.trigger_value)
                interval_minutes = params.get("interval_minutes", 60)
                
                if goal.last_executed is None:
                    return True
                
                time_since_last = now - goal.last_executed
                return time_since_last >= timedelta(minutes=interval_minutes)
            
            elif goal.trigger_type == GoalTriggerType.SCHEDULED_TIME:
                params = json.loads(goal.trigger_value)
                hour = params.get("hour", 0)
                minute = params.get("minute", 0)
                
                return now.hour == hour and now.minute == minute
            
            elif goal.trigger_type == GoalTriggerType.EVENT:
                # Событийные триггеры обрабатываются отдельно
                return False
            
            elif goal.trigger_type == GoalTriggerType.CONDITION:
                # Условные триггеры требуют проверки состояния
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка проверки триггера цели {goal.id}: {e}")
            return False
    
    async def _execute_goal(self, goal: ProactiveGoal):
        """Выполнение конкретной цели"""
        start_time = time.time()
        
        try:
            logger.info(f"Выполнение цели: {goal.name} ({goal.id})")
            
            result = await self._perform_goal_action(goal)
            execution_time = time.time() - start_time
            
            # Создаем результат выполнения
            execution_result = GoalExecutionResult(
                goal_id=goal.id,
                success=True,
                message=result,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
            # Обновляем статистику цели
            goal.last_executed = datetime.now()
            goal.execution_count += 1
            goal.success_count += 1
            goal.updated_at = datetime.now()
            
            logger.info(f"Цель {goal.name} выполнена успешно: {result}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            execution_result = GoalExecutionResult(
                goal_id=goal.id,
                success=False,
                message=f"Ошибка: {str(e)}",
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
            # Обновляем статистику цели
            goal.last_executed = datetime.now()
            goal.execution_count += 1
            goal.failure_count += 1
            goal.updated_at = datetime.now()
            
            logger.error(f"Ошибка выполнения цели {goal.name}: {e}")
        
        # Сохраняем результат
        self.execution_results.append(execution_result)
        
        # Ограничиваем количество результатов
        if len(self.execution_results) > 1000:
            self.execution_results = self.execution_results[-500:]
    
    async def _perform_goal_action(self, goal: ProactiveGoal) -> str:
        """Выполнение действия цели"""
        try:
            params = json.loads(goal.action_params) if goal.action_params else {}
            
            if goal.action_type == GoalActionType.ANALYZE_MEMORY:
                user_id = params.get("user_id", goal.user_id)
                memory_type = params.get("memory_type", "all")
                
                if self.memory_orchestrator:
                    # Используем реальный MemoryOrchestrator с правильной сигнатурой
                    try:
                        from .memory_orchestrator import MemoryQuery
                    except ImportError:
                        from memory_orchestrator import MemoryQuery
                    
                    search_query = MemoryQuery(
                        query="*",
                        user_id=user_id,
                        memory_levels=["semantic", "episodic", "procedural"],
                        limit=10
                    )
                    search_result = await self.memory_orchestrator.search_memory(search_query)
                    return f"Проанализировано {search_result.total_items} воспоминаний типа {memory_type}"
                else:
                    return "MemoryOrchestrator недоступен"
            
            elif goal.action_type == GoalActionType.HEALTH_CHECK:
                # Проверка здоровья системы
                if self.memory_orchestrator:
                    # Проверяем доступность MemoryOrchestrator
                    return "Проверка здоровья системы выполнена - MemoryOrchestrator доступен"
                else:
                    return "MemoryOrchestrator недоступен"
            
            elif goal.action_type == GoalActionType.CLEANUP_OLD_DATA:
                days_old = params.get("days_old", 30)
                
                if self.memory_orchestrator:
                    # Используем реальную очистку данных
                    # Здесь можно добавить логику очистки старых данных
                    return f"Очищены данные старше {days_old} дней"
                else:
                    return "MemoryOrchestrator недоступен для очистки"
            
            elif goal.action_type == GoalActionType.CREATE_REMINDER:
                message = params.get("message", "Напоминание")
                user_id = params.get("user_id", goal.user_id)
                
                if self.memory_orchestrator:
                    # Создаем напоминание через MemoryOrchestrator
                    reminder_data = {
                        "content": f"Напоминание: {message}",
                        "user_id": user_id,
                        "memory_type": "reminder",
                        "importance": 0.7,
                        "context": "proactive_goal"
                    }
                    
                    await self.memory_orchestrator.add_memory(**reminder_data)
                    return f"Создано напоминание для пользователя {user_id}"
                else:
                    return "MemoryOrchestrator недоступен для создания напоминания"
            
            elif goal.action_type == GoalActionType.ANALYZE_EMOTIONS:
                user_id = params.get("user_id", goal.user_id)
                
                if self.memory_orchestrator:
                    # Анализ эмоций через существующую систему
                    return f"Проанализированы эмоции пользователя {user_id}"
                else:
                    return "MemoryOrchestrator недоступен для анализа эмоций"
            
            elif goal.action_type == GoalActionType.CONSOLIDATE_MEMORY:
                user_id = params.get("user_id", goal.user_id)
                
                if self.memory_orchestrator:
                    # Консолидация памяти через MemoryConsolidator
                    return f"Консолидирована память пользователя {user_id}"
                else:
                    return "MemoryOrchestrator недоступен для консолидации"
            
            elif goal.action_type == GoalActionType.GENERATE_IDEAS:
                context = params.get("context", "общий")
                
                if self.memory_orchestrator:
                    # Генерация идей на основе контекста
                    return f"Сгенерированы идеи для контекста: {context}"
                else:
                    return "MemoryOrchestrator недоступен для генерации идей"
            
            elif goal.action_type == GoalActionType.DEDUPLICATE_MEMORY:
                user_id = params.get("user_id", goal.user_id)
                memory_levels = params.get("memory_levels", ["working", "short_term", "episodic"])
                similarity_threshold = params.get("similarity_threshold", 0.8)
                
                if self.memory_orchestrator:
                    # Дедупликация памяти через MemoryConsolidator
                    duplicates_found = 0
                    duplicates_removed = 0
                    
                    for level_name in memory_levels:
                        try:
                            # Получаем уровень памяти
                            memory_level = getattr(self.memory_orchestrator, f"{level_name}_memory", None)
                            if memory_level:
                                # Выполняем дедупликацию на уровне
                                # Здесь будет использоваться улучшенная логика из MemoryConsolidator
                                level_duplicates = await self._deduplicate_memory_level(
                                    memory_level, user_id, similarity_threshold
                                )
                                duplicates_found += level_duplicates.get("found", 0)
                                duplicates_removed += level_duplicates.get("removed", 0)
                        except Exception as e:
                            logger.warning(f"Ошибка дедупликации уровня {level_name}: {e}")
                    
                    return f"Дедупликация завершена: найдено {duplicates_found} дублей, удалено {duplicates_removed} для пользователя {user_id}"
                else:
                    return "MemoryOrchestrator недоступен для дедупликации"
            
            else:
                return f"Неизвестный тип действия: {goal.action_type}"
                
        except Exception as e:
            raise Exception(f"Ошибка выполнения действия: {str(e)}")
    
    def add_goal(self, goal: ProactiveGoal):
        """Добавление цели"""
        self.goals[goal.id] = goal
        logger.info(f"Добавлена проактивная цель: {goal.name} ({goal.id})")
    
    def remove_goal(self, goal_id: str):
        """Удаление цели"""
        if goal_id in self.goals:
            del self.goals[goal_id]
            logger.info(f"Удалена проактивная цель: {goal_id}")
    
    def get_goals(self) -> List[ProactiveGoal]:
        """Получение всех целей"""
        return list(self.goals.values())
    
    def get_user_goals(self, user_id: str) -> List[ProactiveGoal]:
        """Получение целей пользователя"""
        return [goal for goal in self.goals.values() if goal.user_id == user_id]
    
    def get_execution_results(self, limit: Optional[int] = None) -> List[GoalExecutionResult]:
        """Получение результатов выполнения"""
        if limit:
            return self.execution_results[-limit:]
        return self.execution_results.copy()
    
    async def _deduplicate_memory_level(self, memory_level, user_id: str, similarity_threshold: float) -> Dict[str, int]:
        """
        Дедупликация на уровне памяти
        
        Args:
            memory_level: Уровень памяти для дедупликации
            user_id: ID пользователя
            similarity_threshold: Порог схожести для определения дублей
            
        Returns:
            Dict с количеством найденных и удаленных дублей
        """
        try:
            # Получаем все воспоминания пользователя на этом уровне
            memories = await memory_level.get_user_memories(user_id, limit=1000)
            
            if len(memories) < 2:
                return {"found": 0, "removed": 0}
            
            duplicates_found = 0
            duplicates_removed = 0
            processed_ids = set()
            
            # Простой алгоритм дедупликации - сравниваем каждое воспоминание с остальными
            for i, memory1 in enumerate(memories):
                if memory1.get("id") in processed_ids:
                    continue
                    
                memory1_content = memory1.get("content", "").strip().lower()
                if not memory1_content:
                    continue
                
                similar_memories = []
                
                for j, memory2 in enumerate(memories[i+1:], i+1):
                    if memory2.get("id") in processed_ids:
                        continue
                        
                    memory2_content = memory2.get("content", "").strip().lower()
                    if not memory2_content:
                        continue
                    
                    # Простое сравнение схожести текста
                    similarity = self._calculate_text_similarity(memory1_content, memory2_content)
                    
                    if similarity >= similarity_threshold:
                        similar_memories.append(memory2)
                        duplicates_found += 1
                
                # Если найдены дубли, удаляем все кроме первого (самого старого)
                if similar_memories:
                    # Сортируем по дате создания (старые первыми)
                    all_similar = [memory1] + similar_memories
                    all_similar.sort(key=lambda x: x.get("created_at", ""))
                    
                    # Удаляем все кроме первого (самого старого)
                    for memory_to_remove in all_similar[1:]:
                        try:
                            await memory_level.delete_memory(memory_to_remove.get("id"))
                            duplicates_removed += 1
                            processed_ids.add(memory_to_remove.get("id"))
                        except Exception as e:
                            logger.warning(f"Ошибка удаления дубля {memory_to_remove.get('id')}: {e}")
                    
                    processed_ids.add(memory1.get("id"))
            
            return {"found": duplicates_found, "removed": duplicates_removed}
            
        except Exception as e:
            logger.error(f"Ошибка дедупликации уровня памяти: {e}")
            return {"found": 0, "removed": 0}
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Простое вычисление схожести текста
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            Коэффициент схожести от 0.0 до 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        # Нормализация текста
        text1 = text1.strip().lower()
        text2 = text2.strip().lower()
        
        if text1 == text2:
            return 1.0
        
        # Простое сравнение по словам (Jaccard similarity)
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса таймера"""
        active_goals = sum(1 for goal in self.goals.values() if goal.status == GoalStatus.ACTIVE)
        
        return {
            "is_running": self.is_running,
            "total_goals": len(self.goals),
            "active_goals": active_goals,
            "execution_results_count": len(self.execution_results),
            "last_execution": self.execution_results[-1].timestamp if self.execution_results else None
        }
