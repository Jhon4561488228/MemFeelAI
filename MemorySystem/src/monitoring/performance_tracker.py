"""
Расширенный трекер производительности для AIRI Memory System
Отслеживает время ответа, точность, количество результатов и другие метрики
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import json

try:
    from .metrics import inc, set_gauge, record_event
    from .search_metrics import record_search_metrics
except ImportError:
    from metrics import inc, set_gauge, record_event
    from search_metrics import record_search_metrics

@dataclass
class PerformanceMetrics:
    """Метрики производительности для операции"""
    operation_type: str  # "search", "add_memory", "consolidation", "llm_call"
    operation_name: str  # "semantic_search", "graph_search", "integrated_search"
    user_id: str
    timestamp: float
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    result_count: int = 0
    accuracy_score: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit: bool = False
    metadata: Dict[str, Any] = None

@dataclass
class AccuracyMetrics:
    """Метрики точности для поиска"""
    query: str
    search_type: str
    user_id: str
    timestamp: float
    expected_results: int = 0
    actual_results: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    relevance_scores: List[float] = None
    avg_relevance: float = 0.0
    user_satisfaction: Optional[float] = None  # 0.0-1.0, если доступно

class PerformanceTracker:
    """Трекер производительности системы"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.performance_history: deque = deque(maxlen=max_history)
        self.accuracy_history: deque = deque(maxlen=max_history)
        
        # Статистика по типам операций
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_duration_ms": 0.0,
            "min_duration_ms": float('inf'),
            "max_duration_ms": 0.0,
            "p95_duration_ms": 0.0,
            "p99_duration_ms": 0.0,
            "error_rate": 0.0,
            "avg_result_count": 0.0,
            "avg_accuracy": 0.0,
            "recent_durations": deque(maxlen=100),
            "recent_accuracies": deque(maxlen=100)
        })
        
        # Инициализируем базовые метрики
        self._init_performance_metrics()
    
    def _init_performance_metrics(self):
        """Инициализация базовых метрик производительности"""
        # Счетчики операций
        operation_types = ["search", "add_memory", "consolidation", "llm_call", "api_request"]
        for op_type in operation_types:
            inc(f"performance_{op_type}_total", 0)
            inc(f"performance_{op_type}_success", 0)
            inc(f"performance_{op_type}_errors", 0)
            set_gauge(f"performance_{op_type}_avg_duration_ms", 0.0)
            set_gauge(f"performance_{op_type}_error_rate", 0.0)
            set_gauge(f"performance_{op_type}_avg_accuracy", 0.0)
        
        # Общие метрики производительности
        set_gauge("performance_system_avg_response_time_ms", 0.0)
        set_gauge("performance_system_throughput_ops_per_sec", 0.0)
        set_gauge("performance_system_error_rate", 0.0)
        set_gauge("performance_system_availability_percent", 100.0)
    
    def track_operation(self, operation_type: str, operation_name: str, 
                       user_id: str, duration_ms: float, success: bool = True,
                       error_message: str = None, result_count: int = 0,
                       accuracy_score: float = 0.0, cache_hit: bool = False,
                       metadata: Dict[str, Any] = None) -> PerformanceMetrics:
        """Отследить операцию"""
        try:
            metrics = PerformanceMetrics(
                operation_type=operation_type,
                operation_name=operation_name,
                user_id=user_id,
                timestamp=time.time(),
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
                result_count=result_count,
                accuracy_score=accuracy_score,
                cache_hit=cache_hit,
                metadata=metadata or {}
            )
            
            # Добавляем в историю
            self.performance_history.append(metrics)
            
            # Обновляем статистику
            self._update_operation_stats(metrics)
            
            # Обновляем базовые метрики
            self._update_base_metrics(metrics)
            
            # Записываем событие
            record_event("performance", {
                "operation_type": operation_type,
                "operation_name": operation_name,
                "user_id": user_id,
                "duration_ms": duration_ms,
                "success": success,
                "result_count": result_count,
                "accuracy_score": accuracy_score,
                "cache_hit": cache_hit
            })
            
            return metrics
            
        except Exception as e:
            print(f"Error tracking performance: {e}")
            return None
    
    def track_accuracy(self, query: str, search_type: str, user_id: str,
                      actual_results: int, relevance_scores: List[float] = None,
                      expected_results: int = 0, user_satisfaction: float = None) -> AccuracyMetrics:
        """Отследить точность поиска"""
        try:
            # Вычисляем метрики точности
            precision = 0.0
            recall = 0.0
            f1_score = 0.0
            
            if expected_results > 0:
                precision = actual_results / expected_results if expected_results > 0 else 0.0
                recall = actual_results / expected_results if expected_results > 0 else 0.0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
            
            metrics = AccuracyMetrics(
                query=query,
                search_type=search_type,
                user_id=user_id,
                timestamp=time.time(),
                expected_results=expected_results,
                actual_results=actual_results,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                relevance_scores=relevance_scores or [],
                avg_relevance=avg_relevance,
                user_satisfaction=user_satisfaction
            )
            
            # Добавляем в историю
            self.accuracy_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"Error tracking accuracy: {e}")
            return None
    
    def _update_operation_stats(self, metrics: PerformanceMetrics):
        """Обновить статистику операций"""
        stats = self.operation_stats[metrics.operation_name]
        
        # Обновляем счетчики
        stats["total_operations"] += 1
        if metrics.success:
            stats["successful_operations"] += 1
        else:
            stats["failed_operations"] += 1
        
        # Обновляем длительность
        stats["recent_durations"].append(metrics.duration_ms)
        if stats["recent_durations"]:
            durations = list(stats["recent_durations"])
            stats["avg_duration_ms"] = statistics.mean(durations)
            stats["min_duration_ms"] = min(durations)
            stats["max_duration_ms"] = max(durations)
            
            # Вычисляем перцентили
            sorted_durations = sorted(durations)
            n = len(sorted_durations)
            stats["p95_duration_ms"] = sorted_durations[int(n * 0.95)] if n > 0 else 0
            stats["p99_duration_ms"] = sorted_durations[int(n * 0.99)] if n > 0 else 0
        
        # Обновляем точность
        if metrics.accuracy_score > 0:
            stats["recent_accuracies"].append(metrics.accuracy_score)
            if stats["recent_accuracies"]:
                stats["avg_accuracy"] = statistics.mean(list(stats["recent_accuracies"]))
        
        # Обновляем количество результатов
        if metrics.result_count > 0:
            result_counts = [m.result_count for m in self.performance_history 
                           if m.operation_name == metrics.operation_name and m.result_count > 0]
            if result_counts:
                stats["avg_result_count"] = statistics.mean(result_counts)
        
        # Вычисляем частоту ошибок
        if stats["total_operations"] > 0:
            stats["error_rate"] = stats["failed_operations"] / stats["total_operations"]
    
    def _update_base_metrics(self, metrics: PerformanceMetrics):
        """Обновить базовые метрики"""
        # Счетчики по типам операций
        inc(f"performance_{metrics.operation_type}_total")
        if metrics.success:
            inc(f"performance_{metrics.operation_type}_success")
        else:
            inc(f"performance_{metrics.operation_type}_errors")
        
        # Обновляем gauges
        stats = self.operation_stats[metrics.operation_name]
        set_gauge(f"performance_{metrics.operation_type}_avg_duration_ms", stats["avg_duration_ms"])
        set_gauge(f"performance_{metrics.operation_type}_error_rate", stats["error_rate"])
        set_gauge(f"performance_{metrics.operation_type}_avg_accuracy", stats["avg_accuracy"])
        
        # Общие метрики системы
        all_durations = [m.duration_ms for m in self.performance_history]
        if all_durations:
            set_gauge("performance_system_avg_response_time_ms", statistics.mean(all_durations))
        
        # Пропускная способность (операций в секунду)
        recent_operations = [m for m in self.performance_history 
                           if m.timestamp > time.time() - 60]  # Последняя минута
        if recent_operations:
            set_gauge("performance_system_throughput_ops_per_sec", len(recent_operations))
        
        # Общая частота ошибок
        total_operations = len(self.performance_history)
        failed_operations = sum(1 for m in self.performance_history if not m.success)
        if total_operations > 0:
            set_gauge("performance_system_error_rate", failed_operations / total_operations)
            set_gauge("performance_system_availability_percent", 
                     (total_operations - failed_operations) / total_operations * 100)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Получить сводку производительности"""
        return {
            "operation_stats": {
                name: {
                    "total_operations": stats["total_operations"],
                    "successful_operations": stats["successful_operations"],
                    "failed_operations": stats["failed_operations"],
                    "avg_duration_ms": stats["avg_duration_ms"],
                    "min_duration_ms": stats["min_duration_ms"],
                    "max_duration_ms": stats["max_duration_ms"],
                    "p95_duration_ms": stats["p95_duration_ms"],
                    "p99_duration_ms": stats["p99_duration_ms"],
                    "error_rate": stats["error_rate"],
                    "avg_result_count": stats["avg_result_count"],
                    "avg_accuracy": stats["avg_accuracy"]
                }
                for name, stats in self.operation_stats.items()
            },
            "recent_operations": [
                {
                    "operation_type": m.operation_type,
                    "operation_name": m.operation_name,
                    "user_id": m.user_id,
                    "duration_ms": m.duration_ms,
                    "success": m.success,
                    "result_count": m.result_count,
                    "accuracy_score": m.accuracy_score,
                    "timestamp": datetime.fromtimestamp(m.timestamp).isoformat(),
                    "cache_hit": m.cache_hit,
                    "error_message": m.error_message
                }
                for m in list(self.performance_history)[-20:]  # Последние 20 операций
            ],
            "accuracy_summary": self._get_accuracy_summary(),
            "total_operations": len(self.performance_history),
            "operation_types": list(set(m.operation_type for m in self.performance_history))
        }
    
    def _get_accuracy_summary(self) -> Dict[str, Any]:
        """Получить сводку точности"""
        if not self.accuracy_history:
            return {"total_measurements": 0}
        
        search_types = defaultdict(list)
        for metrics in self.accuracy_history:
            search_types[metrics.search_type].append(metrics)
        
        summary = {
            "total_measurements": len(self.accuracy_history),
            "by_search_type": {}
        }
        
        for search_type, metrics_list in search_types.items():
            if metrics_list:
                avg_precision = statistics.mean([m.precision for m in metrics_list])
                avg_recall = statistics.mean([m.recall for m in metrics_list])
                avg_f1 = statistics.mean([m.f1_score for m in metrics_list])
                avg_relevance = statistics.mean([m.avg_relevance for m in metrics_list])
                
                summary["by_search_type"][search_type] = {
                    "measurements": len(metrics_list),
                    "avg_precision": avg_precision,
                    "avg_recall": avg_recall,
                    "avg_f1_score": avg_f1,
                    "avg_relevance": avg_relevance,
                    "avg_results_count": statistics.mean([m.actual_results for m in metrics_list])
                }
        
        return summary
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Получить алерты по производительности"""
        alerts = []
        
        for operation_name, stats in self.operation_stats.items():
            # Алерт по высокой длительности
            if stats["avg_duration_ms"] > 5000:  # 5 секунд
                alerts.append({
                    "type": "high_duration",
                    "operation": operation_name,
                    "value": stats["avg_duration_ms"],
                    "threshold": 5000,
                    "message": f"Average duration for {operation_name} is {stats['avg_duration_ms']:.1f}ms (threshold: 5000ms)"
                })
            
            # Алерт по высокой частоте ошибок
            if stats["error_rate"] > 0.1:  # 10%
                alerts.append({
                    "type": "high_error_rate",
                    "operation": operation_name,
                    "value": stats["error_rate"],
                    "threshold": 0.1,
                    "message": f"Error rate for {operation_name} is {stats['error_rate']:.1%} (threshold: 10%)"
                })
            
            # Алерт по низкой точности
            if stats["avg_accuracy"] > 0 and stats["avg_accuracy"] < 0.3:
                alerts.append({
                    "type": "low_accuracy",
                    "operation": operation_name,
                    "value": stats["avg_accuracy"],
                    "threshold": 0.3,
                    "message": f"Average accuracy for {operation_name} is {stats['avg_accuracy']:.2f} (threshold: 0.3)"
                })
        
        return alerts
    
    def get_throughput_metrics(self, time_window_minutes: int = 5) -> Dict[str, Any]:
        """Получить метрики пропускной способности"""
        cutoff_time = time.time() - (time_window_minutes * 60)
        recent_operations = [m for m in self.performance_history if m.timestamp > cutoff_time]
        
        if not recent_operations:
            return {"time_window_minutes": time_window_minutes, "operations": 0, "throughput_per_minute": 0}
        
        # Группируем по типам операций
        by_type = defaultdict(list)
        for op in recent_operations:
            by_type[op.operation_type].append(op)
        
        throughput_by_type = {}
        for op_type, ops in by_type.items():
            throughput_by_type[op_type] = {
                "operations": len(ops),
                "throughput_per_minute": len(ops) / time_window_minutes,
                "avg_duration_ms": statistics.mean([op.duration_ms for op in ops]) if ops else 0,
                "success_rate": sum(1 for op in ops if op.success) / len(ops) if ops else 0
            }
        
        return {
            "time_window_minutes": time_window_minutes,
            "total_operations": len(recent_operations),
            "total_throughput_per_minute": len(recent_operations) / time_window_minutes,
            "by_operation_type": throughput_by_type
        }

# Глобальный экземпляр трекера производительности
_performance_tracker: Optional[PerformanceTracker] = None

def get_performance_tracker() -> PerformanceTracker:
    """Получить глобальный экземпляр трекера производительности"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker

def track_operation_performance(operation_type: str, operation_name: str, 
                               user_id: str, duration_ms: float, success: bool = True,
                               error_message: str = None, result_count: int = 0,
                               accuracy_score: float = 0.0, cache_hit: bool = False,
                               metadata: Dict[str, Any] = None):
    """Удобная функция для отслеживания производительности операции"""
    try:
        tracker = get_performance_tracker()
        return tracker.track_operation(
            operation_type=operation_type,
            operation_name=operation_name,
            user_id=user_id,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            result_count=result_count,
            accuracy_score=accuracy_score,
            cache_hit=cache_hit,
            metadata=metadata
        )
    except Exception as e:
        print(f"Error tracking operation performance: {e}")

def track_search_accuracy(query: str, search_type: str, user_id: str,
                         actual_results: int, relevance_scores: List[float] = None,
                         expected_results: int = 0, user_satisfaction: float = None):
    """Удобная функция для отслеживания точности поиска"""
    try:
        tracker = get_performance_tracker()
        return tracker.track_accuracy(
            query=query,
            search_type=search_type,
            user_id=user_id,
            actual_results=actual_results,
            relevance_scores=relevance_scores,
            expected_results=expected_results,
            user_satisfaction=user_satisfaction
        )
    except Exception as e:
        print(f"Error tracking search accuracy: {e}")

# Декоратор для автоматического отслеживания производительности
def track_performance(operation_type: str, operation_name: str = None):
    """Декоратор для автоматического отслеживания производительности функций"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_message = None
            result_count = 0
            accuracy_score = 0.0
            
            try:
                result = await func(*args, **kwargs)
                
                # Пытаемся извлечь метрики из результата
                if isinstance(result, dict):
                    result_count = result.get('results_count', result.get('count', 0))
                    accuracy_score = result.get('avg_relevance', result.get('accuracy', 0.0))
                elif hasattr(result, '__len__'):
                    result_count = len(result)
                
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                raise
                
            finally:
                duration_ms = (time.time() - start_time) * 1000
                op_name = operation_name or func.__name__
                
                # Пытаемся извлечь user_id из аргументов
                user_id = "unknown"
                if args and hasattr(args[0], 'user_id'):
                    user_id = args[0].user_id
                elif 'user_id' in kwargs:
                    user_id = kwargs['user_id']
                
                track_operation_performance(
                    operation_type=operation_type,
                    operation_name=op_name,
                    user_id=user_id,
                    duration_ms=duration_ms,
                    success=success,
                    error_message=error_message,
                    result_count=result_count,
                    accuracy_score=accuracy_score
                )
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_message = None
            result_count = 0
            accuracy_score = 0.0
            
            try:
                result = func(*args, **kwargs)
                
                # Пытаемся извлечь метрики из результата
                if isinstance(result, dict):
                    result_count = result.get('results_count', result.get('count', 0))
                    accuracy_score = result.get('avg_relevance', result.get('accuracy', 0.0))
                elif hasattr(result, '__len__'):
                    result_count = len(result)
                
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                raise
                
            finally:
                duration_ms = (time.time() - start_time) * 1000
                op_name = operation_name or func.__name__
                
                # Пытаемся извлечь user_id из аргументов
                user_id = "unknown"
                if args and hasattr(args[0], 'user_id'):
                    user_id = args[0].user_id
                elif 'user_id' in kwargs:
                    user_id = kwargs['user_id']
                
                track_operation_performance(
                    operation_type=operation_type,
                    operation_name=op_name,
                    user_id=user_id,
                    duration_ms=duration_ms,
                    success=success,
                    error_message=error_message,
                    result_count=result_count,
                    accuracy_score=accuracy_score
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
