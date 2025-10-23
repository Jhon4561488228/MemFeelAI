"""
Специализированные метрики для поиска в AIRI Memory System
Отслеживает производительность, точность и качество различных типов поиска
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

try:
    from .metrics import inc, set_gauge, record_event
except ImportError:
    from metrics import inc, set_gauge, record_event

@dataclass
class SearchMetrics:
    """Метрики для одного поискового запроса"""
    query: str
    search_type: str  # "semantic", "graph", "hybrid", "integrated", "contextual"
    user_id: str
    timestamp: float
    duration_ms: float
    results_count: int
    relevance_scores: List[float]
    avg_relevance: float
    max_relevance: float
    min_relevance: float
    cache_hit: bool = False
    error: Optional[str] = None
    memory_levels_searched: List[str] = None
    filters_applied: Dict[str, Any] = None

@dataclass
class SearchPerformanceStats:
    """Статистика производительности поиска"""
    total_searches: int = 0
    avg_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    min_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0

@dataclass
class SearchQualityStats:
    """Статистика качества поиска"""
    avg_relevance: float = 0.0
    max_relevance: float = 0.0
    min_relevance: float = 0.0
    avg_results_count: float = 0.0
    zero_results_rate: float = 0.0
    high_relevance_rate: float = 0.0  # > 0.8

class SearchMetricsCollector:
    """Сборщик метрик поиска"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.search_history: deque = deque(maxlen=max_history)
        self.performance_stats: Dict[str, SearchPerformanceStats] = defaultdict(SearchPerformanceStats)
        self.quality_stats: Dict[str, SearchQualityStats] = defaultdict(SearchQualityStats)
        self.duration_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.relevance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Инициализируем базовые метрики
        self._init_base_metrics()
    
    def _init_base_metrics(self):
        """Инициализация базовых метрик"""
        # Счетчики поиска по типам
        search_types = ["semantic", "graph", "hybrid", "integrated", "contextual", "fts5"]
        for search_type in search_types:
            inc(f"search_{search_type}_total", 0)
            set_gauge(f"search_{search_type}_avg_duration_ms", 0.0)
            set_gauge(f"search_{search_type}_avg_relevance", 0.0)
            set_gauge(f"search_{search_type}_avg_results_count", 0.0)
            set_gauge(f"search_{search_type}_error_rate", 0.0)
            set_gauge(f"search_{search_type}_cache_hit_rate", 0.0)
        
        # Общие метрики поиска
        inc("search_total", 0)
        inc("search_errors_total", 0)
        inc("search_cache_hits_total", 0)
        set_gauge("search_avg_duration_ms", 0.0)
        set_gauge("search_avg_relevance", 0.0)
        set_gauge("search_avg_results_count", 0.0)
        set_gauge("search_error_rate", 0.0)
        set_gauge("search_cache_hit_rate", 0.0)
    
    def record_search(self, metrics: SearchMetrics):
        """Записать метрики поиска"""
        try:
            # Добавляем в историю
            self.search_history.append(metrics)
            
            # Обновляем статистику производительности
            self._update_performance_stats(metrics)
            
            # Обновляем статистику качества
            self._update_quality_stats(metrics)
            
            # Обновляем базовые метрики
            self._update_base_metrics(metrics)
            
            # Записываем событие
            record_event("search", {
                "query": metrics.query[:100],  # Ограничиваем длину
                "search_type": metrics.search_type,
                "user_id": metrics.user_id,
                "duration_ms": metrics.duration_ms,
                "results_count": metrics.results_count,
                "avg_relevance": metrics.avg_relevance,
                "cache_hit": metrics.cache_hit,
                "error": metrics.error
            })
            
        except Exception as e:
            # Не падаем на ошибках метрик
            print(f"Error recording search metrics: {e}")
    
    def _update_performance_stats(self, metrics: SearchMetrics):
        """Обновить статистику производительности"""
        search_type = metrics.search_type
        stats = self.performance_stats[search_type]
        
        # Обновляем счетчики
        stats.total_searches += 1
        
        # Обновляем историю длительности
        self.duration_history[search_type].append(metrics.duration_ms)
        
        # Пересчитываем статистику длительности
        durations = list(self.duration_history[search_type])
        if durations:
            stats.avg_duration_ms = sum(durations) / len(durations)
            stats.max_duration_ms = max(durations)
            stats.min_duration_ms = min(durations)
            
            # Вычисляем перцентили
            sorted_durations = sorted(durations)
            n = len(sorted_durations)
            stats.p95_duration_ms = sorted_durations[int(n * 0.95)] if n > 0 else 0
            stats.p99_duration_ms = sorted_durations[int(n * 0.99)] if n > 0 else 0
        
        # Обновляем статистику ошибок
        if metrics.error:
            self.error_history[search_type].append(metrics.error)
            stats.error_rate = len(self.error_history[search_type]) / stats.total_searches
        
        # Обновляем статистику кэша
        if metrics.cache_hit:
            stats.cache_hit_rate = sum(1 for m in self.search_history if m.search_type == search_type and m.cache_hit) / stats.total_searches
    
    def _update_quality_stats(self, metrics: SearchMetrics):
        """Обновить статистику качества"""
        search_type = metrics.search_type
        stats = self.quality_stats[search_type]
        
        # Обновляем историю релевантности
        if metrics.relevance_scores:
            self.relevance_history[search_type].extend(metrics.relevance_scores)
        
        # Пересчитываем статистику релевантности
        relevances = list(self.relevance_history[search_type])
        if relevances:
            stats.avg_relevance = sum(relevances) / len(relevances)
            stats.max_relevance = max(relevances)
            stats.min_relevance = min(relevances)
            stats.high_relevance_rate = sum(1 for r in relevances if r > 0.8) / len(relevances)
        
        # Статистика количества результатов
        result_counts = [m.results_count for m in self.search_history if m.search_type == search_type]
        if result_counts:
            stats.avg_results_count = sum(result_counts) / len(result_counts)
            stats.zero_results_rate = sum(1 for c in result_counts if c == 0) / len(result_counts)
    
    def _update_base_metrics(self, metrics: SearchMetrics):
        """Обновить базовые метрики"""
        # Счетчики по типам
        inc(f"search_{metrics.search_type}_total")
        
        # Общие счетчики
        inc("search_total")
        if metrics.error:
            inc("search_errors_total")
        if metrics.cache_hit:
            inc("search_cache_hits_total")
        
        # Обновляем gauges
        set_gauge(f"search_{metrics.search_type}_avg_duration_ms", 
                 self.performance_stats[metrics.search_type].avg_duration_ms)
        set_gauge(f"search_{metrics.search_type}_avg_relevance", 
                 self.quality_stats[metrics.search_type].avg_relevance)
        set_gauge(f"search_{metrics.search_type}_avg_results_count", 
                 self.quality_stats[metrics.search_type].avg_results_count)
        set_gauge(f"search_{metrics.search_type}_error_rate", 
                 self.performance_stats[metrics.search_type].error_rate)
        set_gauge(f"search_{metrics.search_type}_cache_hit_rate", 
                 self.performance_stats[metrics.search_type].cache_hit_rate)
        
        # Общие gauges
        all_durations = []
        all_relevances = []
        all_result_counts = []
        total_searches = 0
        total_errors = 0
        total_cache_hits = 0
        
        for metrics_item in self.search_history:
            all_durations.append(metrics_item.duration_ms)
            all_relevances.extend(metrics_item.relevance_scores)
            all_result_counts.append(metrics_item.results_count)
            total_searches += 1
            if metrics_item.error:
                total_errors += 1
            if metrics_item.cache_hit:
                total_cache_hits += 1
        
        if all_durations:
            set_gauge("search_avg_duration_ms", sum(all_durations) / len(all_durations))
        if all_relevances:
            set_gauge("search_avg_relevance", sum(all_relevances) / len(all_relevances))
        if all_result_counts:
            set_gauge("search_avg_results_count", sum(all_result_counts) / len(all_result_counts))
        if total_searches > 0:
            set_gauge("search_error_rate", total_errors / total_searches)
            set_gauge("search_cache_hit_rate", total_cache_hits / total_searches)
    
    def get_search_metrics_summary(self) -> Dict[str, Any]:
        """Получить сводку метрик поиска"""
        return {
            "performance_stats": {
                search_type: asdict(stats) 
                for search_type, stats in self.performance_stats.items()
            },
            "quality_stats": {
                search_type: asdict(stats) 
                for search_type, stats in self.quality_stats.items()
            },
            "recent_searches": [
                {
                    "query": m.query[:50] + "..." if len(m.query) > 50 else m.query,
                    "search_type": m.search_type,
                    "user_id": m.user_id,
                    "duration_ms": m.duration_ms,
                    "results_count": m.results_count,
                    "avg_relevance": m.avg_relevance,
                    "timestamp": datetime.fromtimestamp(m.timestamp).isoformat(),
                    "cache_hit": m.cache_hit,
                    "error": m.error
                }
                for m in list(self.search_history)[-20:]  # Последние 20 поисков
            ],
            "total_searches": len(self.search_history),
            "search_types": list(self.performance_stats.keys())
        }
    
    def get_search_type_metrics(self, search_type: str) -> Dict[str, Any]:
        """Получить метрики для конкретного типа поиска"""
        # Валидные типы поиска
        valid_search_types = ["semantic", "graph", "hybrid", "integrated", "contextual", "fts5"]
        
        if search_type not in valid_search_types:
            return {"error": f"Invalid search type '{search_type}'. Valid types: {valid_search_types}"}
        
        # Если нет данных для этого типа, возвращаем пустые метрики
        if search_type not in self.performance_stats:
            return {
                "search_type": search_type,
                "performance": {
                    "total_searches": 0,
                    "avg_duration_ms": 0.0,
                    "max_duration_ms": 0.0,
                    "min_duration_ms": 0.0,
                    "error_rate": 0.0,
                    "cache_hit_rate": 0.0
                },
                "quality": {
                    "avg_relevance": 0.0,
                    "max_relevance": 0.0,
                    "min_relevance": 0.0,
                    "zero_results_rate": 0.0,
                    "high_relevance_rate": 0.0
                },
                "recent_searches": [],
                "status": "no_data"
            }
        
        performance = self.performance_stats[search_type]
        quality = self.quality_stats[search_type]
        
        # Получаем последние поиски этого типа
        recent_searches = [
            {
                "query": m.query[:50] + "..." if len(m.query) > 50 else m.query,
                "user_id": m.user_id,
                "duration_ms": m.duration_ms,
                "results_count": m.results_count,
                "avg_relevance": m.avg_relevance,
                "timestamp": datetime.fromtimestamp(m.timestamp).isoformat(),
                "cache_hit": m.cache_hit,
                "error": m.error
            }
            for m in self.search_history
            if m.search_type == search_type
        ][-10:]  # Последние 10 поисков
        
        return {
            "search_type": search_type,
            "performance": asdict(performance),
            "quality": asdict(quality),
            "recent_searches": recent_searches,
            "total_searches": performance.total_searches
        }
    
    def get_top_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получить топ запросов по частоте"""
        query_counts = defaultdict(int)
        query_metrics = defaultdict(list)
        
        for metrics in self.search_history:
            query_counts[metrics.query] += 1
            query_metrics[metrics.query].append(metrics)
        
        # Сортируем по частоте
        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        
        top_queries = []
        for query, count in sorted_queries[:limit]:
            metrics_list = query_metrics[query]
            avg_duration = sum(m.duration_ms for m in metrics_list) / len(metrics_list)
            avg_relevance = sum(m.avg_relevance for m in metrics_list) / len(metrics_list)
            avg_results = sum(m.results_count for m in metrics_list) / len(metrics_list)
            
            top_queries.append({
                "query": query,
                "count": count,
                "avg_duration_ms": avg_duration,
                "avg_relevance": avg_relevance,
                "avg_results_count": avg_results,
                "search_types": list(set(m.search_type for m in metrics_list))
            })
        
        return top_queries
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Получить алерты по производительности"""
        alerts = []
        
        for search_type, stats in self.performance_stats.items():
            # Алерт по высокой длительности
            if stats.avg_duration_ms > 5000:  # 5 секунд
                alerts.append({
                    "type": "high_duration",
                    "search_type": search_type,
                    "value": stats.avg_duration_ms,
                    "threshold": 5000,
                    "message": f"Average search duration for {search_type} is {stats.avg_duration_ms:.1f}ms (threshold: 5000ms)"
                })
            
            # Алерт по высокой частоте ошибок
            if stats.error_rate > 0.1:  # 10%
                alerts.append({
                    "type": "high_error_rate",
                    "search_type": search_type,
                    "value": stats.error_rate,
                    "threshold": 0.1,
                    "message": f"Error rate for {search_type} is {stats.error_rate:.1%} (threshold: 10%)"
                })
        
        return alerts
    
    def get_quality_alerts(self) -> List[Dict[str, Any]]:
        """Получить алерты по качеству"""
        alerts = []
        
        for search_type, stats in self.quality_stats.items():
            # Алерт по низкой релевантности
            if stats.avg_relevance < 0.3:
                alerts.append({
                    "type": "low_relevance",
                    "search_type": search_type,
                    "value": stats.avg_relevance,
                    "threshold": 0.3,
                    "message": f"Average relevance for {search_type} is {stats.avg_relevance:.2f} (threshold: 0.3)"
                })
            
            # Алерт по высокому проценту пустых результатов
            if stats.zero_results_rate > 0.5:  # 50%
                alerts.append({
                    "type": "high_zero_results",
                    "search_type": search_type,
                    "value": stats.zero_results_rate,
                    "threshold": 0.5,
                    "message": f"Zero results rate for {search_type} is {stats.zero_results_rate:.1%} (threshold: 50%)"
                })
        
        return alerts

# Глобальный экземпляр сборщика метрик
_search_metrics_collector: Optional[SearchMetricsCollector] = None

def get_search_metrics_collector() -> SearchMetricsCollector:
    """Получить глобальный экземпляр сборщика метрик поиска"""
    global _search_metrics_collector
    if _search_metrics_collector is None:
        _search_metrics_collector = SearchMetricsCollector()
    return _search_metrics_collector

def record_search_metrics(query: str, search_type: str, user_id: str, 
                         duration_ms: float, results: List[Any], 
                         cache_hit: bool = False, error: str = None,
                         memory_levels: List[str] = None,
                         filters: Dict[str, Any] = None):
    """Удобная функция для записи метрик поиска"""
    try:
        # Вычисляем метрики релевантности
        relevance_scores = []
        if results:
            for result in results:
                if isinstance(result, dict):
                    relevance = result.get('relevance_score', result.get('score', 0.0))
                else:
                    relevance = getattr(result, 'relevance_score', getattr(result, 'score', 0.0))
                relevance_scores.append(float(relevance))
        
        # Создаем объект метрик
        metrics = SearchMetrics(
            query=query,
            search_type=search_type,
            user_id=user_id,
            timestamp=time.time(),
            duration_ms=duration_ms,
            results_count=len(results),
            relevance_scores=relevance_scores,
            avg_relevance=sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
            max_relevance=max(relevance_scores) if relevance_scores else 0.0,
            min_relevance=min(relevance_scores) if relevance_scores else 0.0,
            cache_hit=cache_hit,
            error=error,
            memory_levels_searched=memory_levels,
            filters_applied=filters
        )
        
        # Записываем метрики
        collector = get_search_metrics_collector()
        collector.record_search(metrics)
        
    except Exception as e:
        # Не падаем на ошибках метрик
        print(f"Error recording search metrics: {e}")
