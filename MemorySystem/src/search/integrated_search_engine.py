"""
Integrated Search Engine для AIRI Memory System
Объединяет семантический и графовый поиск для максимальной релевантности
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from collections import defaultdict

try:
    from ..monitoring.search_metrics import record_search_metrics
    from ..monitoring.performance_tracker import track_operation_performance, track_search_accuracy
except ImportError:
    from monitoring.search_metrics import record_search_metrics
    from monitoring.performance_tracker import track_operation_performance, track_search_accuracy

from .graph_search import GraphSearchEngine, GraphSearchQuery, GraphSearchResult
from ..memory_levels.semantic_memory import SemanticMemoryManager, SemanticMemoryItem
from ..memory_levels.graph_memory_sqlite import GraphMemoryManagerSQLite

@dataclass
class IntegratedSearchResult:
    """Результат интегрированного поиска"""
    id: str
    content: str
    source_type: str  # "semantic", "graph", "hybrid"
    relevance_score: float
    semantic_score: float = 0.0
    graph_score: float = 0.0
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    node_type: Optional[str] = None
    path_length: int = 0
    path_nodes: List[str] = None
    relationship_strength: float = 0.0
    knowledge_type: Optional[str] = None
    category: Optional[str] = None
    confidence: float = 0.0
    importance: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class IntegratedSearchQuery:
    """Запрос для интегрированного поиска"""
    query: str
    user_id: str
    search_types: List[str] = None  # ["semantic", "graph", "hybrid"]
    limit: int = 10
    semantic_weight: float = 0.6  # Вес семантического поиска
    graph_weight: float = 0.4     # Вес графового поиска
    min_confidence: float = 0.25
    min_importance: float = 0.2
    expand_graph: bool = True     # Расширять ли графовый поиск
    max_graph_depth: int = 2
    use_hybrid_ranking: bool = True
    emotion_filter: Optional[Dict[str, Any]] = None
    time_filter: Optional[Dict[str, Any]] = None

class IntegratedSearchEngine:
    """Движок интегрированного поиска"""
    
    def __init__(self, 
                 semantic_manager: Optional[SemanticMemoryManager] = None,
                 graph_manager: Optional[GraphMemoryManagerSQLite] = None,
                 graph_engine: Optional['GraphSearchEngine'] = None):
        """
        Инициализация интегрированного поиска
        
        Args:
            semantic_manager: Менеджер семантической памяти
            graph_manager: Менеджер графовой памяти
            graph_engine: Готовый экземпляр GraphSearchEngine
        """
        self.semantic_manager = semantic_manager
        self.graph_manager = graph_manager
        self.graph_engine = graph_engine  # Используем переданный экземпляр
        self._lock = asyncio.Lock()
        self._search_cache = {}
        self._cache_ttl = 300  # 5 минут
        logger.info("IntegratedSearchEngine initialized")
    
    def _check_search_cache(self, query: str, user_id: str) -> bool:
        """Проверка кэша поиска"""
        try:
            import time
            cache_key = f"{user_id}:{query}"
            current_time = time.time()
            
            if cache_key in self._search_cache:
                cached_time = self._search_cache[cache_key]
                if current_time - cached_time < self._cache_ttl:
                    return True
                else:
                    # Удаляем устаревший кэш
                    del self._search_cache[cache_key]
            
            # Сохраняем время запроса в кэш
            self._search_cache[cache_key] = current_time
            return False
            
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
            return False
    
    async def search(self, query: IntegratedSearchQuery) -> List[IntegratedSearchResult]:
        """
        Выполнить интегрированный поиск
        
        Args:
            query: Параметры поиска
            
        Returns:
            Список результатов интегрированного поиска
        """
        try:
            start_time = time.time()
            
            async with self._lock:
                # Определяем типы поиска
                search_types = query.search_types or ["semantic", "graph", "hybrid"]
                
                # Выполняем поиски параллельно
                tasks = []
                
                # Hybrid поиск автоматически включает semantic и graph
                if ("semantic" in search_types or "hybrid" in search_types) and self.semantic_manager:
                    # Для hybrid поиска увеличиваем лимит до 20
                    if "hybrid" in search_types:
                        # Создаем копию запроса с увеличенным лимитом
                        hybrid_query = IntegratedSearchQuery(
                            query=query.query,
                            user_id=query.user_id,
                            search_types=query.search_types,
                            limit=20,
                            min_confidence=query.min_confidence,
                            min_importance=query.min_importance,
                            max_graph_depth=query.max_graph_depth,
                            emotion_filter=query.emotion_filter,
                            time_filter=query.time_filter
                        )
                        tasks.append(self._semantic_search(hybrid_query))
                    else:
                        tasks.append(self._semantic_search(query))
                else:
                    tasks.append(asyncio.create_task(self._empty_result()))
                
                if ("graph" in search_types or "hybrid" in search_types) and self.graph_manager:
                    # Для hybrid поиска увеличиваем лимит до 20
                    if "hybrid" in search_types:
                        # Создаем копию запроса с увеличенным лимитом
                        hybrid_query = IntegratedSearchQuery(
                            query=query.query,
                            user_id=query.user_id,
                            search_types=query.search_types,
                            limit=20,
                            min_confidence=query.min_confidence,
                            min_importance=query.min_importance,
                            max_graph_depth=query.max_graph_depth,
                            emotion_filter=query.emotion_filter,
                            time_filter=query.time_filter
                        )
                        tasks.append(self._graph_search(hybrid_query))
                    else:
                        tasks.append(self._graph_search(query))
                else:
                    tasks.append(asyncio.create_task(self._empty_result()))
                
                # Ждем результаты
                semantic_results, graph_results = await asyncio.gather(*tasks)
                
                logger.info(f"IntegratedSearchEngine.search: semantic={len(semantic_results)}, graph={len(graph_results)}")
                
                # Объединяем результаты
                if "hybrid" in search_types:
                    logger.info(f"Calling _create_hybrid_results with semantic={len(semantic_results)}, graph={len(graph_results)}")
                    integrated_results = await self._create_hybrid_results(
                        semantic_results, graph_results, query
                    )
                    logger.info(f"_create_hybrid_results returned {len(integrated_results)} results")
                else:
                    logger.info(f"Calling _merge_results with semantic={len(semantic_results)}, graph={len(graph_results)}")
                    integrated_results = await self._merge_results(
                        semantic_results, graph_results, query
                    )
                    logger.info(f"_merge_results returned {len(integrated_results)} results")
                
                # Сортируем по релевантности
                integrated_results.sort(key=lambda x: x.relevance_score, reverse=True)
                
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            logger.info(f"Integrated search completed in {end_time - start_time:.4f}s, "
                       f"found {len(integrated_results)} results")
            
            # Отслеживаем производительность и метрики поиска
            try:
                # Вычисляем среднюю релевантность для точности
                relevance_scores = [r.relevance_score for r in integrated_results if r.relevance_score > 0]
                avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
                
                # Отслеживаем производительность операции
                track_operation_performance(
                    operation_type="search",
                    operation_name="integrated_search",
                    user_id=query.user_id,
                    duration_ms=duration_ms,
                    success=True,
                    result_count=len(integrated_results),
                    accuracy_score=avg_relevance,
                    cache_hit=False,
                    metadata={
                        "search_types": query.search_types,
                        "semantic_weight": query.semantic_weight,
                        "graph_weight": query.graph_weight,
                        "min_confidence": query.min_confidence,
                        "min_importance": query.min_importance,
                        "expand_graph": query.expand_graph,
                        "max_graph_depth": query.max_graph_depth
                    }
                )
                
                # Отслеживаем точность поиска
                track_search_accuracy(
                    query=query.query,
                    search_type="integrated",
                    user_id=query.user_id,
                    actual_results=len(integrated_results),
                    relevance_scores=relevance_scores
                )
                
                # Записываем метрики поиска
                record_search_metrics(
                    query=query.query,
                    search_type="integrated",
                    user_id=query.user_id,
                    duration_ms=duration_ms,
                    results=integrated_results,
                    cache_hit=self._check_search_cache(query.query, query.user_id),
                    memory_levels=search_types,
                    filters={
                        "semantic_weight": query.semantic_weight,
                        "graph_weight": query.graph_weight,
                        "min_confidence": query.min_confidence,
                        "min_importance": query.min_importance,
                        "expand_graph": query.expand_graph,
                        "max_graph_depth": query.max_graph_depth
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to record performance metrics: {e}")
            
            return integrated_results[:query.limit]
                
        except Exception as e:
            logger.error(f"Integrated search failed: {e}")
            return []
    
    async def _semantic_search(self, query: IntegratedSearchQuery) -> List[SemanticMemoryItem]:
        """Выполнить семантический поиск"""
        try:
            if not self.semantic_manager:
                logger.warning("Semantic manager not available")
                return []
            
            logger.debug(f"IntegratedSearchEngine semantic search: query='{query.query}', user_id='{query.user_id}', semantic_manager={self.semantic_manager}")
            
            results = await self.semantic_manager.search_knowledge(
                user_id=query.user_id,
                query=query.query,
                min_confidence=query.min_confidence,
                limit=query.limit * 2  # Берем больше для фильтрации
            )
            
            logger.debug(f"Semantic search raw results: {len(results)} items")
            
            # Фильтруем по важности
            filtered_results = [
                item for item in results 
                if item.importance >= query.min_importance
            ]
            
            logger.info(f"Semantic search found {len(filtered_results)} results")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _graph_search(self, query: IntegratedSearchQuery) -> List[GraphSearchResult]:
        """Выполнить графовый поиск"""
        try:
            if not self.graph_manager:
                logger.warning("Graph manager not available")
                return []
            
            # Используем переданный graph_engine или создаем новый
            if self.graph_engine is None:
                from ..search import get_graph_engine_lazy
                get_graph_engine = get_graph_engine_lazy()
                self.graph_engine = await get_graph_engine(self.graph_manager)
            
            # Создаем графовый запрос
            graph_query = GraphSearchQuery(
                query=query.query,
                user_id=query.user_id,
                search_type="simple",
                max_depth=query.max_graph_depth,
                limit=query.limit * 2
            )
            
            logger.debug(f"IntegratedSearchEngine graph search: query='{query.query}', user_id='{query.user_id}', graph_manager={self.graph_manager}")
            results = await self.graph_engine.search(graph_query)
            
            # Если включено расширение, добавляем связанные узлы
            if query.expand_graph:
                expanded_results = await self._expand_graph_results(results, query)
                results.extend(expanded_results)
            
            logger.info(f"Graph search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    async def _expand_graph_results(self, 
                                  initial_results: List[GraphSearchResult], 
                                  query: IntegratedSearchQuery) -> List[GraphSearchResult]:
        """Расширить графовые результаты связанными узлами"""
        try:
            expanded_results = []
            visited_nodes = {result.node_id for result in initial_results}
            
            for result in initial_results[:3]:  # Расширяем только топ-3
                # Получаем связанные узлы
                connected_nodes = await self.graph_engine._get_connected_nodes(
                    result.node_id, query.user_id
                )
                
                for connected_node, strength in connected_nodes:
                    if (connected_node.id not in visited_nodes and 
                        strength >= 0.3):  # Только сильные связи
                        
                        expanded_result = GraphSearchResult(
                            node_id=connected_node.id,
                            node_name=connected_node.name,
                            node_type=connected_node.node_type,
                            relevance_score=result.relevance_score * strength * 0.7,  # Снижаем релевантность
                            path_length=result.path_length + 1,
                            path_nodes=result.path_nodes + [connected_node.id],
                            relationship_strength=strength,
                            search_type="expanded"
                        )
                        expanded_results.append(expanded_result)
                        visited_nodes.add(connected_node.id)
            
            return expanded_results
            
        except Exception as e:
            logger.error(f"Graph expansion failed: {e}")
            return []
    
    async def _create_hybrid_results(self, 
                                   semantic_results: List[SemanticMemoryItem],
                                   graph_results: List[GraphSearchResult],
                                   query: IntegratedSearchQuery) -> List[IntegratedSearchResult]:
        """Создать гибридные результаты, объединяющие семантический и графовый поиск"""
        try:
            logger.info(f"_create_hybrid_results: semantic={len(semantic_results)}, graph={len(graph_results)}, query.limit={query.limit}")
            hybrid_results = []
            
            # Создаем индекс графовых результатов по имени узла
            graph_index = {}
            for result in graph_results:
                logger.info(f"Graph node: '{result.node_name}' -> '{result.node_name.lower()}'")
                graph_index[result.node_name.lower()] = result
            
            logger.info(f"Graph index keys: {list(graph_index.keys())}")
            
            # Обрабатываем семантические результаты
            for i, semantic_item in enumerate(semantic_results):
                logger.info(f"Processing semantic item {i+1}: {semantic_item.content[:50]}...")
                
                # Ищем связанные графовые узлы
                related_graph_results = []
                
                # Улучшенный поиск связей
                content_lower = semantic_item.content.lower()
                logger.info(f"Content: {content_lower[:100]}...")
                
                # 1. Поиск по словам (убираем пунктуацию)
                import re
                content_words = set(re.findall(r'\b\w+\b', content_lower))
                logger.info(f"Content words (cleaned): {list(content_words)[:10]}")
                
                for word in content_words:
                    if word in graph_index:
                        logger.info(f"Found connection by word: '{word}' -> {graph_index[word].node_name}")
                        related_graph_results.append(graph_index[word])
                
                # 2. Поиск по полному контенту (подстрока)
                for node_name in graph_index.keys():
                    if node_name in content_lower:
                        logger.info(f"Found connection by substring: '{node_name}' -> {graph_index[node_name].node_name}")
                        if graph_index[node_name] not in related_graph_results:
                            related_graph_results.append(graph_index[node_name])
                
                # 3. Поиск по частичным совпадениям
                for node_name in graph_index.keys():
                    if len(node_name) > 3:  # Только для длинных имен
                        for word in content_words:
                            if len(word) > 3 and (node_name in word or word in node_name):
                                logger.info(f"Found connection by partial match: '{word}' <-> '{node_name}'")
                                if graph_index[node_name] not in related_graph_results:
                                    related_graph_results.append(graph_index[node_name])
                
                # Поиск по связанным концепциям
                for concept in semantic_item.related_concepts or []:
                    concept_lower = concept.lower()
                    if concept_lower in graph_index:
                        related_graph_results.append(graph_index[concept_lower])
                
                # Создаем гибридный результат
                if related_graph_results:
                    # Берем лучший графовый результат
                    best_graph = max(related_graph_results, key=lambda x: x.relevance_score)
                    
                    # Вычисляем гибридный скор
                    hybrid_score = (
                        semantic_item.confidence * query.semantic_weight +
                        best_graph.relevance_score * query.graph_weight
                    )
                    
                    hybrid_result = IntegratedSearchResult(
                        id=semantic_item.id,
                        content=semantic_item.content,
                        source_type="hybrid",
                        relevance_score=hybrid_score,
                        semantic_score=semantic_item.confidence,
                        graph_score=best_graph.relevance_score,
                        node_id=best_graph.node_id,
                        node_name=best_graph.node_name,
                        node_type=best_graph.node_type,
                        path_length=best_graph.path_length,
                        path_nodes=best_graph.path_nodes,
                        relationship_strength=best_graph.relationship_strength,
                        knowledge_type=semantic_item.knowledge_type,
                        category=semantic_item.category,
                        confidence=semantic_item.confidence,
                        importance=semantic_item.importance,
                        metadata={
                            "semantic_id": semantic_item.id,
                            "graph_id": best_graph.node_id,
                            "hybrid_confidence": hybrid_score
                        }
                    )
                    hybrid_results.append(hybrid_result)
                else:
                    # Только семантический результат
                    semantic_result = IntegratedSearchResult(
                        id=semantic_item.id,
                        content=semantic_item.content,
                        source_type="semantic",
                        relevance_score=semantic_item.confidence,
                        semantic_score=semantic_item.confidence,
                        knowledge_type=semantic_item.knowledge_type,
                        category=semantic_item.category,
                        confidence=semantic_item.confidence,
                        importance=semantic_item.importance,
                        metadata={"semantic_id": semantic_item.id}
                    )
                    hybrid_results.append(semantic_result)
            
            # Добавляем графовые результаты, которые не были связаны
            used_graph_nodes = {result.node_id for result in hybrid_results if result.node_id}
            for graph_result in graph_results:
                if graph_result.node_id not in used_graph_nodes:
                    graph_only_result = IntegratedSearchResult(
                        id=f"graph_{graph_result.node_id}",
                        content=graph_result.node_name,
                        source_type="graph",
                        relevance_score=graph_result.relevance_score,
                        graph_score=graph_result.relevance_score,
                        node_id=graph_result.node_id,
                        node_name=graph_result.node_name,
                        node_type=graph_result.node_type,
                        path_length=graph_result.path_length,
                        path_nodes=graph_result.path_nodes,
                        relationship_strength=graph_result.relationship_strength,
                        metadata={"graph_id": graph_result.node_id}
                    )
                    hybrid_results.append(graph_only_result)
            
            # Если hybrid_results пустой, создаем сбалансированные результаты из исходных данных
            if len(hybrid_results) == 0:
                logger.info("No hybrid results found, creating balanced results from source data")
                
                # Создаем семантические результаты
                semantic_balanced = []
                for i, semantic_item in enumerate(semantic_results[:query.limit // 2]):
                    semantic_result = IntegratedSearchResult(
                        id=semantic_item.id,
                        content=semantic_item.content,
                        source_type="semantic",
                        relevance_score=semantic_item.confidence * 1.1,  # Увеличиваем вес семантики
                        semantic_score=semantic_item.confidence,
                        knowledge_type=semantic_item.knowledge_type,
                        category=semantic_item.category,
                        confidence=semantic_item.confidence,
                        importance=semantic_item.importance,
                        metadata={"semantic_id": semantic_item.id}
                    )
                    semantic_balanced.append(semantic_result)
                
                # Создаем графовые результаты
                graph_balanced = []
                for i, graph_item in enumerate(graph_results[:query.limit - len(semantic_balanced)]):
                    graph_result = IntegratedSearchResult(
                        id=f"graph_{graph_item.node_id}",
                        content=graph_item.node_name,
                        source_type="graph",
                        relevance_score=graph_item.relevance_score,
                        graph_score=graph_item.relevance_score,
                        node_id=graph_item.node_id,
                        node_name=graph_item.node_name,
                        node_type=graph_item.node_type,
                        path_length=graph_item.path_length,
                        path_nodes=graph_item.path_nodes,
                        relationship_strength=graph_item.relationship_strength,
                        metadata={"graph_id": graph_item.node_id}
                    )
                    graph_balanced.append(graph_result)
                
                balanced_results = semantic_balanced + graph_balanced
                logger.info(f"Created balanced results: {len(balanced_results)} (semantic: {len(semantic_balanced)}, graph: {len(graph_balanced)})")
                return balanced_results
            
            # Сбалансированное распределение результатов для hybrid поиска
            if len(hybrid_results) > query.limit:
                # Разделяем на семантические и графовые результаты
                semantic_results = [r for r in hybrid_results if r.source_type in ["semantic", "hybrid"]]
                graph_results = [r for r in hybrid_results if r.source_type == "graph"]
                
                # Целевое распределение: 50% семантических, 50% графовых
                target_semantic = min(len(semantic_results), query.limit // 2)
                target_graph = min(len(graph_results), query.limit - target_semantic)
                
                # Сортируем по relevance_score
                semantic_results.sort(key=lambda x: x.relevance_score, reverse=True)
                graph_results.sort(key=lambda x: x.relevance_score, reverse=True)
                
                # Берем лучшие результаты
                balanced_results = semantic_results[:target_semantic] + graph_results[:target_graph]
                
                logger.info(f"Balanced hybrid results: {len(balanced_results)} (semantic: {target_semantic}, graph: {target_graph})")
                return balanced_results
            
            logger.info(f"Created {len(hybrid_results)} hybrid results")
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Hybrid results creation failed: {e}")
            return []
    
    async def _merge_results(self, 
                           semantic_results: List[SemanticMemoryItem],
                           graph_results: List[GraphSearchResult],
                           query: IntegratedSearchQuery) -> List[IntegratedSearchResult]:
        """Объединить результаты без гибридизации"""
        try:
            logger.info(f"_merge_results: semantic={len(semantic_results)}, graph={len(graph_results)}")
            merged_results = []
            
            # Конвертируем семантические результаты
            logger.info(f"Converting {len(semantic_results)} semantic results")
            for i, semantic_item in enumerate(semantic_results):
                result = IntegratedSearchResult(
                    id=semantic_item.id,
                    content=semantic_item.content,
                    source_type="semantic",
                    relevance_score=semantic_item.confidence * 1.1,  # Увеличиваем вес семантики
                    semantic_score=semantic_item.confidence,
                    knowledge_type=semantic_item.knowledge_type,
                    category=semantic_item.category,
                    confidence=semantic_item.confidence,
                    importance=semantic_item.importance,
                    metadata={"semantic_id": semantic_item.id}
                )
                merged_results.append(result)
                logger.info(f"Added semantic result {i+1}: {semantic_item.content[:50]}... (score: {semantic_item.confidence})")
            
            # Конвертируем графовые результаты
            for graph_result in graph_results:
                result = IntegratedSearchResult(
                    id=f"graph_{graph_result.node_id}",
                    content=graph_result.node_name,
                    source_type="graph",
                    relevance_score=graph_result.relevance_score,
                    graph_score=graph_result.relevance_score,
                    node_id=graph_result.node_id,
                    node_name=graph_result.node_name,
                    node_type=graph_result.node_type,
                    path_length=graph_result.path_length,
                    path_nodes=graph_result.path_nodes,
                    relationship_strength=graph_result.relationship_strength,
                    metadata={"graph_id": graph_result.node_id}
                )
                merged_results.append(result)
            
            return merged_results
            
        except Exception as e:
            logger.error(f"Results merging failed: {e}")
            return []
    
    async def _empty_result(self):
        """Возвращает пустой результат для отсутствующих поисков"""
        return []
    
    async def get_search_suggestions(self, query: str, user_id: str, limit: int = 5) -> List[str]:
        """
        Получить предложения для поиска на основе графа
        
        Args:
            query: Частичный запрос
            user_id: ID пользователя
            limit: Максимальное количество предложений
            
        Returns:
            Список предложений
        """
        try:
            if not self.graph_engine:
                return []
            
            # Ищем узлы, начинающиеся с запроса
            suggestions = await self.graph_engine.expand_query(query, user_id, limit)
            
            # Добавляем семантические предложения
            if self.semantic_manager:
                semantic_results = await self.semantic_manager.search_knowledge(
                    user_id=user_id,
                    query=query,
                    limit=limit
                )
                
                for item in semantic_results:
                    if item.content not in suggestions:
                        suggestions.append(item.content)
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Search suggestions failed: {e}")
            return []

# Глобальный экземпляр интегрированного поиска
_integrated_engine: Optional[IntegratedSearchEngine] = None

async def get_integrated_engine(semantic_manager: Optional[SemanticMemoryManager] = None,
                              graph_manager: Optional[GraphMemoryManagerSQLite] = None) -> IntegratedSearchEngine:
    """Получить глобальный экземпляр интегрированного поиска"""
    global _integrated_engine
    if _integrated_engine is None or semantic_manager is not None or graph_manager is not None:
        # Используем тот же подход, что и прямой API - get_graph_engine_lazy()
        graph_engine = None
        if graph_manager is not None:
            from ..search import get_graph_engine_lazy
            get_graph_engine = get_graph_engine_lazy()
            graph_engine = await get_graph_engine(graph_manager)
        
        _integrated_engine = IntegratedSearchEngine(semantic_manager, graph_manager, graph_engine)
    return _integrated_engine
