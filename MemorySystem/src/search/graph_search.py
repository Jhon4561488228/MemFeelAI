"""
Graph Search Engine для AIRI Memory System
Реализует поиск по графу знаний с различными алгоритмами
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from loguru import logger
from collections import deque, defaultdict

try:
    from ..monitoring.search_metrics import record_search_metrics
except ImportError:
    from monitoring.search_metrics import record_search_metrics

from ..memory_levels.graph_memory_sqlite import GraphMemoryManagerSQLite as GraphMemoryManager, GraphNode, GraphEdge

@dataclass
class GraphSearchResult:
    """Результат графового поиска"""
    node_id: str
    node_name: str
    node_type: str
    relevance_score: float
    path_length: int = 0
    path_nodes: List[str] = None
    relationship_strength: float = 0.0
    search_type: str = "graph"

@dataclass
class GraphSearchQuery:
    """Запрос для графового поиска"""
    query: str
    user_id: str
    search_type: str = "simple"  # "simple", "pathfinding", "traversal", "relationship"
    max_depth: int = 3
    min_strength: float = 0.1
    node_types: Optional[List[str]] = None
    relationship_types: Optional[List[str]] = None
    limit: int = 10
    # Параметры для pathfinding
    start_node: Optional[str] = None
    end_node: Optional[str] = None
    
    def __post_init__(self):
        """Инициализация после создания объекта"""
        # НЕ перезаписываем node_types, если он None
        # Это позволяет корректно передавать фильтры

class GraphSearchEngine:
    """Движок графового поиска"""
    
    def __init__(self, graph_manager: Optional[GraphMemoryManager] = None):
        """
        Инициализация графового поиска
        
        Args:
            graph_manager: Менеджер графовой памяти
        """
        self.graph_manager = graph_manager
        self._lock = asyncio.Lock()
        self._search_cache = {}
        self._cache_ttl = 300  # 5 минут
        logger.info("GraphSearchEngine initialized")
    
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
    
    async def search(self, query: GraphSearchQuery) -> List[GraphSearchResult]:
        """
        Выполнить графовый поиск
        
        Args:
            query: Параметры поиска
            
        Returns:
            Список результатов поиска
        """
        if not self.graph_manager:
            logger.warning("GraphMemoryManager not available")
            return []
        
        try:
            start_time = time.time()
            
            async with self._lock:
                if query.search_type == "simple":
                    results = await self._simple_graph_search(query)
                elif query.search_type == "pathfinding":
                    results = await self._pathfinding_search(query)
                elif query.search_type == "traversal":
                    results = await self._traversal_search(query)
                elif query.search_type == "relationship":
                    results = await self._relationship_search(query)
                else:
                    logger.warning(f"Unknown search type: {query.search_type}")
                    results = []
            
            # Сортируем по релевантности
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            logger.info(f"Graph search completed in {end_time - start_time:.4f}s, "
                       f"found {len(results)} results")
            
            # Записываем метрики поиска
            try:
                record_search_metrics(
                    query=query.query,
                    search_type="graph",
                    user_id=query.user_id,
                    duration_ms=duration_ms,
                    results=results,
                    cache_hit=self._check_search_cache(query.query, query.user_id),
                    memory_levels=["graph"],
                    filters={
                        "search_type": query.search_type,
                        "max_depth": query.max_depth,
                        "min_strength": query.min_strength,
                        "node_types": query.node_types,
                        "relationship_types": query.relationship_types
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to record search metrics: {e}")
            
            return results[:query.limit]
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    async def _simple_graph_search(self, query: GraphSearchQuery) -> List[GraphSearchResult]:
        """
        Простой поиск по узлам графа
        
        Args:
            query: Параметры поиска
            
        Returns:
            Список результатов
        """
        try:
            node_type_str = query.node_types[0] if query.node_types and len(query.node_types) > 0 else "all"
            logger.info(f"Simple graph search: user_id={query.user_id}, query='{query.query}', node_type={node_type_str}, node_types={query.node_types}")
            
            # Ищем узлы по тексту
            node_type_filter = query.node_types[0] if query.node_types and len(query.node_types) > 0 else None
            nodes = await self.graph_manager.search_nodes(
                user_id=query.user_id,
                query=query.query,
                node_type=node_type_filter,
                limit=query.limit * 2  # Берем больше для фильтрации
            )
            
            logger.info(f"GraphMemoryManager.search_nodes returned {len(nodes)} nodes")
            
            results = []
            for node in nodes:
                # Вычисляем релевантность на основе важности и типа
                relevance_score = node.importance * 0.7 + 0.3
                
                result = GraphSearchResult(
                    node_id=node.id,
                    node_name=node.name,
                    node_type=node.node_type,
                    relevance_score=relevance_score,
                    path_length=0,
                    path_nodes=[node.id],
                    relationship_strength=1.0,
                    search_type="simple"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Simple graph search failed: {e}")
            return []
    
    async def _pathfinding_search(self, query: GraphSearchQuery) -> List[GraphSearchResult]:
        """
        Поиск пути между узлами (BFS pathfinding)
        
        Args:
            query: Параметры поиска
            
        Returns:
            Список результатов с найденным путем
        """
        try:
            logger.info(f"Pathfinding search: query='{query.query}', user_id='{query.user_id}'")
            
            # Если нет start_node и end_node, используем простой поиск
            if not hasattr(query, 'start_node') or not hasattr(query, 'end_node') or \
               not query.start_node or not query.end_node:
                logger.info("No start/end nodes specified, using simple search")
                results = await self._simple_graph_search(query)
                for result in results:
                    result.search_type = "pathfinding"
                    result.path_length = 1
                return results
            
            # BFS pathfinding между конкретными узлами
            path = await self._bfs_pathfinding(
                start_node_id=query.start_node,
                end_node_id=query.end_node,
                user_id=query.user_id,
                max_depth=query.max_depth
            )
            
            if not path:
                logger.info(f"No path found between {query.start_node} and {query.end_node}")
                return []
            
            # Преобразуем путь в результаты
            results = []
            for i, node_id in enumerate(path):
                try:
                    # Получаем информацию о узле
                    node = await self.graph_manager.get_node(node_id, query.user_id)
                    if node:
                        result = GraphSearchResult(
                            node_id=node.id,
                            node_name=node.name,
                            node_type=node.node_type,
                            relevance_score=1.0 - (i * 0.1),  # Убывающая релевантность по пути
                            path_length=len(path),
                            path_nodes=path,
                            relationship_strength=1.0,
                            search_type="pathfinding"
                        )
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to get node {node_id}: {e}")
                    continue
            
            logger.info(f"Pathfinding found path with {len(results)} nodes")
            return results
            
        except Exception as e:
            logger.error(f"Pathfinding search failed: {e}")
            return []
    
    async def _bfs_pathfinding(self, start_node_id: str, end_node_id: str, 
                              user_id: str, max_depth: int = 3) -> List[str]:
        """
        BFS алгоритм для поиска пути между узлами
        
        Args:
            start_node_id: ID начального узла
            end_node_id: ID конечного узла
            user_id: ID пользователя
            max_depth: Максимальная глубина поиска
            
        Returns:
            Список ID узлов, образующих путь, или пустой список если путь не найден
        """
        try:
            if start_node_id == end_node_id:
                return [start_node_id]
            
            # Очередь для BFS: (node_id, path_to_node)
            queue = [(start_node_id, [start_node_id])]
            visited = {start_node_id}
            
            while queue:
                current_node_id, path = queue.pop(0)
                
                # Проверяем глубину
                if len(path) > max_depth:
                    continue
                
                # Получаем соседние узлы
                try:
                    edges = await self.graph_manager.get_node_edges(current_node_id, user_id)
                    
                    for edge in edges:
                        # Определяем следующий узел
                        next_node_id = edge.target_id if edge.source_id == current_node_id else edge.source_id
                        
                        if next_node_id in visited:
                            continue
                        
                        visited.add(next_node_id)
                        new_path = path + [next_node_id]
                        
                        # Проверяем, достигли ли цели
                        if next_node_id == end_node_id:
                            logger.info(f"Path found: {' -> '.join(new_path)}")
                            return new_path
                        
                        # Добавляем в очередь для дальнейшего поиска
                        queue.append((next_node_id, new_path))
                        
                except Exception as e:
                    logger.warning(f"Failed to get edges for node {current_node_id}: {e}")
                    continue
            
            logger.info(f"No path found between {start_node_id} and {end_node_id}")
            return []
            
        except Exception as e:
            logger.error(f"BFS pathfinding failed: {e}")
            return []
    
    async def _traversal_search(self, query: GraphSearchQuery) -> List[GraphSearchResult]:
        """
        Поиск с обходом графа (BFS)
        
        Args:
            query: Параметры поиска
            
        Returns:
            Список результатов
        """
        try:
            # Сначала находим начальные узлы
            start_nodes = await self.graph_manager.search_nodes(
                user_id=query.user_id,
                query=query.query,
                limit=5
            )
            
            if not start_nodes:
                return []
            
            results = []
            visited = set()
            
            # BFS для каждого начального узла
            for start_node in start_nodes:
                queue = deque([(start_node, 0, [start_node.id])])
                
                while queue:
                    current_node, depth, path = queue.popleft()
                    
                    if depth > query.max_depth or current_node.id in visited:
                        continue
                    
                    visited.add(current_node.id)
                    
                    # Добавляем текущий узел в результаты
                    relevance_score = current_node.importance * (1.0 - depth * 0.2)
                    
                    result = GraphSearchResult(
                        node_id=current_node.id,
                        node_name=current_node.name,
                        node_type=current_node.node_type,
                        relevance_score=relevance_score,
                        path_length=depth,
                        path_nodes=path.copy(),
                        relationship_strength=1.0,
                        search_type="traversal"
                    )
                    results.append(result)
                    
                    # Получаем связанные узлы
                    if depth < query.max_depth:
                        connected_nodes = await self._get_connected_nodes(
                            current_node.id, query.user_id
                        )
                        
                        for connected_node, edge_strength in connected_nodes:
                            if connected_node.id not in visited and edge_strength >= query.min_strength:
                                new_path = path + [connected_node.id]
                                queue.append((connected_node, depth + 1, new_path))
            
            return results
            
        except Exception as e:
            logger.error(f"Traversal search failed: {e}")
            return []
    
    async def _relationship_search(self, query: GraphSearchQuery) -> List[GraphSearchResult]:
        """
        Поиск по силе связей
        
        Args:
            query: Параметры поиска
            
        Returns:
            Список результатов
        """
        try:
            # Находим узлы с сильными связями
            all_nodes = await self.graph_manager.get_all_nodes(query.user_id)
            
            results = []
            for node in all_nodes:
                # Получаем связи узла
                connections = await self._get_connected_nodes(node.id, query.user_id)
                
                # Вычисляем общую силу связей
                total_strength = sum(strength for _, strength in connections)
                avg_strength = total_strength / len(connections) if connections else 0
                
                # Фильтруем по минимальной силе
                if avg_strength >= query.min_strength:
                    relevance_score = node.importance * 0.5 + avg_strength * 0.5
                    
                    result = GraphSearchResult(
                        node_id=node.id,
                        node_name=node.name,
                        node_type=node.node_type,
                        relevance_score=relevance_score,
                        path_length=0,
                        path_nodes=[node.id],
                        relationship_strength=avg_strength,
                        search_type="relationship"
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Relationship search failed: {e}")
            return []
    
    async def _get_connected_nodes(self, node_id: str, user_id: str) -> List[Tuple[GraphNode, float]]:
        """
        Получить узлы, связанные с данным узлом
        
        Args:
            node_id: ID узла
            user_id: ID пользователя
            
        Returns:
            Список связанных узлов с силой связи
        """
        try:
            # Получаем все ребра для узла
            edges = await self.graph_manager.get_node_edges(node_id, user_id)
            
            connected_nodes = []
            for edge in edges:
                # Определяем целевой узел
                target_id = edge.target_id if edge.source_id == node_id else edge.source_id
                
                # Получаем узел
                target_node = await self.graph_manager.get_node(target_id, user_id)
                if target_node:
                    connected_nodes.append((target_node, edge.strength))
            
            return connected_nodes
            
        except Exception as e:
            logger.error(f"Error getting connected nodes: {e}")
            return []
    
    async def expand_query(self, query: str, user_id: str, max_expansions: int = 3) -> List[str]:
        """
        Расширить запрос на основе графа
        
        Args:
            query: Исходный запрос
            user_id: ID пользователя
            max_expansions: Максимальное количество расширений
            
        Returns:
            Список расширенных запросов
        """
        try:
            # Находим узлы, связанные с запросом
            nodes = await self.graph_manager.search_nodes(
                user_id=user_id,
                query=query,
                limit=max_expansions
            )
            
            expansions = [query]  # Начинаем с исходного запроса
            
            for node in nodes:
                # Получаем связанные узлы
                connected = await self._get_connected_nodes(node.id, user_id)
                
                for connected_node, strength in connected:
                    if strength > 0.3:  # Только сильные связи
                        expansions.append(connected_node.name)
                        
                        if len(expansions) >= max_expansions + 1:
                            break
                
                if len(expansions) >= max_expansions + 1:
                    break
            
            return expansions[:max_expansions + 1]
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]

# Глобальный экземпляр графового поиска
_graph_engine: Optional[GraphSearchEngine] = None

async def get_graph_engine(graph_manager: Optional[GraphMemoryManager] = None) -> GraphSearchEngine:
    """Получить глобальный экземпляр графового поиска"""
    global _graph_engine
    if _graph_engine is None:
        _graph_engine = GraphSearchEngine(graph_manager)
    elif graph_manager is not None:
        # Всегда обновляем graph_manager на новый
        _graph_engine.graph_manager = graph_manager
    return _graph_engine
