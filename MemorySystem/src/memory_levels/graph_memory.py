"""
Graph Memory Manager для AIRI Memory System
Уровень 5: Графовая память - связи между концепциями, TTL = 365 дней
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import json

try:
    from ..utils.metadata_serializer import MetadataSerializer
except ImportError:
    from utils.metadata_serializer import MetadataSerializer

logger = logging.getLogger(__name__)

from chromadb import PersistentClient
from chromadb.config import Settings
import os
try:
    from ..storage.graph_storage_sqlite import GraphSQLiteStorage
except Exception:  # pragma: no cover
    from storage.graph_storage_sqlite import GraphSQLiteStorage

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Узел графа памяти"""
    id: str
    name: str
    node_type: str = "concept"
    properties: Dict[str, Any] = None
    importance: float = 0.5
    last_updated: datetime = None

@dataclass
class GraphEdge:
    """Ребро графа памяти"""
    id: str
    source_id: str
    target_id: str
    relationship_type: str = "related"
    strength: float = 0.5
    properties: Dict[str, Any] = None
    last_updated: datetime = None

class GraphMemoryManager:
    """Менеджер графовой памяти"""
    
    def __init__(self, chromadb_path: str = "./data/chroma_db"):
        data_root = os.getenv("AIRI_DATA_DIR", "./data")
        sqlite_db = os.getenv("SQLITE_DB", os.path.join(data_root, "memory_system.db"))
        self.chromadb_path = chromadb_path
        self.client = PersistentClient(
            path=chromadb_path,
            settings=Settings(anonymized_telemetry=False)
        )
        # Используем SentenceTransformerEmbeddingFunction для правильных distances
        try:
            from chromadb.utils import embedding_functions
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer, falling back to DefaultEmbeddingFunction: {e}")
            from chromadb.utils import embedding_functions
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        self.nodes_collection = self.client.get_or_create_collection(
            name="graph_nodes",
            metadata={"description": "Graph nodes for concept relationships"},
            embedding_function=self.embedding_function
        )
        self.edges_collection = self.client.get_or_create_collection(
            name="graph_edges",
            metadata={"description": "Graph edges for concept relationships"},
            embedding_function=self.embedding_function
        )
        # SQLite storage for edges (hybrid according to plan)
        self._sqlite = GraphSQLiteStorage(db_path=sqlite_db)
        self.max_nodes = 5000   # Максимум узлов в графе
        self.max_edges = 10000  # Максимум рёбер в графе
        self.ttl_days = 365     # Время жизни элемента (365 дней)
        
    async def add_node(self, name: str, user_id: str, 
                      node_type: str = "concept", properties: Dict[str, Any] = None,
                      importance: float = 0.5) -> str:
        """
        Добавить узел в граф памяти
        
        Args:
            name: Название узла
            user_id: ID пользователя
            node_type: Тип узла (concept, person, place, event, etc.)
            properties: Свойства узла
            importance: Важность узла (0.0-1.0)
            
        Returns:
            ID созданного узла
        """
        try:
            node_id = f"gn_{uuid.uuid4()}"
            timestamp = datetime.now()
            
            # Создаем метаданные
            metadata = {
                "user_id": user_id,
                "name": name,
                "node_type": node_type,
                "properties": MetadataSerializer.safe_serialize_dict(properties) if properties else "{}",
                "importance": importance,
                "timestamp": timestamp.isoformat(),
                "last_updated": timestamp.isoformat(),
                "memory_type": "graph_node"
            }
            
            # Добавляем в ChromaDB
            self.nodes_collection.add(
                documents=[name],
                metadatas=[metadata],
                ids=[node_id]
            )
            
            # Очищаем старые элементы
            await self._cleanup_old_memories(user_id)
            
            logger.info(f"Added graph node: {node_id}")
            return node_id
            
        except Exception as e:
            logger.error(f"Error adding graph node: {e}")
            raise
    
    async def add_edge(self, source_id: str, target_id: str, user_id: str,
                      relationship_type: str = "related", strength: float = 0.5,
                      properties: Dict[str, Any] = None) -> str:
        """
        Добавить ребро в граф памяти
        
        Args:
            source_id: ID исходного узла
            target_id: ID целевого узла
            user_id: ID пользователя
            relationship_type: Тип связи (related, causes, part_of, etc.)
            strength: Сила связи (0.0-1.0)
            properties: Свойства связи
            
        Returns:
            ID созданного ребра
        """
        try:
            edge_id = f"ge_{uuid.uuid4()}"
            timestamp = datetime.now()
            
            # Создаем метаданные
            metadata = {
                "user_id": user_id,
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
                "strength": strength,
                "properties": MetadataSerializer.safe_serialize_dict(properties) if properties else "{}",
                "timestamp": timestamp.isoformat(),
                "last_updated": timestamp.isoformat(),
                "memory_type": "graph_edge"
            }
            
            # Создаем документ для поиска
            edge_doc = f"{relationship_type} connection between {source_id} and {target_id}"
            
            # Добавляем в ChromaDB
            self.edges_collection.add(
                documents=[edge_doc],
                metadatas=[metadata],
                ids=[edge_id]
            )
            # Duplicate edge in SQLite for fast path/relations
            try:
                self._sqlite.add_edge(source_id, target_id, relationship_type, strength, user_id)
            except Exception:
                pass
            
            logger.info(f"Added graph edge: {edge_id}")
            return edge_id
            
        except Exception as e:
            logger.error(f"Error adding graph edge: {e}")
            raise
    
    async def get_node(self, node_id: str, user_id: str) -> Optional[GraphNode]:
        """
        Получить узел по ID
        
        Args:
            node_id: ID узла
            user_id: ID пользователя
            
        Returns:
            Узел или None
        """
        try:
            results = self.nodes_collection.get(
                ids=[node_id], 
                where={"user_id": user_id},
                include=["metadatas"]
            )
            if not results["metadatas"]:
                return None
            
            metadata = results["metadatas"][0]
            return GraphNode(
                id=node_id,
                name=metadata["name"],
                node_type=metadata.get("node_type", "concept"),
                properties=MetadataSerializer.safe_deserialize_dict(metadata.get("properties", "{}")),
                importance=metadata.get("importance", 0.5),
                last_updated=datetime.fromisoformat(metadata.get("last_updated", metadata["timestamp"]))
            )
            
        except Exception as e:
            logger.error(f"Error getting node: {e}")
            return None
    
    async def get_connected_nodes(self, node_id: str, user_id: str, 
                                relationship_type: Optional[str] = None,
                                limit: int = 20) -> List[Tuple[GraphNode, GraphEdge]]:
        """
        Получить связанные узлы
        
        Args:
            node_id: ID исходного узла
            user_id: ID пользователя
            relationship_type: Тип связи для фильтрации
            limit: Максимальное количество результатов
            
        Returns:
            Список пар (узел, связь)
        """
        try:
            # Ищем рёбра, где узел является источником или целью
            where_filter = {
                "$and": [
                    {"user_id": user_id},
                    {"$or": [
                        {"source_id": node_id},
                        {"target_id": node_id}
                    ]}
                ]
            }
            
            if relationship_type:
                where_filter["relationship_type"] = relationship_type
            
            results = self.edges_collection.get(
                where=where_filter,
                limit=limit,
                include=["metadatas"]
            )
            
            connected_nodes = []
            for i, metadata in enumerate(results["metadatas"]):
                # Определяем связанный узел
                connected_node_id = metadata["target_id"] if metadata["source_id"] == node_id else metadata["source_id"]
                
                # Получаем узел
                node = await self.get_node(connected_node_id)
                if node:
                    # Создаем ребро
                    edge = GraphEdge(
                        id=results["ids"][i],
                        source_id=metadata["source_id"],
                        target_id=metadata["target_id"],
                        relationship_type=metadata.get("relationship_type", "related"),
                        strength=metadata.get("strength", 0.5),
                        properties=MetadataSerializer.safe_deserialize_dict(metadata.get("properties", "{}")),
                        last_updated=datetime.fromisoformat(metadata.get("last_updated", metadata["timestamp"]))
                    )
                    connected_nodes.append((node, edge))
            
            return connected_nodes
            
        except Exception as e:
            logger.error(f"Error getting connected nodes: {e}")
            return []
    
    async def find_path(self, source_id: str, target_id: str, user_id: str,
                       max_depth: int = 3) -> List[GraphNode]:
        """
        Найти путь между узлами
        
        Args:
            source_id: ID исходного узла
            target_id: ID целевого узла
            user_id: ID пользователя
            max_depth: Максимальная глубина поиска
            
        Returns:
            Список узлов на пути
        """
        try:
            # Используем SQLite BFS по рёбрам
            path_ids = self._sqlite.find_path_bfs(user_id, source_id, target_id, max_depth=max_depth)
            if not path_ids:
                return []
            nodes: List[GraphNode] = []
            for nid in path_ids:
                node = await self.get_node(nid)
                if node:
                    nodes.append(node)
            return nodes
            
        except Exception as e:
            logger.error(f"Error finding path: {e}")
            return []
    
    async def search_nodes(self, user_id: str, query: str, 
                          node_type: Optional[str] = None, limit: int = 10) -> List[GraphNode]:
        """
        Поиск узлов в графе
        
        Args:
            user_id: ID пользователя
            query: Поисковый запрос
            node_type: Тип узла для фильтрации
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных узлов
        """
        try:
            # Строим фильтр
            where_filter = {"user_id": user_id}
            if node_type:
                where_filter["node_type"] = node_type
            
            results = self.nodes_collection.query(
                query_texts=[query],
                where=where_filter,
                n_results=limit,
                include=["metadatas", "distances"]
            )
            
            nodes = []
            for i, metadata in enumerate(results["metadatas"][0]):
                distance = results["distances"][0][i]
                
                # Фильтруем по релевантности
                if distance < 0.8:
                    node = GraphNode(
                        id=results["ids"][0][i],
                        name=metadata["name"],
                        node_type=metadata.get("node_type", "concept"),
                        properties=MetadataSerializer.safe_deserialize_dict(metadata.get("properties", "{}")),
                        importance=metadata.get("importance", 0.5),
                        last_updated=datetime.fromisoformat(metadata.get("last_updated", metadata["timestamp"]))
                    )
                    nodes.append(node)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error searching nodes: {e}")
            return []
    
    async def update_node_properties(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """
        Обновить свойства узла
        
        Args:
            node_id: ID узла
            properties: Новые свойства
            
        Returns:
            True если успешно
        """
        try:
            # Получаем текущие метаданные
            results = self.nodes_collection.get(ids=[node_id], include=["metadatas"])
            if not results["metadatas"]:
                return False
            
            metadata = results["metadatas"][0]
            metadata["properties"] = MetadataSerializer.safe_serialize_dict(properties)
            metadata["last_updated"] = datetime.now().isoformat()
            
            # Обновляем метаданные
            self.nodes_collection.update(
                ids=[node_id],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated properties for node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating node properties: {e}")
            return False
    
    async def update_edge_strength(self, edge_id: str, strength: float) -> bool:
        """
        Обновить силу связи
        
        Args:
            edge_id: ID ребра
            strength: Новая сила связи (0.0-1.0)
            
        Returns:
            True если успешно
        """
        try:
            # Получаем текущие метаданные
            results = self.edges_collection.get(ids=[edge_id], include=["metadatas"])
            if not results["metadatas"]:
                return False
            
            metadata = results["metadatas"][0]
            metadata["strength"] = strength
            metadata["last_updated"] = datetime.now().isoformat()
            
            # Обновляем метаданные
            self.edges_collection.update(
                ids=[edge_id],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated strength for edge {edge_id}: {strength}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating edge strength: {e}")
            return False
    
    async def remove_node(self, node_id: str) -> bool:
        """
        Удалить узел из графа
        
        Args:
            node_id: ID узла
            
        Returns:
            True если успешно
        """
        try:
            # Удаляем узел
            self.nodes_collection.delete(ids=[node_id])
            
            # Удаляем все связанные рёбра
            edges_results = self.edges_collection.get(
                where={"$or": [{"source_id": node_id}, {"target_id": node_id}]},
                include=["metadatas"]
            )
            
            if edges_results["ids"]:
                self.edges_collection.delete(ids=edges_results["ids"])
            
            logger.info(f"Removed graph node: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing node: {e}")
            return False
    
    async def remove_edge(self, edge_id: str) -> bool:
        """
        Удалить ребро из графа
        
        Args:
            edge_id: ID ребра
            
        Returns:
            True если успешно
        """
        try:
            self.edges_collection.delete(ids=[edge_id])
            logger.info(f"Removed graph edge: {edge_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing edge: {e}")
            return False
    
    async def _cleanup_old_memories(self, user_id: str):
        """Очистка старых элементов графовой памяти"""
        try:
            current_time = datetime.now()
            
            # Очистка старых узлов
            nodes_results = self.nodes_collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            to_remove_nodes = []
            for i, metadata in enumerate(nodes_results["metadatas"]):
                ts_str = metadata.get("last_updated") or metadata.get("timestamp")
                if not ts_str:
                    # Если нет временных меток, пропускаем элемент
                    continue
                timestamp = datetime.fromisoformat(ts_str)
                age_days = (current_time - timestamp).total_seconds() / (24 * 3600)
                importance = metadata.get("importance", 0.5)
                
                # Удаляем старые узлы с низкой важностью
                if age_days > self.ttl_days and importance < 0.3:
                    to_remove_nodes.append(nodes_results["ids"][i])
            
            if to_remove_nodes:
                self.nodes_collection.delete(ids=to_remove_nodes)
                logger.info(f"Cleaned up {len(to_remove_nodes)} old graph nodes")
            
            # Очистка старых рёбер
            edges_results = self.edges_collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            to_remove_edges = []
            for i, metadata in enumerate(edges_results["metadatas"]):
                ts_str = metadata.get("last_updated") or metadata.get("timestamp")
                if not ts_str:
                    continue
                timestamp = datetime.fromisoformat(ts_str)
                age_days = (current_time - timestamp).total_seconds() / (24 * 3600)
                strength = metadata.get("strength", 0.5)
                
                # Удаляем старые рёбра с низкой силой
                if age_days > self.ttl_days and strength < 0.3:
                    to_remove_edges.append(edges_results["ids"][i])
            
            if to_remove_edges:
                self.edges_collection.delete(ids=to_remove_edges)
                logger.info(f"Cleaned up {len(to_remove_edges)} old graph edges")
            
            # Ограничиваем количество элементов
            await self._limit_collection_size(user_id)
                    
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")
    
    async def _limit_collection_size(self, user_id: str):
        """Ограничить размер коллекций"""
        try:
            # Ограничиваем количество узлов
            nodes_results = self.nodes_collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            if len(nodes_results["ids"]) > self.max_nodes:
                # Сортируем по важности, удаляем наименее важные
                nodes_with_importance = []
                for i, metadata in enumerate(nodes_results["metadatas"]):
                    importance = metadata.get("importance", 0.5)
                    nodes_with_importance.append((nodes_results["ids"][i], importance))
                
                nodes_with_importance.sort(key=lambda x: x[1])
                excess_count = len(nodes_with_importance) - self.max_nodes
                to_remove = [item[0] for item in nodes_with_importance[:excess_count]]
                
                if to_remove:
                    self.nodes_collection.delete(ids=to_remove)
                    logger.info(f"Removed {len(to_remove)} excess graph nodes")
            
            # Ограничиваем количество рёбер
            edges_results = self.edges_collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            if len(edges_results["ids"]) > self.max_edges:
                # Сортируем по силе, удаляем наименее сильные
                edges_with_strength = []
                for i, metadata in enumerate(edges_results["metadatas"]):
                    strength = metadata.get("strength", 0.5)
                    edges_with_strength.append((edges_results["ids"][i], strength))
                
                edges_with_strength.sort(key=lambda x: x[1])
                excess_count = len(edges_with_strength) - self.max_edges
                to_remove = [item[0] for item in edges_with_strength[:excess_count]]
                
                if to_remove:
                    self.edges_collection.delete(ids=to_remove)
                    logger.info(f"Removed {len(to_remove)} excess graph edges")
                    
        except Exception as e:
            logger.error(f"Error limiting collection size: {e}")
    
    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Получить статистику графовой памяти
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Словарь со статистикой
        """
        try:
            nodes_results = self.nodes_collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            edges_results = self.edges_collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            if not nodes_results["metadatas"]:
                return {
                    "total_nodes": 0,
                    "total_edges": 0,
                    "avg_node_importance": 0.0,
                    "avg_edge_strength": 0.0,
                    "node_types": {},
                    "relationship_types": {}
                }
            
            # Статистика узлов
            node_importances = [m.get("importance", 0.5) for m in nodes_results["metadatas"]]
            node_types = {}
            for metadata in nodes_results["metadatas"]:
                node_type = metadata.get("node_type", "concept")
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Статистика рёбер
            edge_strengths = [m.get("strength", 0.5) for m in edges_results["metadatas"]] if edges_results["metadatas"] else []
            relationship_types = {}
            for metadata in edges_results["metadatas"]:
                rel_type = metadata.get("relationship_type", "related")
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
            return {
                "total_nodes": len(nodes_results["metadatas"]),
                "total_edges": len(edges_results["metadatas"]),
                "avg_node_importance": sum(node_importances) / len(node_importances),
                "avg_edge_strength": sum(edge_strengths) / len(edge_strengths) if edge_strengths else 0.0,
                "node_types": node_types,
                "relationship_types": relationship_types
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    async def get_all_nodes(self, user_id: str) -> List[GraphNode]:
        """
        Получить все узлы пользователя
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Список всех узлов
        """
        try:
            results = self.nodes_collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            nodes = []
            for i, metadata in enumerate(results["metadatas"]):
                node = GraphNode(
                    id=results["ids"][i],
                    name=metadata["name"],
                    node_type=metadata.get("node_type", "concept"),
                    properties=MetadataSerializer.safe_deserialize_dict(metadata.get("properties", "{}")),
                    importance=metadata.get("importance", 0.5),
                    last_updated=datetime.fromisoformat(metadata.get("last_updated", metadata["timestamp"]))
                )
                nodes.append(node)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error getting all nodes: {e}")
            return []
    
    async def get_node_edges(self, node_id: str, user_id: str) -> List[GraphEdge]:
        """
        Получить все ребра для узла
        
        Args:
            node_id: ID узла
            user_id: ID пользователя
            
        Returns:
            Список ребер
        """
        try:
            # Ищем ребра где узел является источником
            source_results = self.edges_collection.get(
                where={"source_id": node_id, "user_id": user_id},
                include=["metadatas"]
            )
            
            # Ищем ребра где узел является целью
            target_results = self.edges_collection.get(
                where={"target_id": node_id, "user_id": user_id},
                include=["metadatas"]
            )
            
            # Объединяем результаты
            all_ids = source_results["ids"] + target_results["ids"]
            all_metadatas = source_results["metadatas"] + target_results["metadatas"]
            
            edges = []
            for i, metadata in enumerate(all_metadatas):
                edge = GraphEdge(
                    id=all_ids[i],
                    source_id=metadata["source_id"],
                    target_id=metadata["target_id"],
                    relationship_type=metadata.get("relationship_type", "related"),
                    strength=metadata.get("strength", 0.5),
                    properties=MetadataSerializer.safe_deserialize_dict(metadata.get("properties", "{}")),
                    last_updated=datetime.fromisoformat(metadata.get("last_updated", metadata["timestamp"]))
                )
                edges.append(edge)
            
            return edges
            
        except Exception as e:
            logger.error(f"Error getting node edges: {e}")
            return []
