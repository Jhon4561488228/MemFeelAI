"""
SQLite Graph Memory Manager для AIRI Memory System
Уровень 5: Графовая память - связи между концепциями, TTL = 365 дней
ИСПРАВЛЕННАЯ ВЕРСИЯ: использует SQLite + FTS5 вместо ChromaDB
"""

import asyncio
import uuid
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import os
from pathlib import Path

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

class GraphMemoryManagerSQLite:
    """SQLite-версия менеджера графовой памяти"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Инициализация SQLite графового менеджера
        
        Args:
            db_path: Путь к SQLite базе данных
        """
        data_root = os.getenv("AIRI_DATA_DIR", "./data")
        self.db_path = db_path or os.getenv("SQLITE_DB", os.path.join(data_root, "memory_system.db"))
        
        # Создаем директорию если не существует
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем базу данных
        self._init_database()
        
        logger.info(f"GraphMemoryManagerSQLite initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Инициализация таблиц базы данных"""
        with sqlite3.connect(self.db_path) as conn:
            # Включаем FTS5
            conn.execute("PRAGMA compile_options")
            
            # Создаем таблицу узлов
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    node_type TEXT DEFAULT 'concept',
                    properties TEXT DEFAULT '{}',
                    properties_text TEXT DEFAULT '',
                    importance REAL DEFAULT 0.5,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Добавляем колонку properties_text если её нет (миграция)
            try:
                conn.execute("ALTER TABLE graph_nodes ADD COLUMN properties_text TEXT DEFAULT ''")
                logger.info("Added properties_text column to graph_nodes table")
                
                # Обновляем существующие записи, извлекая текст из properties
                cursor = conn.execute("SELECT id, properties FROM graph_nodes WHERE properties_text = '' OR properties_text IS NULL")
                for row in cursor.fetchall():
                    node_id, properties_json = row
                    try:
                        properties = json.loads(properties_json) if properties_json else {}
                        properties_text = self._extract_properties_text(properties)
                        conn.execute("UPDATE graph_nodes SET properties_text = ? WHERE id = ?", (properties_text, node_id))
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"Failed to extract properties text for node {node_id}: {e}")
                        continue
                
                conn.commit()
                logger.info("Updated existing nodes with extracted properties text")
                
            except sqlite3.OperationalError:
                # Колонка уже существует
                pass
            
            # ОПТИМИЗАЦИЯ: Проверяем, существует ли FTS5 таблица
            # Пересоздаем только если таблица не существует или имеет неправильную структуру
            cur = conn.cursor()
            cur.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='graph_nodes_fts'
            """)
            fts_table_exists = cur.fetchone() is not None
            
            if not fts_table_exists:
                # Создаем FTS5 таблицу только если она не существует
                conn.execute("""
                    CREATE VIRTUAL TABLE graph_nodes_fts USING fts5(
                        id, name, node_type, properties_text, user_id,
                        content='graph_nodes',
                        content_rowid='rowid'
                    )
                """)
                logger.info("Created new FTS5 table with properties_text column")
                
                # Переиндексируем существующие данные только при создании новой таблицы
                conn.execute("INSERT INTO graph_nodes_fts(graph_nodes_fts) VALUES('rebuild')")
                logger.info("Rebuilt FTS5 index with existing data")
            else:
                logger.info("FTS5 table already exists, skipping recreation")
            
            # Удаляем старые триггеры
            conn.execute("DROP TRIGGER IF EXISTS graph_nodes_ai")
            conn.execute("DROP TRIGGER IF EXISTS graph_nodes_ad")
            conn.execute("DROP TRIGGER IF EXISTS graph_nodes_au")
            
            # Создаем новые триггеры для FTS5
            conn.execute("""
                CREATE TRIGGER graph_nodes_ai AFTER INSERT ON graph_nodes BEGIN
                    INSERT INTO graph_nodes_fts(rowid, id, name, node_type, properties_text, user_id)
                    VALUES (new.rowid, new.id, new.name, new.node_type, new.properties_text, new.user_id);
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER graph_nodes_ad AFTER DELETE ON graph_nodes BEGIN
                    INSERT INTO graph_nodes_fts(graph_nodes_fts, rowid, id, name, node_type, properties_text, user_id)
                    VALUES('delete', old.rowid, old.id, old.name, old.node_type, old.properties_text, old.user_id);
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER graph_nodes_au AFTER UPDATE ON graph_nodes BEGIN
                    INSERT INTO graph_nodes_fts(graph_nodes_fts, rowid, id, name, node_type, properties_text, user_id)
                    VALUES('delete', old.rowid, old.id, old.name, old.node_type, old.properties_text, old.user_id);
                    INSERT INTO graph_nodes_fts(rowid, id, name, node_type, properties_text, user_id)
                    VALUES (new.rowid, new.id, new.name, new.node_type, new.properties_text, new.user_id);
                END
            """)
            
            # Создаем таблицу ребер
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT DEFAULT 'related',
                    strength REAL DEFAULT 0.5,
                    properties TEXT DEFAULT '{}',
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES graph_nodes(id),
                    FOREIGN KEY (target_id) REFERENCES graph_nodes(id)
                )
            """)
            
            # Создаем индексы для производительности
            conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_nodes_user_id ON graph_nodes(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_user_id ON graph_edges(user_id)")
            
            conn.commit()
    
    def _extract_properties_text(self, properties: Dict[str, Any]) -> str:
        """
        Извлекает текст из свойств для FTS5 поиска
        
        Args:
            properties: Словарь свойств
            
        Returns:
            Извлеченный текст
        """
        text_parts = []
        
        for key, value in properties.items():
            if isinstance(value, str):
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        text_parts.append(f"{key}: {item}")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        text_parts.append(f"{key}_{sub_key}: {sub_value}")
            else:
                text_parts.append(f"{key}: {str(value)}")
        
        return " ".join(text_parts)
    
    async def add_node(self, name: str, user_id: str, node_type: str = "concept",
                      properties: Optional[Dict[str, Any]] = None, importance: float = 0.5) -> str:
        """
        Добавить узел в граф
        
        Args:
            name: Название узла
            user_id: ID пользователя
            node_type: Тип узла
            properties: Свойства узла
            importance: Важность узла
            
        Returns:
            ID созданного узла
        """
        node_id = f"gn_{uuid.uuid4()}"
        properties = properties or {}
        now = datetime.now()
        
        # Логируем для отладки
        logger.info(f"Adding graph node: name='{name}', type='{node_type}', user_id='{user_id}'")
        
        # Извлекаем текст из свойств для FTS5 поиска
        properties_text = self._extract_properties_text(properties)
        
        try:
            logger.info(f"Starting database transaction for node: {node_id}")
            # Устанавливаем кодировку UTF-8 при подключении
            with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                # Устанавливаем правильную кодировку для UTF-8
                conn.execute("PRAGMA encoding = 'UTF-8'")
                # Устанавливаем режим UTF-8 для всех операций
                conn.execute("PRAGMA foreign_keys = ON")
                logger.info(f"Connected to database for node: {node_id}")
                
                # Проверяем, что FTS5 триггеры работают
                try:
                    conn.execute("""
                        INSERT INTO graph_nodes (id, name, node_type, properties, properties_text, importance, user_id, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (node_id, name, node_type, json.dumps(properties), properties_text, importance, user_id, now))
                    logger.info(f"INSERT executed for node: {node_id}")
                except Exception as fts_error:
                    logger.error(f"FTS5 trigger error for node {node_id}: {fts_error}")
                    raise
                
                conn.commit()
                logger.info(f"Graph node committed to database: {node_id} with name='{name}'")
                
                # Проверяем, что узел действительно сохранился
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM graph_nodes WHERE id = ?", (node_id,))
                count = cursor.fetchone()[0]
                if count == 0:
                    logger.error(f"CRITICAL: Node {node_id} not found after commit!")
                else:
                    logger.info(f"Node {node_id} verified in database")
                
                # Проверяем FTS5
                cursor.execute("SELECT COUNT(*) FROM graph_nodes_fts WHERE id = ?", (node_id,))
                fts_count = cursor.fetchone()[0]
                if fts_count == 0:
                    logger.error(f"CRITICAL: Node {node_id} not found in FTS5 after commit!")
                else:
                    logger.info(f"Node {node_id} verified in FTS5")
            
            # Проверяем, что узел сохранился ПОСЛЕ выхода из блока with
            logger.info(f"Exiting with block for node: {node_id}")
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM graph_nodes WHERE id = ?", (node_id,))
                count = cursor.fetchone()[0]
                if count == 0:
                    logger.error(f"CRITICAL: Node {node_id} not found after exiting with block!")
                else:
                    logger.info(f"Node {node_id} still exists after exiting with block")
                
                # Проверяем FTS5
                cursor.execute("SELECT COUNT(*) FROM graph_nodes_fts WHERE id = ?", (node_id,))
                fts_count = cursor.fetchone()[0]
                if fts_count == 0:
                    logger.error(f"CRITICAL: Node {node_id} not found in FTS5 after exiting with block!")
                else:
                    logger.info(f"Node {node_id} still exists in FTS5 after exiting with block")
            
            logger.info(f"Added graph node: {node_id} with name='{name}'")
            return node_id
        except Exception as e:
            logger.error(f"Failed to add graph node: {node_id} with name='{name}': {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    async def add_edge(self, source_id: str, target_id: str, user_id: str,
                      relationship_type: str = "related", strength: float = 0.5,
                      properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Добавить ребро в граф
        
        Args:
            source_id: ID узла-источника
            target_id: ID узла-цели
            user_id: ID пользователя
            relationship_type: Тип связи
            strength: Сила связи
            properties: Свойства связи
            
        Returns:
            ID созданного ребра
        """
        edge_id = f"ge_{uuid.uuid4()}"
        properties = properties or {}
        now = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO graph_edges (id, source_id, target_id, relationship_type, strength, properties, user_id, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (edge_id, source_id, target_id, relationship_type, strength, json.dumps(properties), user_id, now))
            conn.commit()
        
        logger.info(f"Added graph edge: {edge_id}")
        return edge_id
    
    async def get_node(self, node_id: str, user_id: str) -> Optional[GraphNode]:
        """
        Получить узел по ID
        
        Args:
            node_id: ID узла
            user_id: ID пользователя
            
        Returns:
            Узел или None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("""
                SELECT * FROM graph_nodes WHERE id = ? AND user_id = ?
            """, (node_id, user_id))
            row = cur.fetchone()
            
            if not row:
                return None
            
            return GraphNode(
                id=row['id'],
                name=row['name'],
                node_type=row['node_type'],
                properties=json.loads(row['properties']),
                importance=row['importance'],
                last_updated=datetime.fromisoformat(str(row['last_updated']))
            )
    
    async def search_nodes(self, user_id: str, query: str, 
                          node_type: Optional[str] = None, limit: int = 10) -> List[GraphNode]:
        """
        Поиск узлов в графе с использованием FTS5
        
        Args:
            user_id: ID пользователя
            query: Поисковый запрос
            node_type: Тип узла для фильтрации
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных узлов
        """
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.row_factory = sqlite3.Row
            # Устанавливаем правильную кодировку для UTF-8
            conn.execute("PRAGMA encoding = 'UTF-8'")
            
            # Используем FTS5 для поиска
            # Преобразуем multi-word query в OR запрос для FTS5
            # Пример: "AIRI система" -> "AIRI OR система"
            words = query.strip().split()
            if len(words) > 1:
                fts_query = " OR ".join(words)
            elif len(words) == 1:
                fts_query = query
            else:
                # Пустой запрос - используем поиск всех узлов пользователя через обычный SQL
                # Вместо FTS5 используем обычный SELECT
                if node_type:
                    cur = conn.execute("""
                        SELECT gn.*, 1.0 as rank
                        FROM graph_nodes gn
                        WHERE gn.user_id = ? AND gn.node_type = ?
                        ORDER BY gn.created_at DESC
                        LIMIT ?
                    """, (user_id, node_type, limit))
                else:
                    cur = conn.execute("""
                        SELECT gn.*, 1.0 as rank
                        FROM graph_nodes gn
                        WHERE gn.user_id = ?
                        ORDER BY gn.created_at DESC
                        LIMIT ?
                    """, (user_id, limit))
                
                rows = cur.fetchall()
                nodes = []
                for row in rows:
                    node = GraphNode(
                        id=row['id'],
                        name=row['name'],
                        node_type=row['node_type'],
                        properties=json.loads(row['properties']) if row['properties'] else {},
                        importance=row['importance'],
                        last_updated=row['last_updated']
                    )
                    nodes.append(node)
                return nodes
            
            # Строим запрос в зависимости от наличия фильтра по типу узла
            if node_type:
                cur = conn.execute("""
                    SELECT gn.*, rank
                    FROM graph_nodes gn
                    JOIN graph_nodes_fts fts ON gn.id = fts.id
                    WHERE fts.user_id = ? AND gn.node_type = ? AND graph_nodes_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (user_id, node_type, fts_query, limit))
            else:
                cur = conn.execute("""
                    SELECT gn.*, rank
                    FROM graph_nodes gn
                    JOIN graph_nodes_fts fts ON gn.id = fts.id
                    WHERE fts.user_id = ? AND graph_nodes_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (user_id, fts_query, limit))
            
            rows = cur.fetchall()
            
            nodes = []
            for row in rows:
                # Вычисляем релевантность на основе FTS5 ранга
                # FTS5 rank обычно отрицательный (чем ближе к 0, тем лучше)
                # Преобразуем в положительный relevance score
                rank = abs(float(row['rank']))
                relevance = 1.0 / (rank + 1) if rank > 0 else 1.0
                
                # Фильтруем по релевантности (более мягкий порог)
                if relevance > 0.05:  # Минимальный порог релевантности
                    # Логируем для отладки
                    logger.debug(f"Found node: id={row['id']}, name='{row['name']}', type={row['node_type']}")
                    
                    node = GraphNode(
                        id=row['id'],
                        name=row['name'] or f"Node_{row['id'][:8]}",  # Fallback если name пустой
                        node_type=row['node_type'],
                        properties=json.loads(row['properties']),
                        importance=row['importance'],
                        last_updated=datetime.fromisoformat(str(row['last_updated']))
                    )
                    nodes.append(node)
            
            return nodes
    
    async def get_all_nodes(self, user_id: str) -> List[GraphNode]:
        """
        Получить все узлы пользователя
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Список всех узлов
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("""
                SELECT * FROM graph_nodes WHERE user_id = ? ORDER BY importance DESC
            """, (user_id,))
            rows = cur.fetchall()
            
            nodes = []
            for row in rows:
                node = GraphNode(
                    id=row['id'],
                    name=row['name'],
                    node_type=row['node_type'],
                    properties=json.loads(row['properties']),
                    importance=row['importance'],
                    last_updated=datetime.fromisoformat(str(row['last_updated']))
                )
                nodes.append(node)
            
            return nodes
    
    async def get_node_edges(self, node_id: str, user_id: str) -> List[GraphEdge]:
        """
        Получить все ребра для узла
        
        Args:
            node_id: ID узла
            user_id: ID пользователя
            
        Returns:
            Список ребер
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("""
                SELECT * FROM graph_edges 
                WHERE user_id = ? AND (source_id = ? OR target_id = ?)
                ORDER BY strength DESC
            """, (user_id, node_id, node_id))
            rows = cur.fetchall()
            
            edges = []
            for row in rows:
                edge = GraphEdge(
                    id=row['id'],
                    source_id=row['source_id'],
                    target_id=row['target_id'],
                    relationship_type=row['relationship_type'],
                    strength=row['strength'],
                    properties=json.loads(row['properties']),
                    last_updated=datetime.fromisoformat(str(row['last_updated']))
                )
                edges.append(edge)
            
            return edges
    
    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Получить статистику графа
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Словарь со статистикой
        """
        with sqlite3.connect(self.db_path) as conn:
            # Статистика узлов
            cur = conn.execute("""
                SELECT 
                    COUNT(*) as total_nodes,
                    AVG(importance) as avg_importance,
                    node_type,
                    COUNT(*) as count
                FROM graph_nodes 
                WHERE user_id = ?
                GROUP BY node_type
            """, (user_id,))
            node_stats = cur.fetchall()
            
            # Статистика ребер
            cur = conn.execute("""
                SELECT 
                    COUNT(*) as total_edges,
                    AVG(strength) as avg_strength,
                    relationship_type,
                    COUNT(*) as count
                FROM graph_edges 
                WHERE user_id = ?
                GROUP BY relationship_type
            """, (user_id,))
            edge_stats = cur.fetchall()
            
            # Общая статистика
            cur = conn.execute("""
                SELECT 
                    COUNT(*) as total_nodes,
                    AVG(importance) as avg_importance
                FROM graph_nodes 
                WHERE user_id = ?
            """, (user_id,))
            total_nodes = cur.fetchone()
            
            cur = conn.execute("""
                SELECT 
                    COUNT(*) as total_edges,
                    AVG(strength) as avg_strength
                FROM graph_edges 
                WHERE user_id = ?
            """, (user_id,))
            total_edges = cur.fetchone()
            
            # Формируем результат
            node_types = {row[2]: row[3] for row in node_stats}
            relationship_types = {row[2]: row[3] for row in edge_stats}
            
            return {
                "total_nodes": total_nodes[0] if total_nodes else 0,
                "total_edges": total_edges[0] if total_edges else 0,
                "avg_node_importance": total_nodes[1] if total_nodes else 0.0,
                "avg_edge_strength": total_edges[1] if total_edges else 0.0,
                "node_types": node_types,
                "relationship_types": relationship_types
            }
    
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
            # Извлекаем текст из свойств для FTS5 поиска
            properties_text = self._extract_properties_text(properties)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE graph_nodes 
                    SET properties = ?, properties_text = ?, last_updated = ?
                    WHERE id = ?
                """, (json.dumps(properties), properties_text, datetime.now(), node_id))
                conn.commit()
            
            logger.info(f"Updated node properties: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating node properties: {e}")
            return False
    
    async def update_edge_strength(self, edge_id: str, strength: float) -> bool:
        """
        Обновить силу связи
        
        Args:
            edge_id: ID ребра
            strength: Новая сила связи
            
        Returns:
            True если успешно
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE graph_edges 
                    SET strength = ?, last_updated = ?
                    WHERE id = ?
                """, (strength, datetime.now(), edge_id))
                conn.commit()
            
            logger.info(f"Updated edge strength: {edge_id}")
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
            with sqlite3.connect(self.db_path) as conn:
                # Сначала удаляем все связанные ребра
                conn.execute("DELETE FROM graph_edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
                # Затем удаляем узел
                conn.execute("DELETE FROM graph_nodes WHERE id = ?", (node_id,))
                conn.commit()
            
            logger.info(f"Removed node: {node_id}")
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
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM graph_edges WHERE id = ?", (edge_id,))
                conn.commit()
            
            logger.info(f"Removed edge: {edge_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing edge: {e}")
            return False
    
    async def update_edge(self, edge_id: str, user_id: str, 
                         properties: Optional[Dict[str, Any]] = None,
                         relationship_type: Optional[str] = None) -> bool:
        """
        Обновить ребро в графе
        
        Args:
            edge_id: ID ребра
            user_id: ID пользователя
            properties: Новые свойства
            relationship_type: Новый тип связи
            
        Returns:
            True если успешно
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Проверяем, существует ли ребро
                cur = conn.execute("SELECT id FROM graph_edges WHERE id = ? AND user_id = ?", 
                                 (edge_id, user_id))
                if not cur.fetchone():
                    logger.warning(f"Edge {edge_id} not found for user {user_id}")
                    return False
                
                # Обновляем свойства
                if properties is not None:
                    properties_json = json.dumps(properties)
                    conn.execute("""
                        UPDATE graph_edges 
                        SET properties = ? 
                        WHERE id = ? AND user_id = ?
                    """, (properties_json, edge_id, user_id))
                
                # Обновляем тип связи
                if relationship_type is not None:
                    conn.execute("""
                        UPDATE graph_edges 
                        SET relationship_type = ? 
                        WHERE id = ? AND user_id = ?
                    """, (relationship_type, edge_id, user_id))
                
                conn.commit()
            
            logger.info(f"Updated edge: {edge_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating edge: {e}")
            return False
    
    async def get_node_by_id(self, node_id: str, user_id: str) -> Optional[GraphNode]:
        """
        Получить узел графа по ID
        
        Args:
            node_id: ID узла
            user_id: ID пользователя
            
        Returns:
            GraphNode или None если не найден
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT * FROM graph_nodes 
                    WHERE id = ? AND user_id = ?
                """, (node_id, user_id))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                return self._row_to_node(row)
                
        except Exception as e:
            logger.error(f"Error getting node by ID {node_id}: {e}")
            return None
    
    async def update_node_metadata(self, node_id: str, user_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Обновить метаданные узла графа
        
        Args:
            node_id: ID узла
            user_id: ID пользователя
            metadata: Новые метаданные
            
        Returns:
            True если успешно, False если ошибка
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Проверяем, существует ли узел
                cur = conn.execute("SELECT id FROM graph_nodes WHERE id = ? AND user_id = ?", 
                                 (node_id, user_id))
                if not cur.fetchone():
                    logger.warning(f"Node {node_id} not found for user {user_id}")
                    return False
                
                # Обновляем метаданные
                for key, value in metadata.items():
                    if key == "properties" and isinstance(value, dict):
                        # Обновляем свойства узла
                        properties_json = json.dumps(value)
                        conn.execute("""
                            UPDATE graph_nodes 
                            SET properties = ? 
                            WHERE id = ? AND user_id = ?
                        """, (properties_json, node_id, user_id))
                    elif key == "importance" and isinstance(value, (int, float)):
                        # Обновляем важность
                        conn.execute("""
                            UPDATE graph_nodes 
                            SET importance = ? 
                            WHERE id = ? AND user_id = ?
                        """, (value, node_id, user_id))
                    elif key == "node_type" and isinstance(value, str):
                        # Обновляем тип узла
                        conn.execute("""
                            UPDATE graph_nodes 
                            SET node_type = ? 
                            WHERE id = ? AND user_id = ?
                        """, (value, node_id, user_id))
                    elif key == "name" and isinstance(value, str):
                        # Обновляем имя узла
                        conn.execute("""
                            UPDATE graph_nodes 
                            SET name = ? 
                            WHERE id = ? AND user_id = ?
                        """, (value, node_id, user_id))
                
                # Обновляем время последнего изменения
                conn.execute("""
                    UPDATE graph_nodes 
                    SET last_updated = ? 
                    WHERE id = ? AND user_id = ?
                """, (datetime.now().isoformat(), node_id, user_id))
                
                conn.commit()
            
            logger.debug(f"Updated metadata for graph node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating graph node metadata {node_id}: {e}")
            return False
