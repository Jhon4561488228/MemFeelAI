"""
SQLite FTS5 Full-Text Search implementation for AIRI Memory System.
Provides keyword-based search capabilities to complement semantic search.
"""

import sqlite3
import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from loguru import logger
import os
import time

class FTS5SearchEngine:
    """SQLite FTS5-based keyword search engine with query preparation and stop words filtering"""
    
    # Стоп-слова для русского и английского языков
    STOP_WORDS = {
        # Русские стоп-слова
        'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между',
        # Английские стоп-слова
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with', 'i', 'you', 'we', 'they', 'this', 'these', 'those', 'have', 'had', 'do', 'does', 'did', 'can', 'could', 'should', 'would', 'may', 'might', 'must', 'shall', 'am', 'are', 'is', 'was', 'were', 'been', 'being', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'
    }
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize FTS5 search engine
        
        Args:
            db_path: Path to SQLite database. If None, uses default from env.
        """
        if db_path is None:
            env_data_dir = os.getenv("AIRI_DATA_DIR", "./data")
            db_path_env = os.getenv("SQLITE_DB")
            self.db_path = Path(db_path_env) if db_path_env else Path(env_data_dir) / "memory_system.db"
        else:
            self.db_path = Path(db_path)
        
        self._lock = asyncio.Lock()
        logger.info(f"FTS5SearchEngine initialized with database: {self.db_path}")
    
    def _prepare_fts_query(self, query: str) -> str:
        """
        Подготовка FTS5 запроса с обработкой стоп-слов и нормализацией
        
        Args:
            query: Исходный поисковый запрос
            
        Returns:
            Обработанный запрос для FTS5
        """
        if not query or not query.strip():
            return ""
        
        # Нормализация: приводим к нижнему регистру
        normalized_query = query.lower().strip()
        
        # Сохраняем timestamp паттерны перед удалением знаков препинания
        # Заменяем timestamp на placeholder, чтобы не потерять его
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}\.\d+)'
        timestamp_placeholders = {}
        timestamp_matches = re.findall(timestamp_pattern, normalized_query)
        for i, match in enumerate(timestamp_matches):
            placeholder = f"__timestamp_{i}__"
            timestamp_placeholders[placeholder] = match
            normalized_query = normalized_query.replace(match, placeholder)
        
        # Удаляем лишние пробелы и знаки препинания
        normalized_query = re.sub(r'[^\w\s\-]', ' ', normalized_query)
        normalized_query = re.sub(r'\s+', ' ', normalized_query).strip()
        
        # Восстанавливаем timestamp'ы
        for placeholder, timestamp in timestamp_placeholders.items():
            normalized_query = normalized_query.replace(placeholder, timestamp)
        
        # Разбиваем на слова
        words = normalized_query.split()
        
        # Фильтруем стоп-слова
        filtered_words = []
        for word in words:
            # Убираем стоп-слова, но оставляем слова длиннее 2 символов
            if word not in self.STOP_WORDS and len(word) > 2:
                filtered_words.append(word)
        
        # Если все слова были стоп-словами, возвращаем исходный запрос
        if not filtered_words:
            logger.debug(f"All words in query '{query}' were stop words, using original query")
            return query.strip()
        
        # Объединяем отфильтрованные слова
        prepared_query = ' '.join(filtered_words)
        
        logger.debug(f"Query prepared: '{query}' -> '{prepared_query}' (removed {len(words) - len(filtered_words)} stop words)")
        return prepared_query
    
    def _detect_query_type(self, query: str) -> str:
        """
        Определение типа запроса для выбора стратегии поиска
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Тип запроса: 'phrase', 'boolean', 'wildcard', 'simple'
        """
        query = query.strip()
        
        # Фразовый поиск (в кавычках)
        if query.startswith('"') and query.endswith('"'):
            return 'phrase'
        
        # Булевый поиск
        if any(op in query.upper() for op in [' AND ', ' OR ', ' NOT ']):
            return 'boolean'
        
        # Поиск с подстановочными знаками
        if '*' in query or '?' in query:
            return 'wildcard'
        
        # Простой поиск
        return 'simple'
    
    async def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper settings"""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    
    async def index_memory(self, memory_id: str, content: str, user_id: str, 
                          memory_type: str = "general", importance: float = 0.5) -> bool:
        """
        Index a memory for full-text search
        
        Args:
            memory_id: Unique memory identifier
            content: Text content to index
            user_id: User identifier
            memory_type: Type of memory (working, short_term, etc.)
            importance: Importance score (0.0-1.0)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                conn = await self._get_connection()
                try:
                    cur = conn.cursor()
                    
                    # Insert or replace in FTS5 table
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO memory_fts 
                        (memory_id, content, user_id, memory_type, importance, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (memory_id, content, user_id, memory_type, importance, int(time.time()))
                    )
                    conn.commit()
                    
                    logger.debug(f"Indexed memory {memory_id} for user {user_id}")
                    return True
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(f"FTS5SearchEngine.index_memory failed for memory {memory_id}: {e}")
            return False
    
    async def search(self, query: str, user_id: str, limit: int = 10, 
                    memory_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform full-text search with query preparation and stop words filtering
        
        Args:
            query: Search query
            user_id: User identifier
            limit: Maximum number of results
            memory_types: Filter by memory types (optional)
            
        Returns:
            List of search results with scores
        """
        try:
            # Проверяем на пустой запрос
            if not query or not query.strip():
                logger.debug(f"FTS5 search skipped for empty query")
                return []
            
            # Определяем тип запроса
            query_type = self._detect_query_type(query)
            
            # Подготавливаем запрос (кроме специальных типов)
            if query_type in ['simple']:
                prepared_query = self._prepare_fts_query(query)
                if not prepared_query:
                    logger.debug(f"FTS5 search skipped - no meaningful words after stop words filtering")
                    return []
            else:
                # Для фразового, булевого и wildcard поиска используем исходный запрос
                prepared_query = query.strip()
            
            logger.debug(f"FTS5 search: type={query_type}, original='{query}', prepared='{prepared_query}'")
            
            async with self._lock:
                conn = await self._get_connection()
                try:
                    cur = conn.cursor()
                    
                    # Build FTS5 query with user filter
                    fts_query = f"memory_fts MATCH ? AND user_id = ?"
                    params = [prepared_query, user_id]
                    
                    # Add memory type filter if specified
                    if memory_types:
                        placeholders = ",".join("?" * len(memory_types))
                        fts_query += f" AND memory_type IN ({placeholders})"
                        params.extend(memory_types)
                    
                    # Execute FTS5 search with ranking
                    sql = f"""
                        SELECT 
                            memory_id,
                            content,
                            user_id,
                            memory_type,
                            importance,
                            created_at,
                            bm25(memory_fts) as fts_score
                        FROM memory_fts 
                        WHERE {fts_query}
                        ORDER BY fts_score ASC, importance DESC, created_at DESC
                        LIMIT {limit}
                    """
                    
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    
                    results = []
                    for row in rows:
                        results.append({
                            "memory_id": row['memory_id'],
                            "content": row['content'],
                            "user_id": row['user_id'],
                            "memory_type": row['memory_type'],
                            "importance": row['importance'],
                            "created_at": row['created_at'],
                            "fts_score": row['fts_score'],
                            "search_type": "keyword"
                        })
                    
                    logger.debug(f"FTS5 search for '{query}' (type: {query_type}) returned {len(results)} results")
                    return results
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(f"FTS5SearchEngine.search failed for query '{query}': {e}")
            return []
    
    async def search_phrase(self, phrase: str, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for exact phrase
        
        Args:
            phrase: Exact phrase to search for
            user_id: User identifier
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        # Wrap phrase in quotes for exact match
        quoted_phrase = f'"{phrase}"'
        return await self.search(quoted_phrase, user_id, limit)
    
    async def search_with_operators(self, query: str, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search with FTS5 operators (AND, OR, NOT, etc.)
        
        Args:
            query: Query with FTS5 operators
            user_id: User identifier
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        return await self.search(query, user_id, limit)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Remove memory from FTS5 index
        
        Args:
            memory_id: Memory identifier to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                conn = await self._get_connection()
                try:
                    cur = conn.cursor()
                    
                    cur.execute(
                        "DELETE FROM memory_fts WHERE memory_id = ?",
                        (memory_id,)
                    )
                    conn.commit()
                    
                    logger.debug(f"Removed memory {memory_id} from FTS5 index")
                    return True
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(f"FTS5SearchEngine.delete_memory failed for memory {memory_id}: {e}")
            return False
    
    async def update_memory(self, memory_id: str, content: str, user_id: str,
                           memory_type: str = "general", importance: float = 0.5) -> bool:
        """
        Update memory in FTS5 index
        
        Args:
            memory_id: Memory identifier
            content: New text content
            user_id: User identifier
            memory_type: Type of memory
            importance: Importance score
            
        Returns:
            True if successful, False otherwise
        """
        return await self.index_memory(memory_id, content, user_id, memory_type, importance)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get FTS5 search statistics
        
        Returns:
            Dictionary with search statistics
        """
        try:
            async with self._lock:
                conn = await self._get_connection()
                try:
                    cur = conn.cursor()
                    
                    # Total indexed memories
                    cur.execute("SELECT COUNT(*) as total FROM memory_fts")
                    total = cur.fetchone()['total']
                    
                    # Memories by type
                    cur.execute(
                        "SELECT memory_type, COUNT(*) as count FROM memory_fts GROUP BY memory_type"
                    )
                    by_type = {row['memory_type']: row['count'] for row in cur.fetchall()}
                    
                    # Average importance
                    cur.execute("SELECT AVG(importance) as avg_importance FROM memory_fts")
                    avg_importance = cur.fetchone()['avg_importance'] or 0.0
                    
                    return {
                        "total_indexed": total,
                        "by_type": by_type,
                        "average_importance": round(avg_importance, 3)
                    }
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(f"FTS5SearchEngine.get_stats failed: {e}")
            return {"error": str(e)}
    
    async def optimize_index(self) -> bool:
        """
        Optimize FTS5 index for better performance
        
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                conn = await self._get_connection()
                try:
                    cur = conn.cursor()
                    
                    # Rebuild FTS5 index
                    cur.execute("INSERT INTO memory_fts(memory_fts) VALUES('rebuild')")
                    conn.commit()
                    
                    logger.info("FTS5 index optimized")
                    return True
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(f"FTS5SearchEngine.optimize_index failed: {e}")
            return False

# Global FTS5 search engine instance
_fts5_engine: Optional[FTS5SearchEngine] = None

async def get_fts5_engine() -> FTS5SearchEngine:
    """Get global FTS5 search engine instance"""
    global _fts5_engine
    if _fts5_engine is None:
        _fts5_engine = FTS5SearchEngine()
    return _fts5_engine
