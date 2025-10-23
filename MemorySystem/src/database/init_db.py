"""
Database initialization utilities for AIRI Memory System.

Note: Current managers use ChromaDB for persistence and initialize their
own storage on demand. We keep this initializer to ensure the base data
directory exists and to retain a single place for future DB setup.
"""

from pathlib import Path
from loguru import logger
import sqlite3
import os


async def ensure_databases_initialized(data_dir: str = "./data") -> None:
    """
    Ensure SQLite databases and required tables are created.

    Args:
        data_dir: Base directory to hold SQLite database files.
    """
    try:
        # Resolve data_dir and db path from env
        env_data_dir = os.getenv("AIRI_DATA_DIR", data_dir)
        Path(env_data_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Data directory ensured. Managers will initialize storage on demand.")

        # SQLite: создаём файл БД для графа/целей
        db_path_env = os.getenv("SQLITE_DB")
        db_path = Path(db_path_env) if db_path_env else Path(env_data_dir) / "memory_system.db"
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            # Таблица рёбер графа
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_edges (
                    from_id TEXT NOT NULL,
                    to_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    user_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_user ON memory_edges(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_from_to ON memory_edges(from_id, to_id)")

            # Таблица целей
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    trigger_type TEXT,
                    trigger_value TEXT,
                    action TEXT,
                    status TEXT,
                    progress REAL DEFAULT 0.0,
                    next_run DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id)")

            # Таблица кэша
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed INTEGER NOT NULL,
                    cache_type TEXT DEFAULT 'general'
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_entries(cache_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cache_last_accessed ON cache_entries(last_accessed)")
            
            # FTS5 таблица для полнотекстового поиска
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    memory_id UNINDEXED,
                    content,
                    user_id UNINDEXED,
                    memory_type UNINDEXED,
                    importance UNINDEXED,
                    created_at UNINDEXED,
                    tokenize='porter unicode61'
                )
                """
            )
            
            # Примечание: FTS5 виртуальные таблицы не могут иметь обычные индексы
            # FTS5 автоматически создает внутренние индексы для оптимизации поиска
            
            conn.commit()
            logger.info(f"SQLite initialized at {db_path} with FTS5 support")
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to initialize data directory: {e}")
        raise

