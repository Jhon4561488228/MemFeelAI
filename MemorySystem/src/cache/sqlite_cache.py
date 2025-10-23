"""
SQLite-based cache implementation for AIRI Memory System.
Provides persistent caching with TTL support and access tracking.
"""

import json
import time
import sqlite3
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger
import os

class SQLiteCache:
    """SQLite-based cache with TTL and access tracking"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, db_path: Optional[str] = None):
        """Singleton pattern для предотвращения множественной инициализации"""
        if cls._instance is None:
            cls._instance = super(SQLiteCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize SQLite cache"""
        if self._initialized:
            return  # Уже инициализирован
        if db_path is None:
            env_data_dir = os.getenv("AIRI_DATA_DIR", "./data")
            db_path_env = os.getenv("SQLITE_DB")
            self.db_path = Path(db_path_env) if db_path_env else Path(env_data_dir) / "memory_system.db"
        else:
            self.db_path = Path(db_path)
        
        self._lock = asyncio.Lock()
        self._initialized = False
        logger.info(f"SQLiteCache initialized with database: {self.db_path}")
        self._initialized = True
    
    async def _ensure_initialized(self):
        """Ensure cache database is initialized"""
        if not self._initialized:
            # Используем отдельную блокировку для инициализации
            if not hasattr(self, '_init_lock'):
                self._init_lock = asyncio.Lock()
            async with self._init_lock:
                if not self._initialized:
                    try:
                        # Create directory if it doesn't exist
                        self.db_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Initialize database schema with optimizations
                        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
                        conn.row_factory = sqlite3.Row
                        try:
                            cur = conn.cursor()
                            
                            # ОПТИМИЗАЦИЯ: Настройки SQLite для производительности
                            cur.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
                            cur.execute("PRAGMA synchronous=NORMAL")  # Быстрее чем FULL
                            cur.execute("PRAGMA cache_size=10000")  # Больше кэш
                            cur.execute("PRAGMA temp_store=MEMORY")  # Временные таблицы в памяти
                            
                            cur.execute("""
                                CREATE TABLE IF NOT EXISTS cache_entries (
                                    key TEXT PRIMARY KEY,
                                    value TEXT NOT NULL,
                                    created_at INTEGER NOT NULL,
                                    expires_at INTEGER NOT NULL,
                                    access_count INTEGER DEFAULT 0,
                                    last_accessed INTEGER NOT NULL,
                                    cache_type TEXT DEFAULT 'general'
                                )
                            """)
                            
                            # ОПТИМИЗАЦИЯ: Индексы для быстрого поиска
                            cur.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)")
                            cur.execute("CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_entries(cache_type)")
                            
                            conn.commit()
                            self._initialized = True
                            logger.debug("SQLiteCache database initialized with optimizations")
                        finally:
                            conn.close()
                    except Exception as e:
                        logger.warning(f"Failed to initialize SQLiteCache: {e}")
                        self._initialized = False

    async def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper settings"""
        await self._ensure_initialized()
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    
    async def set(self, key: str, value: dict, ttl_sec: int = 600) -> bool:
        """
        Set cache entry with TTL
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl_sec: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                now = int(time.time())
                expires_at = now + ttl_sec
                
                conn = await self._get_connection()
                try:
                    cur = conn.cursor()
                    
                    # Insert or replace cache entry
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO cache_entries 
                        (key, value, created_at, expires_at, access_count, last_accessed, cache_type)
                        VALUES (?, ?, ?, ?, 0, ?, ?)
                        """,
                        (key, json.dumps(value), now, expires_at, now, self._get_cache_type(key))
                    )
                    conn.commit()
                    return True
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(f"SQLiteCache.set failed for key '{key}': {e}")
            return False
    
    async def get(self, key: str) -> Optional[dict]:
        """
        Get cache entry
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            async with self._lock:
                now = int(time.time())
                
                conn = await self._get_connection()
                try:
                    cur = conn.cursor()
                    
                    # Get cache entry and update access info
                    cur.execute(
                        """
                        SELECT value, expires_at FROM cache_entries 
                        WHERE key = ? AND expires_at > ?
                        """,
                        (key, now)
                    )
                    
                    row = cur.fetchone()
                    if row:
                        # Update access count and last accessed time
                        cur.execute(
                            """
                            UPDATE cache_entries 
                            SET access_count = access_count + 1, last_accessed = ?
                            WHERE key = ?
                            """,
                            (now, key)
                        )
                        conn.commit()
                        
                        return json.loads(row['value'])
                    else:
                        return None
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(f"SQLiteCache.get failed for key '{key}': {e}")
            return None
    
    async def delete_prefix(self, prefix: str) -> int:
        """
        Delete cache entries by prefix
        
        Args:
            prefix: Key prefix to match
            
        Returns:
            Number of deleted entries
        """
        logger.info(f"SQLiteCache.delete_prefix called with prefix: {prefix}")
        try:
            logger.info(f"Acquiring lock for delete_prefix: {prefix}")
            # Добавляем таймаут для блокировки (5 секунд)
            try:
                await asyncio.wait_for(self._lock.acquire(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.error(f"Timeout acquiring lock for delete_prefix: {prefix}")
                return 0
            
            try:
                logger.info(f"Lock acquired, getting connection for delete_prefix: {prefix}")
                conn = await self._get_connection()
                try:
                    cur = conn.cursor()
                    
                    # Count entries to delete
                    logger.info(f"Counting entries for delete_prefix: {prefix}")
                    cur.execute(
                        "SELECT COUNT(*) as count FROM cache_entries WHERE key LIKE ?",
                        (f"{prefix}%",)
                    )
                    count = cur.fetchone()['count']
                    logger.info(f"Found {count} entries to delete for prefix: {prefix}")
                    
                    # Delete entries
                    logger.info(f"Deleting entries for delete_prefix: {prefix}")
                    cur.execute(
                        "DELETE FROM cache_entries WHERE key LIKE ?",
                        (f"{prefix}%",)
                    )
                    conn.commit()
                    logger.info(f"Deleted {count} cache entries with prefix '{prefix}'")
                    return count
                finally:
                    conn.close()
                    logger.info(f"Connection closed for delete_prefix: {prefix}")
            finally:
                self._lock.release()
                logger.info(f"Lock released for delete_prefix: {prefix}")
        except Exception as e:
            logger.error(f"SQLiteCache.delete_prefix failed for prefix '{prefix}': {e}")
            import traceback
            logger.error(f"SQLiteCache delete_prefix traceback: {traceback.format_exc()}")
            return 0
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries
        
        Returns:
            Number of cleaned up entries
        """
        try:
            async with self._lock:
                now = int(time.time())
                
                conn = await self._get_connection()
                try:
                    cur = conn.cursor()
                    
                    # Count expired entries
                    cur.execute(
                        "SELECT COUNT(*) as count FROM cache_entries WHERE expires_at <= ?",
                        (now,)
                    )
                    count = cur.fetchone()['count']
                    
                    # Delete expired entries
                    cur.execute(
                        "DELETE FROM cache_entries WHERE expires_at <= ?",
                        (now,)
                    )
                    conn.commit()
                    
                    if count > 0:
                        logger.info(f"Cleaned up {count} expired cache entries")
                    return count
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(f"SQLiteCache.cleanup_expired failed: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            async with self._lock:
                conn = await self._get_connection()
                try:
                    cur = conn.cursor()
                    
                    # Total entries
                    cur.execute("SELECT COUNT(*) as total FROM cache_entries")
                    total = cur.fetchone()['total']
                    
                    # Expired entries
                    now = int(time.time())
                    cur.execute(
                        "SELECT COUNT(*) as expired FROM cache_entries WHERE expires_at <= ?",
                        (now,)
                    )
                    expired = cur.fetchone()['expired']
                    
                    # Entries by type
                    cur.execute(
                        "SELECT cache_type, COUNT(*) as count FROM cache_entries GROUP BY cache_type"
                    )
                    by_type = {row['cache_type']: row['count'] for row in cur.fetchall()}
                    
                    return {
                        "total_entries": total,
                        "expired_entries": expired,
                        "active_entries": total - expired,
                        "by_type": by_type
                    }
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(f"SQLiteCache.get_stats failed: {e}")
            return {"error": str(e)}
    
    def _get_cache_type(self, key: str) -> str:
        """Determine cache type from key"""
        if key.startswith("ml_search:"):
            return "search"
        elif key.startswith("embedding:"):
            return "embedding"
        elif key.startswith("optimization:"):
            return "optimization"
        elif key.startswith("concepts:"):
            return "concepts"
        elif key.startswith("entities:"):
            return "entities"
        else:
            return "general"

# Global SQLite cache instance
_sqlite_cache: Optional[SQLiteCache] = None

async def get_sqlite_cache() -> SQLiteCache:
    """Get global SQLite cache instance"""
    global _sqlite_cache
    if _sqlite_cache is None:
        _sqlite_cache = SQLiteCache()
    return _sqlite_cache
