import json
import time
import asyncio
from typing import Optional, Dict, Tuple
from loguru import logger
import os

# SQLite cache as primary storage
from .sqlite_cache import get_sqlite_cache

# In-process fallback cache (used when SQLite fails)
_local_cache: Dict[str, Tuple[str, float]] = {}
_local_lock = asyncio.Lock()
_local_max_entries = int(os.getenv("CACHE_LOCAL_MAX", "500"))
_default_ttl = int(os.getenv("CACHE_DEFAULT_TTL", "600"))


async def cache_set(key: str, value: dict, ttl_sec: int = None) -> bool:
    if ttl_sec is None:
        ttl_sec = _default_ttl
    
    # Primary: SQLite cache
    try:
        sqlite_cache = await get_sqlite_cache()
        success = await sqlite_cache.set(key, value, ttl_sec)
        if success:
            return True
    except Exception as e:
        logger.warning(f"cache_set (sqlite) failed: {e}")
    
    # Fallback: local in-memory TTL cache
    try:
        async with _local_lock:
            # Trim if above capacity
            if len(_local_cache) >= _local_max_entries:
                # drop oldest by expiry
                try:
                    oldest_key = min(_local_cache.keys(), key=lambda k: _local_cache[k][1])
                    _local_cache.pop(oldest_key, None)
                except Exception:
                    _local_cache.clear()
            _local_cache[key] = (json.dumps(value), time.time() + float(ttl_sec))
        return True
    except Exception as e:
        logger.warning(f"cache_set (local) failed: {e}")
        return False


async def cache_get(key: str) -> Optional[dict]:
    # Primary: SQLite cache
    try:
        sqlite_cache = await get_sqlite_cache()
        result = await sqlite_cache.get(key)
        if result is not None:
            return result
    except Exception as e:
        logger.warning(f"cache_get (sqlite) failed: {e}")
    
    # Fallback: local in-memory TTL cache
    try:
        async with _local_lock:
            item = _local_cache.get(key)
            if not item:
                return None
            raw, expires_at = item
            if time.time() > expires_at:
                # expired
                _local_cache.pop(key, None)
                return None
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"cache_get (local) failed: {e}")
        return None


async def cache_delete_prefix(prefix: str) -> int:
    """Удалить элементы кэша по префиксу ключа. Возвращает количество удалённых."""
    logger.info(f"cache_delete_prefix called with prefix: {prefix}")
    deleted = 0
    
    # Primary: SQLite cache
    try:
        logger.info(f"Getting SQLite cache for prefix: {prefix}")
        sqlite_cache = await get_sqlite_cache()
        logger.info(f"SQLite cache obtained, calling delete_prefix for: {prefix}")
        result = await sqlite_cache.delete_prefix(prefix)
        if result is not None:
            deleted += result
        logger.info(f"SQLite delete_prefix completed, deleted: {deleted}")
    except Exception as e:
        logger.error(f"cache_delete_prefix (sqlite) failed: {e}")
        import traceback
        logger.error(f"SQLite cache delete_prefix traceback: {traceback.format_exc()}")
    
    # Fallback: local in-memory cache
    try:
        async with _local_lock:
            to_del = [k for k in list(_local_cache.keys()) if k.startswith(prefix)]
            for k in to_del:
                _local_cache.pop(k, None)
            deleted += len(to_del)
    except Exception as e:
        logger.warning(f"cache_delete_prefix (local) failed: {e}")
    return deleted

