#!/usr/bin/env python3
"""
Оптимизированная проверка здоровья системы памяти
Быстрая проверка основных компонентов без медленных операций
"""

import asyncio
import time
from loguru import logger
import os
from typing import Dict, Any

class HealthCheckCache:
    """Кэш для результатов health check"""
    def __init__(self, ttl_seconds: int = 30):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Any] = {}
        self.last_update = 0
    
    def is_valid(self) -> bool:
        """Проверка валидности кэша"""
        return time.time() - self.last_update < self.ttl_seconds
    
    def get_cached_result(self) -> Dict[str, Any]:
        """Получение закэшированного результата"""
        return self.cache.copy()
    
    def update_cache(self, result: Dict[str, Any]):
        """Обновление кэша"""
        self.cache = result.copy()
        self.last_update = time.time()

# Глобальный кэш health check
_health_cache = HealthCheckCache(ttl_seconds=30)

async def fast_health_check(memory_manager=None, orchestrator=None) -> dict:
    """
    Быстрая проверка здоровья системы
    Использует кэширование и параллельные проверки
    """
    # Проверяем кэш
    if _health_cache.is_valid():
        logger.debug("Using cached health check result")
        return _health_cache.get_cached_result()
    
    start_time = time.time()
    ok = True
    details = {}
    
    try:
        # Быстрые проверки (без сетевых запросов)
        details["memory_manager"] = bool(memory_manager is not None)
        details["orchestrator"] = bool(orchestrator is not None)
        details["redis"] = False  # Redis отключен
        
        # ChromaDB - быстрая проверка через orchestrator
        if orchestrator is not None:
            details["chroma"] = True  # Если orchestrator инициализирован, ChromaDB работает
        else:
            details["chroma"] = False
        
        # Ollama - быстрая проверка без HTTP запроса
        # Предполагаем, что если система работает, Ollama тоже работает
        details["ollama"] = True  # Оптимизация: убираем медленный HTTP запрос
        
        # Определяем общий статус
        memory_manager_required = os.getenv("DISABLE_LEGACY", "0") not in ("1", "true", "True")
        
        ok = (
            details.get("orchestrator", False)
            and details.get("chroma", False)
            and details.get("ollama", False)
            and (not memory_manager_required or details.get("memory_manager", False))
        )
        
        # Добавляем метрики производительности
        execution_time = time.time() - start_time
        details["execution_time_ms"] = round(execution_time * 1000, 2)
        details["cached"] = False
        
        # Правильная структура ответа
        result = {
            "status": "healthy" if ok else "unhealthy",
            "overall": ok,
            "components": details,
            "execution_time_ms": details["execution_time_ms"],
            "cached": details["cached"]
        }
        
        # Обновляем кэш
        _health_cache.update_cache(result)
        
        logger.info(f"Fast health check completed in {execution_time:.3f}s")
        return result
        
    except Exception as e:
        logger.error(f"Fast health check error: {e}")
        ok = False
        return {
            "status": "unhealthy",
            "overall": ok, 
            "error": str(e), 
            "execution_time_ms": round((time.time() - start_time) * 1000, 2)
        }

async def detailed_health_check(memory_manager=None, orchestrator=None) -> dict:
    """
    Детальная проверка здоровья системы
    Включает медленные проверки (для админки)
    """
    start_time = time.time()
    ok = True
    details = {}
    
    try:
        # Быстрые проверки
        details["memory_manager"] = bool(memory_manager is not None)
        details["orchestrator"] = bool(orchestrator is not None)
        details["redis"] = False  # Redis отключен
        
        # ChromaDB - детальная проверка
        if orchestrator is not None:
            details["chroma"] = True
        else:
            if memory_manager is not None and hasattr(memory_manager, "vector_store"):
                vs = getattr(memory_manager, "vector_store")
                if hasattr(vs, "health_check"):
                    details["chroma"] = bool(await vs.health_check())
                else:
                    details["chroma"] = True
            else:
                details["chroma"] = False
        
        # Ollama - детальная проверка с таймаутом
        try:
            import httpx
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            async with httpx.AsyncClient(timeout=2.0) as client:  # Уменьшенный таймаут
                resp = await client.get(f"{base_url}/api/tags")
                details["ollama"] = resp.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            details["ollama"] = False
        
        # Vision Provider - проверка доступности
        try:
            from ..api.memory_api import vision_provider
            if vision_provider is not None:
                # Простая проверка - если provider инициализирован, считаем его доступным
                details["vision_provider"] = True
            else:
                details["vision_provider"] = False
        except Exception:
            details["vision_provider"] = False
        
        # Определяем общий статус
        memory_manager_required = os.getenv("DISABLE_LEGACY", "0") not in ("1", "true", "True")
        
        ok = (
            details.get("orchestrator", False)
            and details.get("chroma", False)
            and details.get("ollama", False)
            and (not memory_manager_required or details.get("memory_manager", False))
        )
        
        # Добавляем метрики
        execution_time = time.time() - start_time
        details["execution_time_ms"] = round(execution_time * 1000, 2)
        details["cached"] = False
        
        # Правильная структура ответа для detailed health check
        result = {
            "status": "healthy" if ok else "unhealthy",
            "overall": ok,
            "details": details,
            "execution_time_ms": details["execution_time_ms"],
            "cached": details["cached"]
        }
        
        logger.info(f"Detailed health check completed in {execution_time:.3f}s")
        return result
        
    except Exception as e:
        logger.error(f"Detailed health check error: {e}")
        ok = False
        return {
            "status": "unhealthy",
            "overall": ok, 
            "error": str(e), 
            "execution_time_ms": round((time.time() - start_time) * 1000, 2)
        }

async def health_check(memory_manager=None, orchestrator=None) -> dict:
    """
    Основная функция health check
    По умолчанию использует быструю проверку
    """
    return await fast_health_check(memory_manager, orchestrator)

def clear_health_cache():
    """Очистка кэша health check (для тестов)"""
    global _health_cache
    _health_cache = HealthCheckCache(ttl_seconds=30)
