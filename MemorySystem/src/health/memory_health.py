from loguru import logger
import os


async def health_check(memory_manager=None, orchestrator=None) -> dict:
    ok = True
    details = {}
    try:
        details["memory_manager"] = bool(memory_manager is not None)
        details["orchestrator"] = bool(orchestrator is not None)

        # Redis
        try:
            # Redis removed - using SQLite as main cache
            # from ..cache.redis_client import get_redis_client
            # redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            # redis_client = await get_redis_client(redis_url)
            details["redis"] = False  # Redis disabled
        except Exception as e:
            logger.error(f"health_check redis error: {e}")
            details["redis"] = False

        # ChromaDB: не создаем новый клиент, доверяем инициализации менеджеров
        try:
            # Если оркестратор инициализирован, его менеджеры уже подняли Chroma-клиенты
            if orchestrator is not None:
                details["chroma"] = True
            else:
                # Пытаемся использовать векторный сторедж из memory_manager (если есть)
                if memory_manager is not None and hasattr(memory_manager, "vector_store"):
                    vs = getattr(memory_manager, "vector_store")
                    if hasattr(vs, "health_check"):
                        details["chroma"] = bool(await vs.health_check())
                    else:
                        details["chroma"] = True
                else:
                    details["chroma"] = False
        except Exception as e:
            logger.error(f"health_check chroma error: {e}")
            details["chroma"] = False

        # Ollama
        try:
            import httpx
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{base_url}/api/tags")
                details["ollama"] = resp.status_code == 200
        except Exception as e:
            logger.error(f"health_check ollama error: {e}")
            details["ollama"] = False

        # overall = True только если основные компоненты готовы; Redis и memory_manager опциональны
        # memory_manager отключен при DISABLE_LEGACY=1
        memory_manager_required = os.getenv("DISABLE_LEGACY", "0") not in ("1", "true", "True")
        
        ok = (
            details.get("orchestrator", False)
            and details.get("chroma", False)
            and details.get("ollama", False)
            and (not memory_manager_required or details.get("memory_manager", False))
        )
    except Exception as e:
        logger.error(f"health_check error: {e}")
        ok = False
    return {"overall": ok, **details}

