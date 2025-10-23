# Alias endpoints per plan (declared after app creation)
def _register_alias_endpoints(app):
    @app.post("/memories/multilevel", dependencies=[Depends(require_api_key)])
    async def add_multilevel_memory(request: MultiLevelMemoryRequest):
        try:
            if not memory_orchestrator:
                raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
            # Приоритет: корневое поле importance > metadata.importance > 0.5
            if hasattr(request, 'importance') and request.importance is not None:
                importance = float(request.importance)
                # Валидация корневого поля
                if not (0.0 <= importance <= 1.0):
                    raise HTTPException(status_code=400, detail="Importance must be between 0.0 and 1.0")
            elif request.metadata and "importance" in request.metadata:
                importance = float(request.metadata.get("importance", 0.5))
                # Валидация metadata.importance
                if not (0.0 <= importance <= 1.0):
                    raise HTTPException(status_code=400, detail="Importance in metadata must be between 0.0 and 1.0")
            else:
                importance = 0.5
            emotion_data = request.metadata.get("emotion_data") if request.metadata else None
            context = request.metadata.get("context") if request.metadata else None
            participants = request.metadata.get("participants") if request.metadata else []
            location = request.metadata.get("location") if request.metadata else None
            results = await memory_orchestrator.add_memory(
                content=request.content,
                user_id=request.user_id,
                memory_type=request.level.value,
                importance=importance,
                emotion_data=emotion_data,
                context=context,
                participants=participants or [],
                location=location
            )
            return {"success": True, "results": results, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error in add_multilevel_memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/memories/context/{user_id}")
    async def get_memory_context_alias(user_id: str, query: Optional[str] = None, context_type: str = "full"):
        try:
            if not memory_orchestrator:
                raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
            context = await memory_orchestrator.get_memory_context(user_id, context_type)
            return {"success": True, "user_id": user_id, "context": context, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error in get_memory_context_alias: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/memories/consolidate/{user_id}", dependencies=[Depends(require_api_key)])
    async def consolidate_user_memories_alias(user_id: str):
        try:
            if not memory_orchestrator:
                raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
            results = await memory_orchestrator.consolidate_memories(user_id)
            return {"success": True, "moved": results, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error in consolidate_user_memories_alias: {e}")
            raise HTTPException(status_code=500, detail=str(e))
"""
FastAPI сервер для AIRI Memory System
Предоставляет REST API для работы с системой памяти
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import httpx
import base64
import uuid
import uvicorn

from ..core.memory_manager import MemoryManager
from ..emotion_formatter import EmotionFormatter
from .models import EmotionData
from ..emotion_analysis_service import get_emotion_service, EmotionAnalysisService
from ..emotion_combiner import emotion_combiner
from ..memory_levels.memory_orchestrator import MemoryOrchestrator
# MemoryQuery импортируется из memory_orchestrator
try:
    from ..memory_levels.memory_orchestrator import MemoryQuery
except ImportError:
    # Fallback если MemoryQuery не найден
    MemoryQuery = None
from ..database.init_db import ensure_databases_initialized
from ..search import get_hybrid_engine_lazy, get_contextual_engine_lazy, get_graph_engine_lazy, get_auto_entity_extractor_lazy, get_semantic_engine_lazy
from ..ab_testing import ExperimentManager
# Redis removed - using SQLite as main cache
# from ..cache.redis_client import get_redis_client, close_redis_client
from asyncio import create_task, sleep
import contextlib
from ..monitoring.metrics import inc, set_gauge, snapshot as metrics_snapshot, time_block
from ..health.memory_health_optimized import health_check as overall_health
from ..cache.memory_cache import cache_get, cache_set, cache_delete_prefix
from ..providers.lm_studio_provider import LMStudioProvider
from .models import (
    MemoryResult,
    GetMemoryResponse, UpdateMemoryRequest, UpdateMemoryResponse,
    DeleteMemoryResponse, UserMemory,
    SystemStats, HealthCheckResponse,
    CleanupRequest, CleanupResponse,
    ErrorResponse, EmotionEnhancedMemoryRequest, EmotionEnhancedMemoryResponse,
    EmotionAnalysisRequest, EmotionAnalysisResponse, FormattedEmotion,
    MemoryLevel, MultiLevelMemoryRequest, MultiLevelAddRequest, MultiLevelSearchRequest,
    GoalItem, GoalsListResponse, CreateGoalRequest, CreateGoalResponse,
    CreateProactiveGoalRequest, CreateProactiveGoalResponse, ProactiveGoalItem,
    ProactiveGoalsListResponse, DeleteProactiveGoalResponse, ProactiveGoalsStatusResponse,
    ProactiveGoalsResultsResponse, GoalTriggerType, GoalActionType,
    VisionAnalyzeRequest, VisionAnalyzeResponse, IngestAudioRequest, IngestAudioResponse, EmotionLabel,
    ClickRequest, SatisfactionRequest, SearchMetricsRequest
)

# Глобальные экземпляры
memory_manager: Optional[MemoryManager] = None
emotion_formatter: Optional[EmotionFormatter] = None
memory_orchestrator: Optional[MemoryOrchestrator] = None
vision_provider: Optional[LMStudioProvider] = None
experiment_manager: Optional[ExperimentManager] = None

# Создаем FastAPI приложение
app = FastAPI(
    title="AIRI Memory System API",
    description="REST API для системы памяти AIRI с поддержкой ChromaDB и LM Studio",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Настройка кодировки для правильной работы с UTF-8
import json
from fastapi.responses import JSONResponse

class UTF8JSONResponse(JSONResponse):
    """JSONResponse с правильной UTF-8 кодировкой"""
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,  # Важно! Позволяет UTF-8 символы
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

# Переопределяем стандартный JSONResponse
# app.default_response_class = UTF8JSONResponse  # FastAPI не поддерживает этот атрибут

# Настройка CORS из переменных окружения
origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
methods_env = os.getenv("CORS_ALLOW_METHODS", "*")
headers_env = os.getenv("CORS_ALLOW_HEADERS", "*")
creds_env = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() in ("1", "true", "yes")

allow_origins = [o.strip() for o in origins_env.split(",")] if origins_env != "*" else ["*"]
allow_methods = [m.strip() for m in methods_env.split(",")] if methods_env != "*" else ["*"]
allow_headers = [h.strip() for h in headers_env.split(",")] if headers_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=creds_env,
    allow_methods=allow_methods,
    allow_headers=allow_headers,
)

# Опциональная аутентификация по ключу API для write-операций
async def require_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    expected = os.getenv("API_KEY")
    if not expected:
        return True  # не настроено — пропускаем
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# register alias routes (now that app exists)
_register_alias_endpoints(app)

async def get_memory_manager() -> MemoryManager:
    """Dependency для получения менеджера памяти"""
    global memory_manager
    if memory_manager is None:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    return memory_manager

async def get_memory_orchestrator() -> MemoryOrchestrator:
    """Dependency для получения оркестратора памяти"""
    global memory_orchestrator
    if memory_orchestrator is None:
        raise HTTPException(status_code=503, detail="Memory orchestrator not initialized")
    return memory_orchestrator

async def get_emotion_formatter() -> EmotionFormatter:
    """Dependency для получения форматтера эмоций"""
    global emotion_formatter
    if emotion_formatter is None:
        raise HTTPException(status_code=503, detail="Emotion formatter not initialized")
    return emotion_formatter

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global memory_manager, emotion_formatter, memory_orchestrator, vision_provider
    try:
        logger.info("Инициализация AIRI Memory System...")
        # Ensure SQLite DBs exist (respect env AIRI_DATA_DIR/SQLITE_DB)
        base_data_dir = os.getenv("AIRI_DATA_DIR", "./data")
        await ensure_databases_initialized(data_dir=base_data_dir)
        # Connect Redis (optional)
        # Redis removed - using SQLite as main cache
        # _ = await get_redis_client()
        # MemoryManager отключен - используем только многоуровневую память
        memory_manager = None
        emotion_formatter = EmotionFormatter()
        # Initialize Orchestrator if not already
        if memory_orchestrator is None:
            # Orchestrator respects CHROMADB_DIR via env; pass None or resolved path
            chroma_path = os.getenv("CHROMADB_DIR", os.path.join(base_data_dir, "chroma_db"))
            memory_orchestrator = MemoryOrchestrator(chromadb_path=chroma_path)
            logger.info("Memory Orchestrator инициализирован в startup event")
        # Vision provider for multimodal
        global vision_provider
        if vision_provider is None:
            vision_provider = LMStudioProvider("config/lm_studio_config.yaml")
        
        # Initialize Experiment Manager for A/B testing
        global experiment_manager
        if experiment_manager is None:
            experiment_manager = ExperimentManager()
            await experiment_manager.initialize()
            logger.info("Experiment Manager инициализирован для A/B тестирования")
        
        # Автоматическая система восстановления ОТКЛЮЧЕНА
        # if memory_orchestrator and hasattr(memory_orchestrator, 'auto_recovery'):
        #     try:
        #         await memory_orchestrator.start_auto_recovery()
        #         logger.info("Автоматическая система восстановления запущена")
        #     except Exception as e:
        #         logger.error(f"Ошибка запуска системы восстановления: {e}")
        logger.info("Автоматическая система восстановления ОТКЛЮЧЕНА")
        
        logger.info("AIRI Memory System готов к работе")
        logger.info("Форматтер эмоций инициализирован")
        logger.info("MemoryManager отключен - используется только многоуровневая память")
        inc("app_starts")

        # Запускаем периодическую консолидацию каждые 15 минут
        async def periodic_consolidation():
            # Интервал из ENV (секунды) или по умолчанию 15 минут
            interval_seconds = int(os.getenv("MEMORY_AUTOCONSOLIDATE_SEC", str(15 * 60)))
            # Добавляем флаг для graceful shutdown
            consolidation_running = True
            
            while consolidation_running:
                try:
                    if memory_orchestrator:
                        # Собираем известных пользователей и запускаем консолидацию для каждого
                        user_ids = await memory_orchestrator.get_known_user_ids()
                        for uid in user_ids:
                            try:
                                await memory_orchestrator.consolidate_memories(uid)
                            except Exception as e:
                                logger.error(f"Periodic consolidation error for {uid}: {e}")
                    else:
                        logger.warning("Periodic consolidation skipped: orchestrator not initialized")
                except Exception as e:
                    logger.error(f"Periodic consolidation loop error: {e}")
                except asyncio.CancelledError:
                    logger.info("Periodic consolidation cancelled")
                    consolidation_running = False
                    break
                finally:
                    if consolidation_running:
                        await sleep(interval_seconds)

        # Сохраняем task, чтобы корректно останавливать
        global _periodic_consolidation_task
        try:
            _periodic_consolidation_task.cancel()  # type: ignore
        except Exception:
            pass
        _periodic_consolidation_task = create_task(periodic_consolidation())
    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении"""
    global memory_manager, memory_orchestrator
    
    if memory_manager:
        await memory_manager.close()
        logger.info("AIRI Memory System закрыт")
    
    # Автоматическая система восстановления ОТКЛЮЧЕНА
    # if memory_orchestrator and hasattr(memory_orchestrator, 'auto_recovery'):
    #     try:
    #         await memory_orchestrator.stop_auto_recovery()
    #         logger.info("Автоматическая система восстановления остановлена")
    #     except Exception as e:
    #         logger.error(f"Ошибка остановки системы восстановления: {e}")
    logger.info("Автоматическая система восстановления была отключена")
    
    # Корректное завершение MemoryOrchestrator
    if memory_orchestrator:
        await memory_orchestrator.shutdown()
        logger.info("MemoryOrchestrator shutdown complete")

    # Остановить периодическую консолидацию
    global _periodic_consolidation_task
    if '_periodic_consolidation_task' in globals() and _periodic_consolidation_task:
        with contextlib.suppress(Exception):
            _periodic_consolidation_task.cancel()

    # Close Redis if connected
    try:
        # Redis removed - using SQLite as main cache
        # await close_redis_client()
        pass
    except Exception:
        pass
    # Закрыть vision провайдер
    global vision_provider
    if vision_provider:
        with contextlib.suppress(Exception):
            await vision_provider.close()

@app.get("/api/metrics")
async def get_metrics():
    set_gauge("uptime_gauge", 1.0)
    snap = metrics_snapshot()
    # добавим краткий снимок статусов уровней/рекомендаций procedural
    try:
        global memory_orchestrator
        if memory_orchestrator:
            users = list(await memory_orchestrator.get_known_user_ids())[:1]
            if users:
                u = users[0]
                proc_stats = await memory_orchestrator.procedural_memory.get_stats(u)
                snap["procedural_recommendations"] = proc_stats.get("recommendations", [])
            # размеры коллекций уровней
            try:
                snap["level_sizes"] = {
                    "working": int(memory_orchestrator.working_memory.collection.count()),
                    "short_term": int(memory_orchestrator.short_term_memory.collection.count()),
                    "episodic": int(memory_orchestrator.episodic_memory.collection.count()),
                    "semantic": int(memory_orchestrator.semantic_memory.collection.count()),
                    "graph_nodes": int(getattr(memory_orchestrator.graph_memory, 'get_nodes_count', lambda: 0)() if memory_orchestrator.graph_memory else 0),
                    "graph_edges": int(getattr(memory_orchestrator.graph_memory, 'get_edges_count', lambda: 0)() if memory_orchestrator.graph_memory else 0),
                    "procedural": int(memory_orchestrator.procedural_memory.collection.count()),
                }
            except Exception:
                pass
    except Exception:
        pass
    return JSONResponse(content=snap)

@app.get("/status")
async def status_page():
    """Простая HTML-страница со статусом и автообновлением."""
    snap = metrics_snapshot()
    import json
    # Попробуем динамически вычислить размеры коллекций уровней как в /api/metrics
    level_sizes = {}
    try:
        global memory_orchestrator
        if memory_orchestrator:
            try:
                level_sizes = {
                    "working": int(memory_orchestrator.working_memory.collection.count()),
                    "short_term": int(memory_orchestrator.short_term_memory.collection.count()),
                    "episodic": int(memory_orchestrator.episodic_memory.collection.count()),
                    "semantic": int(memory_orchestrator.semantic_memory.collection.count()),
                    "graph_nodes": int(getattr(memory_orchestrator.graph_memory, 'get_nodes_count', lambda: 0)() if memory_orchestrator.graph_memory else 0),
                    "graph_edges": int(getattr(memory_orchestrator.graph_memory, 'get_edges_count', lambda: 0)() if memory_orchestrator.graph_memory else 0),
                    "procedural": int(memory_orchestrator.procedural_memory.collection.count()),
                }
            except Exception:
                level_sizes = {}
    except Exception:
        level_sizes = {}
    
    # Извлекаем recent_events с проверкой типа
    recent_events = snap.get('recent_events')
    if isinstance(recent_events, list):
        recent_events_display = recent_events[-20:]
    else:
        recent_events_display = []
    
    html = f"""
<!DOCTYPE html>
<html lang=\"ru\">
<head>
  <meta charset=\"utf-8\" />
  <meta http-equiv=\"refresh\" content=\"5\" />
  <title>AIRI Memory Status</title>
  <style>body{{font-family:Segoe UI,Arial,sans-serif;margin:20px}} pre{{background:#f5f5f5;padding:12px;overflow:auto}}</style>
  </head>
<body>
  <h3>AIRI Memory System — Status</h3>
  <p>Uptime: {snap.get('uptime_sec', 0):.1f} sec</p>
  <h4>Counters</h4>
  <pre>{json.dumps(snap.get('counters', {}), ensure_ascii=False, indent=2)}</pre>
  <h4>Gauges</h4>
  <pre>{json.dumps(snap.get('gauges', {}), ensure_ascii=False, indent=2)}</pre>
  <h4>Level Sizes</h4>
  <pre>{json.dumps(level_sizes or snap.get('level_sizes', {}), ensure_ascii=False, indent=2)}</pre>
  <h4>Recent Events</h4>
  <pre>{json.dumps(recent_events_display, ensure_ascii=False, indent=2)}</pre>
  <p>Auto-refresh every 5 seconds.</p>
</body>
</html>
"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)

@app.get("/api/status")
async def status_page_alias():
    """Alias для /status, чтобы избежать путаницы с префиксом /api."""
    return await status_page()

@app.get("/api/perf")
async def perf_snapshot():
    """Агрегированная сводка производительности по recent_events и гейджам."""
    try:
        snap = metrics_snapshot()
        events = snap.get("recent_events") or []
        if isinstance(events, list):
            add_durations = [e.get("duration", 0.0) for e in events if e.get("kind") == "add_memory"]
            search_durations = [e.get("duration", 0.0) for e in events if e.get("kind") == "search_memory"]
        else:
            add_durations = []
            search_durations = []
        # усреднение по уровням
        from collections import defaultdict
        lvl_sum = defaultdict(float)
        lvl_cnt = defaultdict(int)
        if isinstance(events, list):
            for e in events:
                if e.get("kind") == "search_memory":
                    lt = e.get("level_times", {}) or {}
                    for k, v in lt.items():
                        try:
                            lvl_sum[k] += float(v)
                            lvl_cnt[k] += 1
                        except Exception:
                            pass
        level_avg = {k: (lvl_sum[k] / lvl_cnt[k]) if lvl_cnt[k] else 0.0 for k in lvl_sum.keys()}
        # p95 per-level
        from collections import defaultdict
        lvl_values = defaultdict(list)
        if isinstance(events, list):
            for e in events:
                if e.get("kind") == "search_memory":
                    lt = e.get("level_times", {}) or {}
                    for k, v in lt.items():
                        try:
                            lvl_values[k].append(float(v))
                        except Exception:
                            pass
        level_p95 = {}
        for k, arr in lvl_values.items():
            if not arr:
                level_p95[k] = 0.0
                continue
            arr_sorted = sorted(arr)
            idx = int(0.95 * (len(arr_sorted) - 1))
            level_p95[k] = arr_sorted[idx]
        return {
            "adds": {
                "count": len(add_durations),
                "avg_sec": (sum(add_durations) / len(add_durations)) if add_durations else 0.0,
                "p95_sec": sorted(add_durations)[int(0.95 * len(add_durations))] if add_durations else 0.0,
            },
            "searches": {
                "count": len(search_durations),
                "avg_sec": (sum(search_durations) / len(search_durations)) if search_durations else 0.0,
                "p95_sec": sorted(search_durations)[int(0.95 * len(search_durations))] if search_durations else 0.0,
                    "level_avg_sec": level_avg,
                    "level_p95_sec": level_p95,
            },
            "gauges": snap.get("gauges", {}),
            "counters": snap.get("counters", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    """Быстрая проверка здоровья системы (оптимизированная)"""
    global memory_manager, memory_orchestrator
    status = await overall_health(memory_manager, memory_orchestrator)
    return JSONResponse(content=status, status_code=200 if status.get("overall") else 503)

@app.get("/api/health/detailed")
async def detailed_health():
    """Детальная проверка здоровья системы (для админки)"""
    global memory_manager, memory_orchestrator
    from ..health.memory_health_optimized import detailed_health_check
    status = await detailed_health_check(memory_manager, memory_orchestrator)
    return JSONResponse(content=status, status_code=200 if status.get("overall") else 503)

@app.post("/api/memory/multi-level/consolidate/{user_id}", dependencies=[Depends(require_api_key)])
async def consolidate_user_memories(user_id: str):
    try:
        start_ts = time.time()
        inc("memory_consolidation_total")
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        results = await memory_orchestrator.consolidate_memories(user_id)
        set_gauge("memory_processing_time_seconds", time.time() - start_ts)
        return {"success": True, "moved": results, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error consolidating memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Глобальный обработчик исключений"""
    logger.error(f"Необработанная ошибка: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.get("/", response_model=Dict[str, str])
async def root():
    """Корневой эндпоинт"""
    return {
        "service": "AIRI Memory System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Проверка здоровья системы - 6-уровневая память (РЕАЛЬНАЯ ПРОВЕРКА)"""
    try:
        global memory_manager, memory_orchestrator
        if memory_orchestrator is None:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        # РЕАЛЬНАЯ ПРОВЕРКА через overall_health
        health_status = await overall_health(memory_manager, memory_orchestrator)
        
        # Преобразуем в формат HealthCheckResponse с правильной структурой
        health_data = {
            "status": health_status.get("status", "unhealthy"),
            "overall": health_status.get("overall", False),
            "llm": health_status.get("components", {}).get("ollama", False),
            "embeddings": health_status.get("components", {}).get("chroma", False),
            "vector_store": health_status.get("components", {}).get("chroma", False),
            "service": "AIRI Memory System",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "details": health_status  # Добавляем детали для отладки
        }
        
        return HealthCheckResponse(**health_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка проверки здоровья: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== ЭНДПОИНТЫ ДЛЯ ЦЕЛЕЙ (Procedural Goals / SQLite) =====

@app.get("/api/goals", response_model=GoalsListResponse)
async def list_goals(user_id: str):
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        goals = await memory_orchestrator.procedural_memory.get_active_goals(user_id, limit=100)
        items = [GoalItem(**g) for g in goals]
        return GoalsListResponse(goals=items)
    except Exception as e:
        logger.error(f"Ошибка получения целей: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/goals", response_model=CreateGoalResponse, dependencies=[Depends(require_api_key)])
async def create_goal(request: CreateGoalRequest):
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        gid = await memory_orchestrator.procedural_memory.add_goal(
            user_id=request.user_id,
            name=request.name,
            description=request.description or "",
            next_run=request.next_run,
        )
        return CreateGoalResponse(id=gid)
    except Exception as e:
        logger.error(f"Ошибка создания цели: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/goals/{goal_id}", response_model=UpdateMemoryResponse, dependencies=[Depends(require_api_key)])
async def delete_goal(goal_id: str, user_id: str):
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        await memory_orchestrator.procedural_memory.update_goal_progress(goal_id, progress=1.0, status="completed")
        _ = await memory_orchestrator.procedural_memory.cleanup_old_memories(user_id)
        return UpdateMemoryResponse(success=True, message="Goal removed")
    except Exception as e:
        logger.error(f"Ошибка удаления цели: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vision/analyze", response_model=VisionAnalyzeResponse)
async def analyze_image(request: VisionAnalyzeRequest):
    try:
        global vision_provider
        if not vision_provider:
            raise HTTPException(status_code=503, detail="Vision provider not initialized")
        # Debug logging and basic validation
        img_len = len(request.image_b64 or "")
        logger.info(f"/api/vision/analyze received, image_b64_len={img_len}")
        if img_len == 0:
            raise HTTPException(status_code=400, detail="Empty image_b64")
        if img_len > 12 * 1024 * 1024:
            logger.warning("image_b64 too large, rejecting >12MB")
            raise HTTPException(status_code=413, detail="Image too large")
        start_ts = time.time()
        logger.info("Calling vision provider...")
        result = await vision_provider.generate_multimodal(
            prompt=request.prompt,
            images_b64=[request.image_b64],
            task_type="image_analysis"
        )
        logger.info(f"Vision provider returned in {time.time()-start_ts:.2f}s")
        logger.info(f"Vision response length: {len(result)} characters")
        logger.info(f"Vision response preview: {result[:200]}...")
        return VisionAnalyzeResponse(result=result)
    except HTTPException:
        # Перебрасываем HTTPException без изменений
        raise
    except Exception as e:
        logger.error(f"Ошибка мультимодального анализа: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/edges")
async def list_graph_edges(user_id: str, node_id: Optional[str] = None, limit: int = 50):
    """Список рёбер графа пользователя (SQLite), опционально вокруг конкретного узла."""
    try:
        # Валидация limit
        if limit <= 0:
            raise HTTPException(status_code=422, detail="Limit must be positive")
        
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        items = []
        if node_id:
            # Получаем рёбра для конкретного узла
            node_edges = await memory_orchestrator.graph_memory.get_node_edges(node_id, user_id)
            for edge in node_edges:
                # Получаем информацию о связанном узле
                other_node_id = edge.target_id if edge.source_id == node_id else edge.source_id
                other_node = await memory_orchestrator.graph_memory.get_node(other_node_id, user_id)
                
                items.append({
                    "relation_type": edge.relationship_type,
                    "strength": edge.strength,
                    "from_id": edge.source_id,
                    "to_id": edge.target_id,
                    "other_node_id": other_node_id,
                    "other_node_name": other_node.name if other_node else "Unknown",
                })
            
            # Ограничиваем количество результатов
            items = items[:limit]
        else:
            # Используем правильное хранилище графа через memory_orchestrator
            try:
                # Получаем все узлы пользователя
                nodes = await memory_orchestrator.graph_memory.search_nodes(user_id=user_id, query="", limit=100)
                
                # Для каждого узла получаем его рёбра
                for node in nodes:
                    node_edges = await memory_orchestrator.graph_memory.get_node_edges(node.id, user_id)
                    for edge in node_edges:
                        items.append({
                            "relation_type": edge.relationship_type,
                            "strength": edge.strength,
                            "from_id": edge.source_id,
                            "to_id": edge.target_id,
                        })
                
                # Ограничиваем количество результатов
                items = items[:limit]
                
            except Exception as e:
                logger.warning(f"Не удалось получить рёбра через graph_memory: {e}")
                items = []
        return {"user_id": user_id, "count": len(items), "edges": items}
    except HTTPException:
        # Перебрасываем HTTPException без изменений
        raise
    except Exception as e:
        logger.error(f"Ошибка получения рёбер графа: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vision/analyze-multipart", response_model=VisionAnalyzeResponse)
async def analyze_image_multipart(
    file: UploadFile = File(...),
    prompt: str = Form("Опиши изображение подробно на русском")
):
    try:
        global vision_provider
        if not vision_provider:
            raise HTTPException(status_code=503, detail="Vision provider not initialized")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        import base64
        b64 = base64.b64encode(content).decode("ascii")
        logger.info(f"/api/vision/analyze-multipart received, size={len(content)} bytes")
        result = await vision_provider.generate_multimodal(
            prompt=prompt,
            images_b64=[b64],
            task_type="image_analysis"
        )
        logger.info(f"Vision multipart response length: {len(result)} characters")
        logger.info(f"Vision multipart response preview: {result[:200]}...")
        return VisionAnalyzeResponse(result=result)
    except HTTPException:
        # Перебрасываем HTTPException без изменений
        raise
    except Exception as e:
        logger.error(f"Ошибка мультимодального анализа (multipart): {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/audio", response_model=IngestAudioResponse, dependencies=[Depends(require_api_key)])
async def ingest_audio(request: IngestAudioRequest):
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")

        # Обязательная валидация входа: нужен audio_b64
        audio_b64_len = len(request.audio_b64 or "") if isinstance(request.audio_b64, str) else 0
        logger.info(f"/api/ingest/audio received: user={request.user_id}, audio_b64_len={audio_b64_len}")
        if audio_b64_len == 0:
            raise HTTPException(status_code=400, detail="audio_b64 is required and must be a non-empty base64 string")

        svc = get_emotion_service()

        transcribed_text = ""
        if request.metadata and isinstance(request.metadata, dict):
            transcribed_text = str(request.metadata.get("transcribed_text", ""))

        # Если текста нет, пытаемся транскрибировать через WhisperCPP (8002)
        if not transcribed_text:
            try:
                whisper_base = os.getenv("WHISPER_BASE_URL", "http://127.0.0.1:8002")
                async with httpx.AsyncClient(base_url=whisper_base, timeout=30) as ac:
                    # Сначала пробуем JSON маршрут /transcribe
                    resp = await ac.post("/transcribe", json={"audio_b64": request.audio_b64})
                    if resp.status_code == 200:
                        transcribed_text = (resp.json() or {}).get("text", "")
                    else:
                        # fallback на multipart /transcribe/file
                        audio_bytes = base64.b64decode(request.audio_b64)
                        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
                        resp2 = await ac.post("/transcribe/file", files=files)
                        if resp2.status_code == 200:
                            transcribed_text = (resp2.json() or {}).get("text", "")
            except Exception as e:
                logger.warning(f"WhisperCPP transcription failed: {e}")

        # Анализ голоса (audio_b64 гарантирован после валидации)
        voice_res = await svc.analyze_voice_emotions(audio_b64=request.audio_b64)
        text_res = await svc._analyze_text_with_aniemore(transcribed_text) if transcribed_text else None
        dost_res = await svc._analyze_with_dostoevsky(transcribed_text) if transcribed_text else None

        combined = svc._combine_emotion_results(text_res, voice_res, dost_res, validation_applied=True)

        # Build per-source top3 with categories using real helper (FakeEmotionService may not expose privates)
        _helper = EmotionAnalysisService()
        voice_top = _helper._normalize_top3(voice_res or {})
        text_top = _helper._normalize_top3(text_res or {})
        merged_top = []
        # merged: use combined primary/secondary/tertiary
        merged_seq = [
            (combined.primary_emotion, combined.primary_confidence),
            (combined.secondary_emotion, combined.secondary_confidence),
            (combined.tertiary_emotion, combined.tertiary_confidence),
        ]
        def _to_label_list(seq):
            out = []
            for lbl, cf in seq:
                if lbl is None:
                    continue
                try:
                    out.append(EmotionLabel(label=lbl, confidence=float(cf or 0.0), category=_helper._label_category(lbl)))
                except Exception:
                    out.append(EmotionLabel(label=str(lbl), confidence=float(cf or 0.0)))
            return out
        merged_top = _to_label_list(merged_seq)
        voice_top3 = _to_label_list(voice_top[:3])
        text_top3 = _to_label_list(text_top[:3])
        emotions = _to_label_list([(combined.primary_emotion, combined.primary_confidence), (combined.secondary_emotion, combined.secondary_confidence)])
        # Conflict and guidance using helper
        conflict, guidance = _helper._build_conflict_and_guidance(voice_top, text_top)

        results = await memory_orchestrator.add_memory(
            content=transcribed_text or "",
            user_id=request.user_id,
            memory_type="conversation",
            importance=0.6,
            emotion_data={
                "top3": [e.dict() for e in emotions],
                "sentiment": combined.sentiment,
                "dominant_source": combined.dominant_source,
                "consistency": combined.consistency,
                "merged_top3": [e.dict() for e in merged_top],
                "voice_top3": [e.dict() for e in voice_top3],
                "text_top3": [e.dict() for e in text_top3],
                "conflict": conflict,
                "guidance": guidance,
            },
            context=(request.metadata or {}).get("context"),
        )

        return IngestAudioResponse(
            text=transcribed_text or "",
            emotions=emotions,
            sentiment=combined.sentiment,
            dominant_source=combined.dominant_source,
            consistency=combined.consistency,
            memory={"results": results},
            merged_top3=merged_top,
            voice_top3=voice_top3,
            text_top3=text_top3,
            conflict=conflict,
            guidance=guidance,
        )
    except HTTPException as he:
        # Пробрасываем клиентские ошибки как есть (например, 400)
        raise he
    except Exception as e:
        logger.error(f"Ошибка ingest_audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/audio-multipart", response_model=IngestAudioResponse, dependencies=[Depends(require_api_key)])
async def ingest_audio_multipart(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    transcribed_text: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
):
    """Альтернативный прием аудио как multipart/form-data.
    Поля: user_id, file (audio/wav), transcribed_text (опц.), context (опц.).
    """
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file content")
        audio_b64 = base64.b64encode(content).decode("ascii")

        svc = get_emotion_service()

        # Если текста нет, пытаемся транскрибировать через WhisperCPP (8002)
        if not transcribed_text:
            try:
                whisper_base = os.getenv("WHISPER_BASE_URL", "http://127.0.0.1:8002")
                async with httpx.AsyncClient(base_url=whisper_base, timeout=30) as ac:
                    # Сначала пробуем JSON маршрут /transcribe
                    resp = await ac.post("/transcribe", json={"audio_data": audio_b64})
                    if resp.status_code == 200:
                        transcribed_text = (resp.json() or {}).get("text", "")
                    else:
                        # fallback на multipart /transcribe/file
                        files = {"file": ("audio.wav", content, "audio/wav")}
                        resp2 = await ac.post("/transcribe/file", files=files)
                        if resp2.status_code == 200:
                            transcribed_text = (resp2.json() or {}).get("text", "")
            except Exception as e:
                logger.warning(f"WhisperCPP transcription failed (multipart): {e}")

        voice_res = await svc.analyze_voice_emotions(audio_b64=audio_b64)
        text_res = await svc._analyze_text_with_aniemore(transcribed_text) if transcribed_text else None
        dost_res = await svc._analyze_with_dostoevsky(transcribed_text) if transcribed_text else None

        combined = svc._combine_emotion_results(text_res, voice_res, dost_res, validation_applied=True)

        emotions: list[EmotionLabel] = []
        for label, conf in [
            (combined.primary_emotion, combined.primary_confidence),
            (combined.secondary_emotion, combined.secondary_confidence),
            (combined.tertiary_emotion, combined.tertiary_confidence),
        ]:
            if label is None:
                continue
            emotions.append(EmotionLabel(label=label, confidence=float(conf or 0.0)))

        results = await memory_orchestrator.add_memory(
            content=transcribed_text or "",
            user_id=user_id,
            memory_type="conversation",
            importance=0.6,
            emotion_data={
                "top3": [e.dict() for e in emotions],
                "sentiment": combined.sentiment,
                "dominant_source": combined.dominant_source,
                "consistency": combined.consistency,
            },
            context=context,
        )

        return IngestAudioResponse(
            text=transcribed_text or "",
            emotions=emotions,
            sentiment=combined.sentiment,
            dominant_source=combined.dominant_source,
            consistency=combined.consistency,
            memory={"results": results},
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Ошибка ingest_audio_multipart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/hybrid-search")
async def hybrid_search_memories(
    query: str,
    user_id: str,
    limit: int = 10,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    memory_levels: Optional[List[str]] = None  # Стандартизировано на memory_levels
):
    """
    Гибридный поиск воспоминаний (семантический + ключевой)
    
    Args:
        query: Поисковый запрос
        user_id: ID пользователя
        limit: Максимальное количество результатов
        semantic_weight: Вес семантического поиска (0.0-1.0)
        keyword_weight: Вес ключевого поиска (0.0-1.0)
        memory_types: Фильтр по типам памяти (optional)
    """
    try:
        logger.info(f"Hybrid search request: query='{query}', user_id='{user_id}', "
                   f"semantic_weight={semantic_weight}, keyword_weight={keyword_weight}")
        
        # Получаем гибридный поисковый движок с таймаутом
        try:
            get_hybrid_engine = get_hybrid_engine_lazy()
            hybrid_engine = await asyncio.wait_for(
                get_hybrid_engine(memory_orchestrator), 
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.error("Hybrid engine initialization timeout")
            raise HTTPException(status_code=503, detail="Search engine timeout")
        except Exception as e:
            logger.error(f"Hybrid engine initialization failed: {e}")
            raise HTTPException(status_code=503, detail="Search engine error")
        
        # Выполняем гибридный поиск с таймаутом
        try:
            results = await asyncio.wait_for(
                hybrid_engine.search(
                    query=query,
                    user_id=user_id,
                    limit=limit,
                    semantic_weight=semantic_weight,
                    keyword_weight=keyword_weight,
                    memory_types=memory_levels  # Используем memory_levels
                ),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.error("Hybrid search timeout")
            raise HTTPException(status_code=504, detail="Search timeout")
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise HTTPException(status_code=500, detail="Search error")
        
        logger.info(f"Hybrid search results: found {len(results)} memories")
        
        return {
            "success": True,
            "query": query,
            "user_id": user_id,
            "results": results,
            "search_type": "hybrid",
            "semantic_weight": semantic_weight,
            "keyword_weight": keyword_weight,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка гибридного поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/contextual-search")
async def contextual_search_memories(
    query: str,
    user_id: str,
    limit: int = 10,
    offset: int = 0,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    memory_levels: Optional[List[str]] = None,  # Стандартизировано на memory_levels
    # Эмоциональные фильтры
    primary_emotions: Optional[List[str]] = None,
    secondary_emotions: Optional[List[str]] = None,
    min_emotion_confidence: float = 0.0,
    max_emotion_confidence: float = 1.0,
    sentiment: Optional[str] = None,  # "positive", "negative", "neutral"
    emotion_consistency: Optional[str] = None,  # "high", "medium", "low"
    # Временные фильтры
    start_date: Optional[str] = None,  # ISO format
    end_date: Optional[str] = None,    # ISO format
    days_ago: Optional[int] = None,
    hours_ago: Optional[int] = None,
    # Фильтры важности
    min_importance: float = 0.0,
    max_importance: float = 1.0
):
    """
    Контекстный поиск с фильтрацией по эмоциям, времени и важности
    
    Args:
        query: Поисковый запрос
        user_id: ID пользователя
        limit: Максимальное количество результатов
        offset: Смещение для пагинации
        semantic_weight: Вес семантического поиска (0.0-1.0)
        keyword_weight: Вес ключевого поиска (0.0-1.0)
        memory_types: Фильтр по типам памяти
        # Эмоциональные фильтры
        primary_emotions: Список основных эмоций для фильтрации
        secondary_emotions: Список вторичных эмоций для фильтрации
        min_emotion_confidence: Минимальная уверенность в эмоции
        max_emotion_confidence: Максимальная уверенность в эмоции
        sentiment: Тональность ("positive", "negative", "neutral")
        emotion_consistency: Консистентность эмоций ("high", "medium", "low")
        # Временные фильтры
        start_date: Начальная дата (ISO format)
        end_date: Конечная дата (ISO format)
        days_ago: Количество дней назад
        hours_ago: Количество часов назад
        # Фильтры важности
        min_importance: Минимальная важность
        max_importance: Максимальная важность
        
    Returns:
        Результаты контекстного поиска
    """
    try:
        logger.info(f"Contextual search request: query='{query}', user_id='{user_id}', "
                   f"emotions={primary_emotions}, sentiment={sentiment}")
        
        # Импортируем контекстный поиск
        from ..search.contextual_search import (
            ContextualSearchQuery, EmotionFilter, TimeFilter, ImportanceFilter
        )
        
        # Создаем фильтры
        emotion_filter = None
        if any([primary_emotions, secondary_emotions, sentiment, emotion_consistency]) or \
           min_emotion_confidence > 0.0 or max_emotion_confidence < 1.0:
            emotion_filter = EmotionFilter(
                primary_emotions=primary_emotions,
                secondary_emotions=secondary_emotions,
                min_confidence=min_emotion_confidence,
                max_confidence=max_emotion_confidence,
                sentiment=sentiment,
                consistency=emotion_consistency
            )
        
        time_filter = None
        if any([start_date, end_date, days_ago, hours_ago]):
            start_dt = None
            end_dt = None
            
            if start_date:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if end_date:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            time_filter = TimeFilter(
                start_date=start_dt,
                end_date=end_dt,
                days_ago=days_ago,
                hours_ago=hours_ago
            )
        
        importance_filter = None
        if min_importance > 0.0 or max_importance < 1.0:
            importance_filter = ImportanceFilter(
                min_importance=min_importance,
                max_importance=max_importance
            )
        
        # Создаем контекстный запрос
        contextual_query = ContextualSearchQuery(
            query=query,
            user_id=user_id,
            emotion_filter=emotion_filter,
            time_filter=time_filter,
            importance_filter=importance_filter,
            memory_types=memory_levels,  # Используем memory_levels
            limit=limit,
            offset=offset,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
        
        # Получаем контекстный поисковый движок
        get_contextual_engine = get_contextual_engine_lazy()
        contextual_engine = await get_contextual_engine(memory_orchestrator)
        
        # Выполняем контекстный поиск
        results = await contextual_engine.search(contextual_query)
        
        logger.info(f"Contextual search completed: {len(results)} results")
        
        return {
            "success": True,
            "query": query,
            "user_id": user_id,
            "results": results,
            "search_type": "contextual",
            "filters": {
                "emotion": emotion_filter.__dict__ if emotion_filter else None,
                "time": time_filter.__dict__ if time_filter else None,
                "importance": importance_filter.__dict__ if importance_filter else None
            },
            "semantic_weight": semantic_weight,
            "keyword_weight": keyword_weight,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка контекстного поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/graph-search")
async def graph_search_memories(
    query: str,
    user_id: str,
    limit: int = 10,
    offset: int = 0,
    search_type: str = "simple",  # "simple", "pathfinding", "expansion"
    # Параметры для фильтрации по типу узлов
    node_types: Optional[str] = None,
    # Параметры для pathfinding
    start_node: Optional[str] = None,
    end_node: Optional[str] = None,
    max_depth: int = 3,
    # Параметры для expansion
    expand_related: bool = True,
    max_expansions: int = 5
):
    """
    Графовый поиск по узлам и связям в памяти
    
    Args:
        query: Поисковый запрос
        user_id: ID пользователя
        limit: Максимальное количество результатов
        offset: Смещение для пагинации
        search_type: Тип поиска ("simple", "pathfinding", "expansion")
        start_node: Начальный узел для pathfinding
        end_node: Конечный узел для pathfinding
        max_depth: Максимальная глубина поиска
        expand_related: Расширять ли связанные узлы
        max_expansions: Максимальное количество расширений
        
    Returns:
        Результаты графового поиска
    """
    try:
        logger.info(f"Graph search request: query='{query}', user_id='{user_id}', "
                   f"search_type='{search_type}'")
        
        if not memory_orchestrator:
            logger.error("Memory Orchestrator not initialized")
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        # Импортируем графовый поиск
        from ..search.graph_search import GraphSearchQuery, GraphSearchResult
        
        # Преобразуем node_types из строки в список
        node_types_list = [node_types] if node_types else None
        
        # Создаем графовый запрос
        graph_query = GraphSearchQuery(
            query=query,
            user_id=user_id,
            search_type=search_type,
            limit=limit,
            max_depth=max_depth,
            start_node=start_node,
            end_node=end_node,
            node_types=node_types_list
        )
        
        # Получаем графовый поисковый движок
        get_graph_engine = get_graph_engine_lazy()
        graph_engine = await get_graph_engine(memory_orchestrator.graph_memory)
        
        # Выполняем графовый поиск
        results = await graph_engine.search(graph_query)
        
        # Преобразуем результаты в словари для JSON сериализации
        results_dict = []
        for result in results:
            results_dict.append({
                "id": result.node_id,
                "name": result.node_name,
                "node_type": result.node_type,
                       "relevance_score": getattr(result, 'relevance_score', 1.0 - getattr(result, 'distance', 0.0)),
                "path_length": result.path_length,
                "path_nodes": result.path_nodes,
                "relationship_strength": result.relationship_strength,
                "search_type": result.search_type
            })
        
        logger.info(f"Graph search completed: {len(results_dict)} results")
        
        return {
            "success": True,
            "query": query,
            "user_id": user_id,
            "results": results_dict,
            "search_type": "graph",
            "graph_search_type": search_type,
            "parameters": {
                "start_node": start_node,
                "end_node": end_node,
                "max_depth": max_depth,
                "expand_related": expand_related,
                "max_expansions": max_expansions
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка графового поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/integrated-search")
async def integrated_search_memories(
    body: Dict[str, Any]
):
    """
    Интегрированный поиск, объединяющий семантический и графовый поиск
    
    Args:
        body: {
            "query": "поисковый запрос",
            "user_id": "ID пользователя",
            "search_types": ["semantic", "graph", "hybrid"],  # опционально
            "limit": 10,
            "semantic_weight": 0.6,  # вес семантического поиска
            "graph_weight": 0.4,     # вес графового поиска
            "min_confidence": 0.25,
            "min_importance": 0.2,
            "expand_graph": true,
            "max_graph_depth": 2,
            "use_hybrid_ranking": true
        }
        
    Returns:
        Результаты интегрированного поиска
    """
    try:
        query = body.get("query", "")
        user_id = body.get("user_id", "")
        search_types = body.get("search_types", ["semantic", "graph", "hybrid"])
        limit = body.get("limit", 10)
        semantic_weight = body.get("semantic_weight", 0.6)
        graph_weight = body.get("graph_weight", 0.4)
        min_confidence = body.get("min_confidence", 0.25)
        min_importance = body.get("min_importance", 0.2)
        expand_graph = body.get("expand_graph", True)
        max_graph_depth = body.get("max_graph_depth", 2)
        use_hybrid_ranking = body.get("use_hybrid_ranking", True)
        
        logger.info(f"Integrated search request: query='{query}', user_id='{user_id}', "
                   f"search_types={search_types}, limit={limit}")
        
        if not memory_orchestrator:
            logger.error("Memory Orchestrator not initialized")
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        # Импортируем интегрированный поиск
        from ..search.integrated_search_engine import IntegratedSearchQuery, get_integrated_engine
        
        # Создаем запрос
        integrated_query = IntegratedSearchQuery(
            query=query,
            user_id=user_id,
            search_types=search_types,
            limit=limit,
            semantic_weight=semantic_weight,
            graph_weight=graph_weight,
            min_confidence=min_confidence,
            min_importance=min_importance,
            expand_graph=expand_graph,
            max_graph_depth=max_graph_depth,
            use_hybrid_ranking=use_hybrid_ranking
        )
        
        # Получаем интегрированный движок
        integrated_engine = await get_integrated_engine(
            semantic_manager=memory_orchestrator.semantic_memory,
            graph_manager=memory_orchestrator.graph_memory
        )
        
        # Выполняем интегрированный поиск
        results = await integrated_engine.search(integrated_query)
        
        # Преобразуем результаты в словари для JSON сериализации
        results_dict = []
        for result in results:
            result_dict = {
                "id": result.id,
                "content": result.content,
                "source_type": result.source_type,
                       "relevance_score": getattr(result, 'relevance_score', 1.0 - getattr(result, 'distance', 0.0)),
                "semantic_score": result.semantic_score,
                "graph_score": result.graph_score,
                "confidence": result.confidence,
                "importance": result.importance,
                "metadata": result.metadata or {}
            }
            
            # Добавляем графовую информацию если есть
            if result.node_id:
                result_dict.update({
                    "node_id": result.node_id,
                    "node_name": result.node_name,
                    "node_type": result.node_type,
                    "path_length": result.path_length,
                    "path_nodes": result.path_nodes or [],
                    "relationship_strength": result.relationship_strength
                })
            
            # Добавляем семантическую информацию если есть
            if result.knowledge_type:
                result_dict.update({
                    "knowledge_type": result.knowledge_type,
                    "category": result.category
                })
            
            results_dict.append(result_dict)
        
        logger.info(f"Integrated search completed: {len(results_dict)} results")
        
        return {
            "success": True,
            "query": query,
            "user_id": user_id,
            "results": results_dict,
            "search_type": "integrated",
            "search_types_used": search_types,
            "parameters": {
                "semantic_weight": semantic_weight,
                "graph_weight": graph_weight,
                "min_confidence": min_confidence,
                "min_importance": min_importance,
                "expand_graph": expand_graph,
                "max_graph_depth": max_graph_depth,
                "use_hybrid_ranking": use_hybrid_ranking
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка интегрированного поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/search-suggestions")
async def get_search_suggestions(
    query: str,
    user_id: str,
    limit: int = 5
):
    """
    Получить предложения для поиска на основе графа и семантической памяти
    
    Args:
        query: Частичный поисковый запрос
        user_id: ID пользователя
        limit: Максимальное количество предложений
        
    Returns:
        Список предложений для поиска
    """
    try:
        logger.info(f"Search suggestions request: query='{query}', user_id='{user_id}', limit={limit}")
        
        if not memory_orchestrator:
            logger.error("Memory Orchestrator not initialized")
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        # Импортируем интегрированный поиск
        from ..search.integrated_search_engine import get_integrated_engine
        
        # Получаем интегрированный движок
        integrated_engine = await get_integrated_engine(
            semantic_manager=memory_orchestrator.semantic_memory,
            graph_manager=memory_orchestrator.graph_memory
        )
        
        # Получаем предложения
        suggestions = await integrated_engine.get_search_suggestions(query, user_id, limit)
        
        logger.info(f"Search suggestions completed: {len(suggestions)} suggestions")
        
        return {
            "success": True,
            "query": query,
            "user_id": user_id,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения предложений поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/search-metrics")
async def get_search_metrics():
    """
    Получить метрики поиска
    
    Returns:
        Сводка метрик поиска включая производительность и качество
    """
    try:
        from ..monitoring.search_metrics import get_search_metrics_collector
        from ..monitoring.performance_tracker import get_performance_tracker
        
        collector = get_search_metrics_collector()
        metrics_summary = collector.get_search_metrics_summary()
        
        return {
            "success": True,
            "metrics": metrics_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения метрик поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/search-metrics/top-queries")
async def get_top_search_queries(limit: int = 10):
    """
    Получить топ поисковых запросов по частоте
    
    Args:
        limit: Максимальное количество запросов (по умолчанию 10)
        
    Returns:
        Список самых популярных поисковых запросов
    """
    # Валидация параметров (ДО try-except)
    if limit <= 0:
        raise HTTPException(status_code=422, detail="Limit must be positive")
    
    try:
        from ..monitoring.search_metrics import get_search_metrics_collector
        
        collector = get_search_metrics_collector()
        top_queries = collector.get_top_queries(limit)
        
        return {
            "success": True,
            "top_queries": top_queries,
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения топ запросов: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/search-metrics/alerts")
async def get_search_alerts():
    """
    Получить алерты по производительности и качеству поиска
    
    Returns:
        Список алертов по производительности и качеству
    """
    try:
        from ..monitoring.search_metrics import get_search_metrics_collector
        
        collector = get_search_metrics_collector()
        performance_alerts = collector.get_performance_alerts()
        quality_alerts = collector.get_quality_alerts()
        
        return {
            "success": True,
            "alerts": {
                "performance": performance_alerts,
                "quality": quality_alerts,
                "total": len(performance_alerts) + len(quality_alerts)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения алертов поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/search-metrics/{search_type}")
async def get_search_type_metrics(search_type: str):
    """
    Получить метрики для конкретного типа поиска
    
    Args:
        search_type: Тип поиска (semantic, graph, hybrid, integrated, contextual)
        
    Returns:
        Метрики для указанного типа поиска
    """
    try:
        from ..monitoring.search_metrics import get_search_metrics_collector
        
        collector = get_search_metrics_collector()
        type_metrics = collector.get_search_type_metrics(search_type)
        
        # Проверяем на ошибки валидации (недопустимые типы)
        if "error" in type_metrics and "Invalid search type" in type_metrics["error"]:
            raise HTTPException(status_code=400, detail=type_metrics["error"])
        
        # Для типов без данных возвращаем 200 с пустыми метриками
        return {
            "success": True,
            "search_type": search_type,
            "metrics": type_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения метрик для типа поиска {search_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/performance")
async def get_performance_metrics():
    """
    Получить метрики производительности системы
    
    Returns:
        Сводка производительности всех операций
    """
    try:
        from ..monitoring.performance_tracker import get_performance_tracker
        
        tracker = get_performance_tracker()
        performance_summary = tracker.get_performance_summary()
        
        return {
            "success": True,
            "performance": performance_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения метрик производительности: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/performance/alerts")
async def get_performance_alerts():
    """
    Получить алерты по производительности
    
    Returns:
        Список алертов по производительности
    """
    try:
        from ..monitoring.performance_tracker import get_performance_tracker
        
        tracker = get_performance_tracker()
        alerts = tracker.get_performance_alerts()
        
        return {
            "success": True,
            "alerts": alerts,
            "total_alerts": len(alerts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения алертов производительности: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/performance/throughput")
async def get_throughput_metrics(time_window_minutes: int = 5):
    """
    Получить метрики пропускной способности
    
    Args:
        time_window_minutes: Временное окно в минутах (по умолчанию 5)
        
    Returns:
        Метрики пропускной способности
    """
    # Валидация параметров (ДО try-except)
    if time_window_minutes <= 0:
        raise HTTPException(status_code=422, detail="Time window must be positive")
    
    try:
        from ..monitoring.performance_tracker import get_performance_tracker
        
        tracker = get_performance_tracker()
        throughput_metrics = tracker.get_throughput_metrics(time_window_minutes)
        
        return {
            "success": True,
            "throughput": throughput_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения метрик пропускной способности: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/performance/accuracy")
async def get_accuracy_metrics():
    """
    Получить метрики точности поиска
    
    Returns:
        Сводка точности поиска по типам
    """
    try:
        from ..monitoring.performance_tracker import get_performance_tracker
        
        tracker = get_performance_tracker()
        performance_summary = tracker.get_performance_summary()
        accuracy_summary = performance_summary.get("accuracy_summary", {})
        
        return {
            "success": True,
            "accuracy": accuracy_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения метрик точности: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/performance/dashboard")
async def get_performance_dashboard():
    """
    Получить полную сводку производительности системы для дашборда
    
    Returns:
        Комплексная сводка всех метрик производительности
    """
    try:
        from ..monitoring.performance_tracker import get_performance_tracker
        from ..monitoring.search_metrics import get_search_metrics_collector
        
        # Получаем трекер производительности
        performance_tracker = get_performance_tracker()
        performance_summary = performance_tracker.get_performance_summary()
        performance_alerts = performance_tracker.get_performance_alerts()
        throughput_metrics = performance_tracker.get_throughput_metrics(5)  # 5 минут
        
        # Получаем метрики поиска
        search_collector = get_search_metrics_collector()
        search_summary = search_collector.get_search_metrics_summary()
        search_alerts = search_collector.get_performance_alerts() + search_collector.get_quality_alerts()
        top_queries = search_collector.get_top_queries(10)
        
        # Получаем общие метрики системы
        try:
            from ..monitoring.metrics import snapshot as metrics_snapshot
            system_metrics = metrics_snapshot()
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            system_metrics = {}
        
        # Создаем сводку для дашборда
        dashboard_data = {
            "system_overview": {
                "total_operations": performance_summary.get("total_operations", 0),
                "operation_types": performance_summary.get("operation_types", []),
                "system_metrics": system_metrics
            },
            "performance": {
                "operation_stats": performance_summary.get("operation_stats", {}),
                "recent_operations": performance_summary.get("recent_operations", []),
                "throughput": throughput_metrics
            },
            "search": {
                "performance_stats": search_summary.get("performance_stats", {}),
                "quality_stats": search_summary.get("quality_stats", {}),
                "recent_searches": search_summary.get("recent_searches", []),
                "top_queries": top_queries
            },
            "alerts": {
                "performance_alerts": performance_alerts,
                "search_alerts": search_alerts,
                "total_alerts": len(performance_alerts) + len(search_alerts)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "dashboard": dashboard_data
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения дашборда производительности: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/deduplication/metrics")
async def get_deduplication_metrics():
    """
    Получить метрики дедупликации
    
    Returns:
        Метрики эффективности дедупликации
    """
    try:
        # Используем глобальный экземпляр memory_orchestrator
        global memory_orchestrator
        
        if not memory_orchestrator or not hasattr(memory_orchestrator, 'consolidator'):
            return {
                "success": False,
                "error": "Memory orchestrator or consolidator not available"
            }
        
        consolidator = memory_orchestrator.consolidator
        metrics = consolidator.get_deduplication_metrics()
        
        return {
            "success": True,
            "deduplication_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения метрик дедупликации: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/deduplication/alerts")
async def get_deduplication_alerts():
    """
    Получить алерты по дедупликации
    
    Returns:
        Алерты по эффективности дедупликации
    """
    try:
        # Используем глобальный экземпляр memory_orchestrator
        global memory_orchestrator
        
        if not memory_orchestrator or not hasattr(memory_orchestrator, 'consolidator'):
            return {
                "success": False,
                "error": "Memory orchestrator or consolidator not available"
            }
        
        consolidator = memory_orchestrator.consolidator
        metrics = consolidator.get_deduplication_metrics()
        
        alerts = []
        
        # Алерт по низкой эффективности кэша
        cache_hit_rate = metrics.get("cache_hit_rate", 0.0)
        if cache_hit_rate < 0.3:  # Менее 30% попаданий в кэш
            alerts.append({
                "type": "low_cache_hit_rate",
                "value": cache_hit_rate,
                "threshold": 0.3,
                "message": f"Cache hit rate is {cache_hit_rate:.1%} (threshold: 30%)"
            })
        
        # Алерт по низкой эффективности дедупликации
        duplication_efficiency = metrics.get("duplication_efficiency", 0.0)
        if duplication_efficiency < 0.5:  # Менее 50% дублей удаляется
            alerts.append({
                "type": "low_duplication_efficiency",
                "value": duplication_efficiency,
                "threshold": 0.5,
                "message": f"Duplication efficiency is {duplication_efficiency:.1%} (threshold: 50%)"
            })
        
        # Алерт по большому размеру кэша
        cache_size = metrics.get("cache_size", 0)
        if cache_size > 10000:  # Более 10000 записей в кэше
            alerts.append({
                "type": "large_cache_size",
                "value": cache_size,
                "threshold": 10000,
                "message": f"Cache size is {cache_size} entries (threshold: 10000)"
            })
        
        return {
            "success": True,
            "alerts": alerts,
            "total_alerts": len(alerts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения алертов дедупликации: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/extract-entities")
async def extract_entities_from_text(
    text: str,
    user_id: str,
    importance: float = 0.5,
    build_graph: bool = True
):
    """
    Автоматическое извлечение сущностей и связей из текста
    """
    try:
        logger.info(f"Entity extraction request: text_length={len(text)}, user_id='{user_id}', "
                   f"importance={importance}, build_graph={build_graph}")
        
        if not memory_orchestrator:
            logger.error("Memory Orchestrator not initialized")
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")

        # Получаем AutoEntityExtractor
        get_auto_extractor = get_auto_entity_extractor_lazy()
        auto_extractor = get_auto_extractor(
            graph_manager=memory_orchestrator.graph_memory,
            memory_orchestrator=memory_orchestrator
        )

        # Извлекаем сущности и строим граф
        result = await auto_extractor.extract_and_build_from_text(
            text=text,
            user_id=user_id,
            importance=importance
        )

        logger.info(f"Entity extraction completed: {result.total_entities} entities, "
                   f"{result.total_relationships} relationships, "
                   f"{result.build_result.nodes_added} nodes added, "
                   f"{result.build_result.edges_added} edges added")

        return {
            "success": result.success,
            "text_length": len(text),
            "user_id": user_id,
            "extraction_result": {
                "entities": [
                    {
                        "name": e.name,
                        "type": e.entity_type,
                        "confidence": e.confidence,
                        "context": e.context
                    } for e in result.extraction_result.entities
                ],
                "relationships": [
                    {
                        "source": r.source_entity,
                        "target": r.target_entity,
                        "type": r.relationship_type,
                        "confidence": r.confidence,
                        "context": r.context
                    } for r in result.extraction_result.relationships
                ],
                "confidence": result.extraction_result.confidence
            },
            "build_result": {
                "nodes_added": result.build_result.nodes_added,
                "edges_added": result.build_result.edges_added,
                "nodes_updated": result.build_result.nodes_updated,
                "edges_updated": result.build_result.edges_updated,
                "errors": result.build_result.errors
            },
            "statistics": {
                "total_entities": result.total_entities,
                "total_relationships": result.total_relationships,
                "processing_time": result.processing_time
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Ошибка извлечения сущностей: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/batch-extract-entities")
async def batch_extract_entities(
    request: dict
):
    """
    Пакетное извлечение сущностей из нескольких текстов
    """
    try:
        texts = request.get("texts", [])
        user_id = request.get("user_id", "")
        importance = request.get("importance", 0.5)
        
        logger.info(f"Batch entity extraction request: {len(texts)} texts, user_id='{user_id}', "
                   f"importance={importance}")

        if not memory_orchestrator:
            logger.error("Memory Orchestrator not initialized")
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")

        # Получаем AutoEntityExtractor
        get_auto_extractor = get_auto_entity_extractor_lazy()
        auto_extractor = get_auto_extractor(
            graph_manager=memory_orchestrator.graph_memory,
            memory_orchestrator=memory_orchestrator
        )

        # Пакетное извлечение
        results = await auto_extractor.batch_extract_and_build(
            texts=texts,
            user_id=user_id,
            importance=importance
        )

        # Статистика
        stats = auto_extractor.get_extraction_stats(results)

        logger.info(f"Batch extraction completed: {stats['total_entities']} entities, "
                   f"{stats['total_relationships']} relationships, "
                   f"{stats['total_nodes_added']} nodes added, "
                   f"{stats['total_edges_added']} edges added")

        return {
            "success": stats['success_rate'] > 0,
            "texts_count": len(texts),
            "user_id": user_id,
            "statistics": stats,
            "results": [
                {
                    "text_index": i,
                    "success": r.success,
                    "entities_count": r.total_entities,
                    "relationships_count": r.total_relationships,
                    "nodes_added": r.build_result.nodes_added,
                    "edges_added": r.build_result.edges_added,
                    "processing_time": r.processing_time
                } for i, r in enumerate(results)
            ],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Ошибка пакетного извлечения сущностей: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/{memory_id}", response_model=GetMemoryResponse)
async def get_memory(
    memory_id: str
):
    """Получение конкретного воспоминания (УСТАРЕЛО - используйте /memories/context/{user_id})"""
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use /memories/context/{user_id} for multi-level memory context."
    )

@app.put("/api/memory/{memory_id}", response_model=UpdateMemoryResponse, dependencies=[Depends(require_api_key)])
async def update_memory(
    memory_id: str,
    request: UpdateMemoryRequest
):
    """Обновление воспоминания (УСТАРЕЛО - используйте /memories/multilevel для добавления новой памяти)"""
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use /memories/multilevel to add new memory instead of updating."
    )

@app.delete("/api/memory/{memory_id}", response_model=DeleteMemoryResponse, dependencies=[Depends(require_api_key)])
async def delete_memory(
    memory_id: str
):
    """Удаление воспоминания (УСТАРЕЛО - многоуровневая память не поддерживает прямое удаление)"""
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Multi-level memory system does not support direct deletion."
    )

@app.get("/api/stats")
async def get_stats(
    user_id: Optional[str] = None,
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Получение статистики системы"""
    try:
        # Используем MemoryOrchestrator
        if user_id:
            stats = await orchestrator.get_memory_stats(user_id)
        else:
            # Возвращаем общую статистику системы с правильной структурой
            stats = {
                "stats": {
                    "vector_store": {"status": "active", "provider": "chroma"},
                    "embeddings": {"status": "active", "provider": "sentence-transformers"},
                    "llm_provider": "ollama",
                    "embedder_provider": "sentence-transformers",
                    "vector_store_provider": "chroma",
                    "user_stats": None
                }
            }
        return stats
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/performance")
async def get_performance_metrics_deprecated():
    """Получение метрик производительности (УСТАРЕЛО - используйте /api/monitoring/performance)"""
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use /api/monitoring/performance for performance metrics."
    )

# === Legacy cleanup endpoints removed - using new orchestrator-based endpoints ===

# ===== ЭНДПОИНТЫ ДЛЯ ЭМОЦИОНАЛЬНЫХ ДАННЫХ =====

@app.post("/api/emotion/analyze-enhanced")
async def analyze_emotions_enhanced(
    text: Optional[str] = None,
    audio_b64: Optional[str] = None,
    include_validation: bool = True
):
    """
    Улучшенный анализ эмоций с логикой из Rust emotion-engine
    
    Args:
        text: Текст для анализа
        audio_b64: Аудио данные в base64
        include_validation: Включать ли валидацию тональности
        
    Returns:
        CombinedEmotionResult с улучшенной логикой комбинирования
    """
    try:
        logger.info("Enhanced emotion analysis requested")
        
        # Получаем сервис анализа эмоций
        emotion_service = get_emotion_service()
        
        # Выполняем улучшенный анализ
        result = await emotion_service.analyze_emotions_enhanced(
            text=text,
            audio_b64=audio_b64,
            include_validation=include_validation
        )
        
        # Преобразуем в словарь для JSON ответа
        response = {
            "success": True,
            "emotion": result.emotion,
            "confidence": result.confidence,
            "source": result.source,
            "secondary_emotion": result.secondary_emotion,
            "secondary_confidence": result.secondary_confidence,
            "consistency": result.consistency,
            "dominant_source": result.dominant_source,
            "metadata": {
                "weights": {
                    "voice": emotion_combiner.VOICE_WEIGHT,
                    "text_aniemore": emotion_combiner.TEXT_ANIEMORE_WEIGHT,
                    "text_dostoevsky": emotion_combiner.TEXT_DOSTOEVSKY_WEIGHT
                },
                "validation_applied": include_validation
            }
        }
        
        logger.info(f"Enhanced emotion analysis completed: {result.emotion} (confidence: {result.confidence:.3f})")
        return response
        
    except Exception as e:
        logger.error(f"Error in enhanced emotion analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/add-emotion", response_model=EmotionEnhancedMemoryResponse, dependencies=[Depends(require_api_key)])
async def add_emotion_enhanced_memory(
    request: EmotionEnhancedMemoryRequest,
    formatter: EmotionFormatter = Depends(get_emotion_formatter)
):
    """Добавление воспоминания с эмоциональными данными"""
    try:
        logger.info(f"Добавление эмоционально-усиленного воспоминания для пользователя {request.user_id}")
        
        # Подготавливаем метаданные
        metadata = request.metadata or {}
        
        # Если есть эмоциональные данные, форматируем их
        formatted_emotion = None
        emotion_enhancement_applied = False
        
        if request.emotion_data:
            # Создаем EmotionData объект с проверкой атрибутов
            emotion_obj = EmotionData(
                primary_emotion=request.emotion_data.primary_emotion,
                primary_confidence=request.emotion_data.primary_confidence,
                secondary_emotion=getattr(request.emotion_data, 'secondary_emotion', None),
                secondary_confidence=getattr(request.emotion_data, 'secondary_confidence', None)
            )
            
            # Форматируем эмоции - создаем совместимый объект
            try:
                # Создаем объект совместимый с emotion_formatter
                from ..emotion_formatter import EmotionData as FormatterEmotionData
                formatter_emotion = FormatterEmotionData(
                    primary_emotion=emotion_obj.primary_emotion,
                    primary_confidence=emotion_obj.primary_confidence,
                    secondary_emotion=emotion_obj.secondary_emotion,
                    secondary_confidence=emotion_obj.secondary_confidence
                )
                formatted_emotion = formatter.format_emotions_for_ai(formatter_emotion)
            except Exception as e:
                logger.warning(f"Failed to format emotions: {e}")
                formatted_emotion = None
            
            # Добавляем эмоциональные данные в метаданные (сериализуем для ChromaDB)
            metadata.update({
                "emotion_data": json.dumps(request.emotion_data.dict(), ensure_ascii=False),
                "formatted_emotion": json.dumps({
                    "emotion_prompt": formatted_emotion.emotion_prompt if formatted_emotion else "",
                    "complexity": formatted_emotion.complexity if formatted_emotion else "low",
                    "context_hints": formatted_emotion.context_hints if formatted_emotion else [],
                    "metadata": formatted_emotion.metadata if formatted_emotion else {}
                }, ensure_ascii=False),
                "emotion_enhancement": True,
                "primary_emotion": request.emotion_data.primary_emotion,
                "primary_confidence": request.emotion_data.primary_confidence,
                "emotion_consistency": getattr(request.emotion_data, 'consistency', 'high'),
                "emotion_source": getattr(request.emotion_data, 'dominant_source', 'voice')
            })
            
            emotion_enhancement_applied = True
            
            logger.info(f"Эмоциональное усиление применено: {formatted_emotion.complexity if formatted_emotion else 'low'}")
        
        # Добавляем воспоминание через orchestrator
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        # Вызываем add_memory с отдельными аргументами
        result = await memory_orchestrator.add_memory(
            content=request.content,
            user_id=request.user_id,
            memory_type="knowledge",  # По умолчанию для эмоциональных воспоминаний
            importance=0.7,  # Высокая важность для эмоциональных воспоминаний
            emotion_data=request.emotion_data.dict() if request.emotion_data else None
        )
        memory_id = result.get("memory_id", "unknown")
        
        # Создаем базовую структуру памяти для ответа
        memory = {
            "content": request.content,
            "metadata": metadata,
            "created_at": datetime.now().isoformat()
        }
        
        # Создаем Pydantic объект FormattedEmotion
        formatted_emotion_response = None
        if formatted_emotion:
            # Создаем совместимый объект EmotionComplexity
            from .models import EmotionComplexity as APIEmotionComplexity
            complexity_value = str(formatted_emotion.complexity)
            if complexity_value in ["simple", "medium", "complex"]:
                complexity = APIEmotionComplexity(complexity_value)
            else:
                complexity = APIEmotionComplexity.SIMPLE
            
            formatted_emotion_response = FormattedEmotion(
                emotion_prompt=formatted_emotion.emotion_prompt,
                complexity=complexity,
                context_hints=formatted_emotion.context_hints,
                metadata=formatted_emotion.metadata
            )
        
        return EmotionEnhancedMemoryResponse(
            id=memory_id,
            content=memory["content"],
            user_id=request.user_id,
            emotion_data=request.emotion_data,
            formatted_emotion=formatted_emotion_response,
            metadata=memory["metadata"],
            created_at=memory["created_at"],
            emotion_enhancement_applied=emotion_enhancement_applied
        )
        
    except Exception as e:
        logger.error(f"Ошибка добавления эмоционально-усиленного воспоминания: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emotion/analyze", response_model=EmotionAnalysisResponse)
async def analyze_emotions(
    request: EmotionAnalysisRequest,
    formatter: EmotionFormatter = Depends(get_emotion_formatter)
):
    """Полноценный анализ эмоций в тексте"""
    try:
        logger.info(f"Полноценный анализ эмоций для пользователя {request.user_id}")
        logger.info(f"Текст для анализа: '{request.text[:100]}{'...' if len(request.text) > 100 else ''}'")
        
        # Получаем сервис анализа эмоций
        emotion_service = get_emotion_service()
        
        # Выполняем полноценный анализ эмоций
        analysis_result = await emotion_service.analyze_text_emotions(
            text=request.text,
            include_validation=request.include_validation
        )
        
        # Создаем EmotionData объект с проверкой атрибутов
        emotion_data = EmotionData(
            primary_emotion=analysis_result.primary_emotion,
            primary_confidence=analysis_result.primary_confidence,
            secondary_emotion=getattr(analysis_result, 'secondary_emotion', None),
            secondary_confidence=getattr(analysis_result, 'secondary_confidence', None)
        )
        
        # Форматируем эмоции - создаем совместимый объект
        try:
            from ..emotion_formatter import EmotionData as FormatterEmotionData
            formatter_emotion = FormatterEmotionData(
                primary_emotion=emotion_data.primary_emotion,
                primary_confidence=emotion_data.primary_confidence,
                secondary_emotion=getattr(emotion_data, 'secondary_emotion', None),
                secondary_confidence=getattr(emotion_data, 'secondary_confidence', None)
            )
            formatted_emotion = formatter.format_emotions_for_ai(formatter_emotion)
        except Exception as e:
            logger.warning(f"Failed to format emotions: {e}")
            formatted_emotion = None
        
        logger.info(f"Анализ эмоций завершен за {analysis_result.analysis_time:.3f}с")
        logger.info(f"Результат: {emotion_data.primary_emotion} ({emotion_data.primary_confidence:.3f})")
        logger.info(f"Согласованность: {getattr(emotion_data, 'consistency', 'high')}")
        logger.info(f"Валидация применена: {getattr(emotion_data, 'validation_applied', False)}")
        
        # Создаем Pydantic объект FormattedEmotion
        # Создаем совместимый объект EmotionComplexity
        from .models import EmotionComplexity as APIEmotionComplexity
        if formatted_emotion:
            complexity_value = str(formatted_emotion.complexity)
            if complexity_value in ["simple", "medium", "complex"]:
                complexity = APIEmotionComplexity(complexity_value)
            else:
                complexity = APIEmotionComplexity.SIMPLE
        else:
            complexity = APIEmotionComplexity.SIMPLE
        
        formatted_emotion_response = FormattedEmotion(
            emotion_prompt=formatted_emotion.emotion_prompt if formatted_emotion else "",
            complexity=complexity,
            context_hints=formatted_emotion.context_hints if formatted_emotion else [],
            metadata=formatted_emotion.metadata if formatted_emotion else {}
        )
        
        return EmotionAnalysisResponse(
            emotion_data=emotion_data,
            formatted_emotion=formatted_emotion_response,
            analysis_time=analysis_result.analysis_time,
            validation_applied=analysis_result.validation_applied
        )
        
    except Exception as e:
        logger.error(f"Ошибка полноценного анализа эмоций: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emotion/analyze-voice", response_model=EmotionAnalysisResponse)
async def analyze_voice_emotions(
    request: dict,  # Используем dict вместо VoiceEmotionAnalysisRequest
    formatter: EmotionFormatter = Depends(get_emotion_formatter)
):
    """Анализ эмоций в голосе"""
    try:
        logger.info(f"Анализ эмоций голоса для пользователя {request.get('user_id', 'unknown')}")

        # Получаем сервис анализа эмоций
        emotion_service = get_emotion_service()
        
        # Выполняем анализ эмоций голоса
        analysis_result = await emotion_service.analyze_voice_emotions(
            audio_b64=request.get('audio_b64', '')
        )

        # Создаем EmotionData объект
        emotion_data = EmotionData(
            primary_emotion=analysis_result.get("emotion", "нейтральная"),
            primary_confidence=analysis_result.get("confidence", 0.5),
            secondary_emotion=None,
            secondary_confidence=None
        )

        # Форматируем эмоции - создаем совместимый объект
        try:
            from ..emotion_formatter import EmotionData as FormatterEmotionData
            formatter_emotion = FormatterEmotionData(
                primary_emotion=emotion_data.primary_emotion,
                primary_confidence=emotion_data.primary_confidence,
                secondary_emotion=getattr(emotion_data, 'secondary_emotion', None),
                secondary_confidence=getattr(emotion_data, 'secondary_confidence', None)
            )
            formatted_emotion = formatter.format_emotions_for_ai(formatter_emotion)
        except Exception as e:
            logger.warning(f"Failed to format emotions: {e}")
            formatted_emotion = None

        logger.info(f"Анализ эмоций голоса завершен: {emotion_data.primary_emotion} ({emotion_data.primary_confidence:.3f})")

        # Создаем Pydantic объект FormattedEmotion
        # Создаем совместимый объект EmotionComplexity
        from .models import EmotionComplexity as APIEmotionComplexity
        if formatted_emotion:
            complexity_value = str(formatted_emotion.complexity)
            if complexity_value in ["simple", "medium", "complex"]:
                complexity = APIEmotionComplexity(complexity_value)
            else:
                complexity = APIEmotionComplexity.SIMPLE
        else:
            complexity = APIEmotionComplexity.SIMPLE
        
        formatted_emotion_response = FormattedEmotion(
            emotion_prompt=formatted_emotion.emotion_prompt if formatted_emotion else "",
            complexity=complexity,
            context_hints=formatted_emotion.context_hints if formatted_emotion else [],
            metadata=formatted_emotion.metadata if formatted_emotion else {}
        )

        return EmotionAnalysisResponse(
            emotion_data=emotion_data,
            formatted_emotion=formatted_emotion_response,
            analysis_time=0.1,  # Заглушка для времени анализа
            validation_applied=False
        )

    except Exception as e:
        logger.error(f"Ошибка анализа эмоций голоса: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/emotion/formatter/stats")
async def get_emotion_formatter_stats(
    formatter: EmotionFormatter = Depends(get_emotion_formatter)
):
    """Получение статистики форматтера эмоций"""
    try:
        stats = formatter.get_formatter_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Ошибка получения статистики форматтера: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/emotion/health")
async def get_emotion_services_health():
    """Проверка здоровья сервисов анализа эмоций"""
    try:
        emotion_service = get_emotion_service()
        health_status = await emotion_service.health_check()
        
        return JSONResponse(content={
            "emotion_services_health": health_status,
            "timestamp": time.time(),
            "status": "healthy" if health_status["overall"] else "degraded"
        })
    except Exception as e:
        logger.error(f"Ошибка проверки здоровья сервисов эмоций: {e}")
        return JSONResponse(content={
            "emotion_services_health": {"overall": False, "error": str(e)},
            "timestamp": time.time(),
            "status": "unhealthy"
        }, status_code=503)

@app.get("/api/emotion/service/stats")
async def get_emotion_service_stats():
    """Получение статистики сервиса анализа эмоций"""
    try:
        emotion_service = get_emotion_service()
        stats = emotion_service.get_service_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Ошибка получения статистики сервиса эмоций: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Инициализация оркестратора памяти
# Removed duplicate startup_event; orchestrator is initialized in the primary startup_event above.

# Новые эндпоинты для многоуровневой памяти

@app.post("/api/memory/multi-level/add", dependencies=[Depends(require_api_key)])
async def add_multi_level_memory(payload: MultiLevelAddRequest):
    """
    Добавить память на все уровни системы
    
    Args:
        content: Содержимое памяти
        user_id: ID пользователя
        memory_type: Тип памяти (conversation, event, knowledge, skill, etc.)
        importance: Важность (0.0-1.0)
        emotion_data: Эмоциональные данные
        context: Контекст
        participants: Участники
        location: Местоположение
        
    Returns:
        Словарь с ID созданных элементов на каждом уровне
    """
    try:
        inc("memory_add_total")
        if not memory_orchestrator:
            raise HTTPException(status_code=500, detail="Memory Orchestrator not initialized")
        
        results = await memory_orchestrator.add_memory(
            content=payload.content,
            user_id=payload.user_id,
            memory_type=payload.level or "working",  # Используем level вместо memory_type
            importance=payload.importance,
            emotion_data=(payload.emotion_data.dict() if payload.emotion_data else None),
            context=payload.context,
            participants=payload.participants or [],
            location=payload.location
        )
        # Инвалидация кэша поиска (новые данные могут повлиять на выдачу)
        try:
            await cache_delete_prefix("ml_search:")
        except Exception:
            pass
        
        return {
            "success": True,
            "message": "Memory added to multiple levels",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error adding multi-level memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/multi-level/search")
async def search_multi_level_memory(payload: MultiLevelSearchRequest):
    """
    Поиск по всем уровням памяти
    
    Args:
        query: Поисковый запрос
        user_id: ID пользователя
        memory_levels: Уровни для поиска (working, short_term, episodic, semantic, graph, procedural)
        limit: Максимальное количество результатов
        include_emotions: Включать эмоциональные данные
        include_relationships: Включать связи
        
    Returns:
        Унифицированный результат поиска
    """
    try:
        inc("memory_search_total")
        if not memory_orchestrator:
            raise HTTPException(status_code=500, detail="Memory Orchestrator not initialized")
        
        # simple cache key
        # Нормализуем и хешируем ключ кэша для устойчивости
        import hashlib
        lvls = ','.join(payload.memory_levels or [])
        key_raw = f"user={payload.user_id}|q={payload.query}|lvls={lvls}|lim={payload.limit}|off={payload.offset}|e={int(payload.include_emotions)}|r={int(payload.include_relationships)}"
        cache_key = "ml_search:" + hashlib.md5(key_raw.encode('utf-8')).hexdigest()
        cached = await cache_get(cache_key)
        if cached:
            inc("memory_cache_hits_total")
            return cached

        if MemoryQuery is None:
            raise HTTPException(status_code=503, detail="MemoryQuery not available")
        
        memory_query = MemoryQuery(
            query=payload.query,
            user_id=payload.user_id,
            memory_levels=payload.memory_levels or ["working", "short_term", "episodic", "semantic"],
            limit=payload.limit,
            offset=payload.offset,
            include_emotions=payload.include_emotions,
            include_relationships=payload.include_relationships
        )
        
        results = await memory_orchestrator.search_memory(memory_query)

        # Build response, optionally including relationships for graph nodes
        response_results = {}
        for level, result in results.results.items():
            items_payload = []
            for item in result.items:
                item_payload = {
                    "id": getattr(item, 'id', None),
                    "content": getattr(item, 'content', getattr(item, 'name', str(item))),
                    "timestamp": str(getattr(item, 'timestamp', None)) if getattr(item, 'timestamp', None) else None,
                    "importance": getattr(item, 'importance', 0.5),
                }
                # Level-specific details so the user can see what's stored/searched per level
                if level == "working":
                    item_payload.update({
                        "context": getattr(item, 'context', None),
                        "emotion_data": getattr(item, 'emotion_data', None) if payload.include_emotions else None,
                    })
                elif level == "short_term":
                    item_payload.update({
                        "event_type": getattr(item, 'event_type', None),
                        "location": getattr(item, 'location', None),
                        "participants": getattr(item, 'participants', None),
                        "emotion_data": getattr(item, 'emotion_data', None) if payload.include_emotions else None,
                    })
                elif level == "episodic":
                    item_payload.update({
                        "event_type": getattr(item, 'event_type', None),
                        "location": getattr(item, 'location', None),
                        "participants": getattr(item, 'participants', None),
                        "significance": getattr(item, 'significance', None),
                        "vividness": getattr(item, 'vividness', None),
                        "context": getattr(item, 'context', None),
                        "emotion_data": getattr(item, 'emotion_data', None) if payload.include_emotions else None,
                    })
                elif level == "semantic":
                    item_payload.update({
                        "knowledge_type": getattr(item, 'knowledge_type', None),
                        "category": getattr(item, 'category', None),
                        "confidence": getattr(item, 'confidence', None),
                        "related_concepts": getattr(item, 'related_concepts', None),
                        "tags": getattr(item, 'tags', None),
                        "last_accessed": str(getattr(item, 'last_accessed', None)) if getattr(item, 'last_accessed', None) else None,
                        "access_count": getattr(item, 'access_count', None),
                    })
                elif level == "procedural":
                    item_payload.update({
                        "name": getattr(item, 'name', getattr(item, 'content', None)),
                        "skill_type": getattr(item, 'skill_type', None),
                        "difficulty": getattr(item, 'difficulty', None),
                        "proficiency": getattr(item, 'proficiency', None),
                        "success_rate": getattr(item, 'success_rate', None),
                        "practice_count": getattr(item, 'practice_count', None),
                        "last_practiced": str(getattr(item, 'last_practiced', None)) if getattr(item, 'last_practiced', None) else None,
                    })
                if payload.include_relationships and level == "graph":
                    try:
                        connected = []
                        if hasattr(memory_orchestrator.graph_memory, 'get_connected_nodes'):
                            try:
                                # Используем getattr для безопасного вызова
                                get_connected_nodes = getattr(memory_orchestrator.graph_memory, 'get_connected_nodes')
                                connected = await get_connected_nodes(
                                    node_id=getattr(item, 'id', None),
                                    user_id=results.user_id,
                                    relationship_type=None,
                                    limit=10,
                                )
                            except Exception as e:
                                logger.warning(f"Failed to get connected nodes: {e}")
                                connected = []
                        item_payload["relationships"] = [
                            {
                                "relation_type": edge.relationship_type,
                                "strength": edge.strength,
                                "node_id": node.id,
                                "node_name": node.name,
                            }
                            for (node, edge) in connected
                        ]
                    except Exception:
                        item_payload["relationships"] = []
                items_payload.append(item_payload)

            response_results[level] = {
                "items": items_payload,
                "relevance_scores": result.relevance_scores,
                "total_found": result.total_found,
            }

        response = {
            "success": True,
            "query": results.query,
            "user_id": results.user_id,
            "results": response_results,
            "total_items": results.total_items,
            "processing_time": results.processing_time,
            "recommendations": results.recommendations,
            "timestamp": datetime.now().isoformat()
        }
        await cache_set(cache_key, response, ttl_sec=120)
        inc("memory_cache_misses_total")
        return response
        
    except Exception as e:
        logger.error(f"Error searching multi-level memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/multi-level/context/{user_id}")
async def get_multi_level_context(
    user_id: str,
    context_type: str = "full"
):
    """
    Получить контекст памяти пользователя
    
    Args:
        user_id: ID пользователя
        context_type: Тип контекста (full, recent, important)
        
    Returns:
        Контекст всех уровней памяти
    """
    try:
        if not memory_orchestrator:
            raise HTTPException(status_code=500, detail="Memory Orchestrator not initialized")
        
        context = await memory_orchestrator.get_memory_context(user_id, context_type)
        
        return {
            "success": True,
            "user_id": user_id,
            "context_type": context_type,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting multi-level context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/multi-level/stats/{user_id}")
async def get_multi_level_stats(user_id: str):
    """
    Получить статистику всех уровней памяти
    
    Args:
        user_id: ID пользователя
        
    Returns:
        Статистика всех уровней памяти
    """
    try:
        if not memory_orchestrator:
            raise HTTPException(status_code=500, detail="Memory Orchestrator not initialized")
        
        stats = await memory_orchestrator.get_memory_stats(user_id)
        
        return {
            "success": True,
            "user_id": user_id,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting multi-level stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/multi-level/cleanup/{user_id}", dependencies=[Depends(require_api_key)])
async def cleanup_multi_level_memories(user_id: str):
    """
    Очистка старых воспоминаний на всех уровнях
    
    Args:
        user_id: ID пользователя
        
    Returns:
        Результат очистки
    """
    try:
        inc("memory_cleanup_total")
        if not memory_orchestrator:
            raise HTTPException(status_code=500, detail="Memory Orchestrator not initialized")
        
        await memory_orchestrator.cleanup_memories(user_id)
        
        return {
            "success": True,
            "message": f"Memories cleaned up for user {user_id}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up multi-level memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === LLM Proxy endpoints (Ollama/LM Studio) ===
@app.get("/api/llm/models", dependencies=[Depends(require_api_key)])
async def list_llm_models():
    try:
        global vision_provider
        prov = vision_provider or LMStudioProvider("config/lm_studio_config.yaml")
        models = await prov.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Ошибка получения списка моделей LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/llm/generate", dependencies=[Depends(require_api_key)])
async def llm_generate(body: Dict[str, Any]):
    try:
        prompt = str(body.get("prompt", ""))
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        task_type = str(body.get("task_type", "text_generation"))
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature")
        emotional_context = body.get("emotional_context")
        global vision_provider
        prov = vision_provider or LMStudioProvider("config/lm_studio_config.yaml")
        text = await prov.generate_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            task_type=task_type,
            emotional_context=emotional_context,
        )
        return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка генерации через прокси: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/llm/vision", dependencies=[Depends(require_api_key)])
async def llm_vision(body: Dict[str, Any]):
    try:
        prompt = str(body.get("prompt", "Опиши изображение подробно на русском"))
        images_b64 = body.get("images_b64") or []
        if not images_b64:
            raise HTTPException(status_code=400, detail="images_b64 is required")
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature")
        global vision_provider
        prov = vision_provider or LMStudioProvider("config/lm_studio_config.yaml")
        result = await prov.generate_multimodal(
            prompt=prompt,
            images_b64=images_b64,
            task_type="image_analysis",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка мультимодальной генерации через прокси: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MEMORY CONSOLIDATION API ENDPOINTS
# ============================================================================

@app.post("/api/memory/consolidate", dependencies=[Depends(require_api_key)])
async def consolidate_memory(user_id: str, level: Optional[str] = None):
    """
    Запустить консолидацию памяти
    
    Args:
        user_id: ID пользователя для консолидации
        level: Конкретный уровень для консолидации (working, short_term, episodic, semantic, graph, procedural)
    """
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        logger.info(f"Memory consolidation requested for user {user_id}, level: {level or 'all'}")
        result = await memory_orchestrator.consolidate_memory(user_id, level)
        
        return {
            "success": result.get("success", False),
            "level": level or "all",
            "processing_time": result.get("processing_time", 0),
            "total_removed": result.get("total_removed", 0),
            "total_consolidated": result.get("total_consolidated", 0),
            "level_results": result.get("level_results", {}),
            "error": result.get("error")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in memory consolidation API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/consolidation/stats")
async def get_consolidation_stats():
    """
    Получить статистику консолидации памяти
    """
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        stats = await memory_orchestrator.get_consolidation_stats()
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting consolidation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/consolidation/schedule", dependencies=[Depends(require_api_key)])
async def schedule_consolidation(interval_hours: int = 24):
    """
    Запланировать периодическую консолидацию памяти
    
    Args:
        interval_hours: Интервал между консолидациями в часах (по умолчанию 24)
    """
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        if interval_hours < 1 or interval_hours > 168:  # От 1 часа до 1 недели
            raise HTTPException(status_code=400, detail="Interval must be between 1 and 168 hours")
        
        # Запускаем планировщик в фоне
        asyncio.create_task(memory_orchestrator.schedule_consolidation("system", interval_hours))
        
        logger.info(f"Scheduled memory consolidation every {interval_hours} hours")
        return {
            "success": True,
            "message": f"Memory consolidation scheduled every {interval_hours} hours",
            "interval_hours": interval_hours
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling consolidation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/cleanup", dependencies=[Depends(require_api_key)])
async def cleanup_old_memories(days: int = 365):
    """
    Очистка старых воспоминаний
    
    Args:
        days: Количество дней для удержания воспоминаний (по умолчанию 365)
    """
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        if days < 1 or days > 3650:  # От 1 дня до 10 лет
            raise HTTPException(status_code=400, detail="Days must be between 1 and 3650")
        
        logger.info(f"Memory cleanup requested for memories older than {days} days")
        result = await memory_orchestrator.cleanup_old_memories(days)
        
        return {
            "success": result.get("success", False),
            "deleted_count": result.get("deleted_count", 0),
            "retention_days": result.get("retention_days", days),
            "error": result.get("error")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in memory cleanup API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== A/B TESTING ENDPOINTS ====================

@app.get("/api/ab-testing/experiments")
async def get_experiments():
    """
    Получить список всех экспериментов
    """
    try:
        if not experiment_manager:
            raise HTTPException(status_code=503, detail="Experiment Manager not initialized")
        
        summary = await experiment_manager.get_all_experiments_summary()
        return {
            "success": True,
            "experiments": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ab-testing/experiment/{experiment_id}")
async def get_experiment(experiment_id: str):
    """
    Получить информацию об эксперименте
    """
    try:
        if not experiment_manager:
            raise HTTPException(status_code=503, detail="Experiment Manager not initialized")
        
        experiment_info = experiment_manager.get_experiment_info(experiment_id)
        if not experiment_info:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        
        dashboard_data = await experiment_manager.get_experiment_dashboard_data(experiment_id)
        
        return {
            "success": True,
            "experiment": experiment_info,
            "dashboard_data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ab-testing/experiment/{experiment_id}/analysis")
async def analyze_experiment(experiment_id: str):
    """
    Анализировать эксперимент
    """
    try:
        if not experiment_manager:
            raise HTTPException(status_code=503, detail="Experiment Manager not initialized")
        
        analysis = await experiment_manager.analyze_experiment(experiment_id)
        if not analysis:
            raise HTTPException(status_code=404, detail=f"No data available for experiment {experiment_id}")
        
        return {
            "success": True,
            "analysis": {
                "experiment_id": analysis.experiment_id,
                "total_users": analysis.total_users,
                "duration_days": analysis.duration_days,
                "winner_variant": analysis.winner_variant,
                "confidence_level": analysis.confidence_level,
                "recommendations": analysis.recommendations,
                "statistical_tests": [
                    {
                        "metric_name": test.metric_name,
                        "variant_a": test.variant_a,
                        "variant_b": test.variant_b,
                        "value_a": test.value_a,
                        "value_b": test.value_b,
                        "p_value": test.p_value,
                        "confidence_interval": test.confidence_interval,
                        "is_significant": test.is_significant,
                        "effect_size": test.effect_size,
                        "test_type": test.test_type
                    }
                    for test in analysis.statistical_tests
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ab-testing/experiment/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """
    Остановить эксперимент
    """
    try:
        if not experiment_manager:
            raise HTTPException(status_code=503, detail="Experiment Manager not initialized")
        
        success = await experiment_manager.stop_experiment(experiment_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        
        return {
            "success": True,
            "message": f"Experiment {experiment_id} stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ab-testing/click")
async def record_click(request: ClickRequest):
    """
    Записать клик по результату поиска
    """
    try:
        if not experiment_manager:
            raise HTTPException(status_code=503, detail="Experiment Manager not initialized")
        
        success = await experiment_manager.record_click(
            user_id=request.user_id,
            experiment_id=request.experiment_id,
            variant_name=request.variant_name,
            query=request.query,
            result_id=request.result_id
        )
        
        return {
            "success": success,
            "message": "Click recorded" if success else "Failed to record click",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error recording click: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ab-testing/satisfaction")
async def record_satisfaction(request: SatisfactionRequest):
    """
    Записать оценку удовлетворенности
    """
    try:
        if not experiment_manager:
            raise HTTPException(status_code=503, detail="Experiment Manager not initialized")
        
        if not (0.0 <= request.satisfaction_score <= 1.0):
            raise HTTPException(status_code=400, detail="Satisfaction score must be between 0.0 and 1.0")
        
        success = await experiment_manager.record_satisfaction(
            user_id=request.user_id,
            experiment_id=request.experiment_id,
            variant_name=request.variant_name,
            query=request.query,
            satisfaction_score=request.satisfaction_score
        )
        
        return {
            "success": success,
            "message": "Satisfaction recorded" if success else "Failed to record satisfaction",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording satisfaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ab-testing/search-metrics")
async def record_search_metrics(request: SearchMetricsRequest):
    """
    Записать метрики поиска
    """
    try:
        if not experiment_manager:
            raise HTTPException(status_code=503, detail="Experiment Manager not initialized")
        
        success = await experiment_manager.record_search_metrics(
            user_id=request.user_id,
            experiment_id=request.experiment_id,
            variant_name=request.variant_name,
            query=request.query,
            response_time_ms=request.response_time_ms,
            results_count=request.results_count
        )
        
        return {
            "success": success,
            "message": "Search metrics recorded" if success else "Failed to record search metrics",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording search metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ab-testing/export/{experiment_id}")
async def export_experiment_data(experiment_id: str):
    """
    Экспортировать данные эксперимента
    """
    try:
        if not experiment_manager:
            raise HTTPException(status_code=503, detail="Experiment Manager not initialized")
        
        # Создаем имя файла с timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{experiment_id}_{timestamp}.json"
        filepath = f"data/exports/{filename}"
        
        # Создаем директорию если не существует
        os.makedirs("data/exports", exist_ok=True)
        
        success = await experiment_manager.export_experiment_data(experiment_id, filepath)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to export experiment data")
        
        return {
            "success": True,
            "message": f"Experiment data exported to {filename}",
            "filename": filename,
            "filepath": filepath,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting experiment data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Настройка логирования
    logger.add(
        "data/logs/memory_api.log",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )
    
# ============================================================================
# СЕМАНТИЧЕСКИЕ API ENDPOINTS
# ============================================================================

@app.get("/api/memory/semantic/search")
async def semantic_search_memories(
    query: str,
    user_id: str,
    limit: int = 10,
    offset: int = 0,
    min_confidence: float = 0.25
):
    """
    Семантический поиск в памяти
    
    Args:
        query: Поисковый запрос
        user_id: ID пользователя
        limit: Максимальное количество результатов
        offset: Смещение для пагинации
        min_confidence: Минимальная уверенность (0.0-1.0)
    """
    try:
        logger.info(f"Semantic search request: query='{query}', user_id='{user_id}', limit={limit}")
        
        # Используем semantic_memory напрямую (как в интегрированном поиске)
        if not memory_orchestrator or not memory_orchestrator.semantic_memory:
            raise HTTPException(status_code=503, detail="Semantic memory not available")
        
        if memory_orchestrator and memory_orchestrator.semantic_memory:
            all_results = await memory_orchestrator.semantic_memory.search_knowledge(
                user_id=user_id,
                query=query,
                limit=limit + offset,
            min_confidence=min_confidence
        )
        
        # Применяем offset вручную
        results = all_results[offset:offset + limit] if offset < len(all_results) else []
        
        # Преобразуем результаты в словари
        search_results = []
        for result in results:
            distance = getattr(result, 'distance', None)
            relevance_score = getattr(result, 'relevance_score', 1.0 - distance if distance is not None else 1.0)
            
            search_results.append({
                "id": result.id,
                "content": result.content,
                "user_id": result.user_id,
                "distance": distance,
                "relevance_score": relevance_score,
                "knowledge_type": getattr(result, 'knowledge_type', None),
                "category": getattr(result, 'category', None),
                "confidence": getattr(result, 'confidence', None),
                "importance": getattr(result, 'importance', None),
                "created_at": (getattr(result, 'created_at', getattr(result, 'timestamp', None)) or datetime.now()).isoformat(),
                "updated_at": (getattr(result, 'updated_at', getattr(result, 'timestamp', None)) or datetime.now()).isoformat()
            })
        
        return {
            "success": True,
            "results": search_results,
            "search_type": "semantic",
            "query": query,
            "user_id": user_id,
            "total_results": len(search_results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Semantic search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

@app.get("/api/memory/semantic/list")
async def list_semantic_memories(
    user_id: str,
    limit: int = 10,
    offset: int = 0
):
    """
    Получить список семантической памяти
    
    Args:
        user_id: ID пользователя
        limit: Максимальное количество результатов
        offset: Смещение для пагинации
    """
    try:
        logger.info(f"List semantic memories request: user_id='{user_id}', limit={limit}")
        
        # Используем search_knowledge с пустым запросом для получения всех записей
        if memory_orchestrator and memory_orchestrator.semantic_memory:
            all_results = await memory_orchestrator.semantic_memory.search_knowledge(
                user_id=user_id,
                query="",  # Пустой запрос для получения всех записей
            limit=limit + offset,
            min_confidence=0.25  # Унифицированный порог confidence
        )
        
        # Применяем offset вручную
        results = all_results[offset:offset + limit] if offset < len(all_results) else []
        
        # Преобразуем результаты в словари
        memory_list = []
        for result in results:
            memory_list.append({
                "id": result.id,
                "content": result.content,
                "user_id": result.user_id,
                "knowledge_type": getattr(result, 'knowledge_type', None),
                "category": getattr(result, 'category', None),
                "confidence": getattr(result, 'confidence', None),
                "importance": getattr(result, 'importance', None),
                       "created_at": (getattr(result, 'created_at', getattr(result, 'timestamp', None)) or datetime.now()).isoformat(),
                "updated_at": (getattr(result, 'updated_at', getattr(result, 'timestamp', None)) or datetime.now()).isoformat()
            })
        
        return {
            "success": True,
            "results": memory_list,
            "user_id": user_id,
            "total_results": len(memory_list),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"List semantic memories error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"List semantic memories failed: {str(e)}")

# ==================== ПРОАКТИВНЫЕ ЦЕЛИ API ====================

@app.post("/api/proactive-goals", response_model=CreateProactiveGoalResponse, dependencies=[Depends(require_api_key)])
async def create_proactive_goal(request: CreateProactiveGoalRequest):
    """Создание проактивной цели"""
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        try:
            from ..memory_levels.proactive_goals_timer import GoalTriggerType, GoalActionType
        except ImportError:
            from memory_levels.proactive_goals_timer import GoalTriggerType, GoalActionType
        
        goal_id = memory_orchestrator.create_proactive_goal(
            user_id=request.user_id,
            name=request.name,
            description=request.description,
            trigger_type=GoalTriggerType(request.trigger_type.value),
            trigger_value=request.trigger_value,
            action_type=GoalActionType(request.action_type.value),
            action_params=request.action_params
        )
        
        return CreateProactiveGoalResponse(
            success=True, 
            goal_id=goal_id, 
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error creating proactive goal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/proactive-goals", response_model=ProactiveGoalsListResponse)
async def list_proactive_goals(user_id: Optional[str] = None):
    """Получение списка проактивных целей"""
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        goals = memory_orchestrator.get_proactive_goals(user_id)
        
        goals_data = []
        for goal in goals:
            goals_data.append(ProactiveGoalItem(
                id=goal.id,
                user_id=goal.user_id,
                name=goal.name,
                description=goal.description,
                trigger_type=goal.trigger_type.value,
                trigger_value=goal.trigger_value,
                action_type=goal.action_type.value,
                action_params=goal.action_params,
                status=goal.status.value,
                created_at=goal.created_at.isoformat(),
                last_triggered=goal.last_executed.isoformat() if goal.last_executed else None
            ))
        
        return ProactiveGoalsListResponse(
            goals=goals_data, 
            total_count=len(goals_data)
        )
    except Exception as e:
        logger.error(f"Error listing proactive goals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/proactive-goals/{goal_id}", response_model=DeleteProactiveGoalResponse, dependencies=[Depends(require_api_key)])
async def delete_proactive_goal(goal_id: str):
    """Удаление проактивной цели"""
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        success = memory_orchestrator.remove_proactive_goal(goal_id)
        
        if success:
            return DeleteProactiveGoalResponse(
                success=True, 
                message="Goal deleted", 
                timestamp=datetime.now().isoformat()
            )
        else:
            raise HTTPException(status_code=404, detail="Goal not found")
    except Exception as e:
        logger.error(f"Error deleting proactive goal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/proactive-goals/start", dependencies=[Depends(require_api_key)])
async def start_proactive_timer():
    """Запуск проактивного таймера"""
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        await memory_orchestrator.start_proactive_timer()
        
        return {"success": True, "message": "Proactive timer started", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error starting proactive timer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/proactive-goals/stop", dependencies=[Depends(require_api_key)])
async def stop_proactive_timer():
    """Остановка проактивного таймера"""
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        await memory_orchestrator.stop_proactive_timer()
        
        return {"success": True, "message": "Proactive timer stopped", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error stopping proactive timer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/proactive-goals/status", response_model=ProactiveGoalsStatusResponse)
async def get_proactive_timer_status():
    """Получение статуса проактивного таймера"""
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        status_data = memory_orchestrator.get_proactive_timer_status()
        
        is_running = status_data.get("is_running", False)
        status_text = "running" if is_running else "stopped"
        
        return ProactiveGoalsStatusResponse(
            status=status_text,
            active_goals=status_data.get("active_goals", 0),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting proactive timer status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/proactive-goals/results", response_model=ProactiveGoalsResultsResponse)
async def get_proactive_execution_results(limit: Optional[int] = None):
    """Получение результатов выполнения проактивных целей"""
    try:
        global memory_orchestrator
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        results = memory_orchestrator.get_proactive_execution_results(limit)
        
        return ProactiveGoalsResultsResponse(
            results=results,
            total_executed=len(results),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting proactive execution results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== ЭНДПОИНТЫ МОНИТОРИНГА ПАМЯТИ =====

@app.get("/api/memory/monitoring/status")
async def get_memory_monitoring_status(
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Получение статуса мониторинга памяти - КРИТИЧЕСКИ ВАЖНО!"""
    try:
        # Используем новый метод проверки мониторинга
        status = orchestrator.get_monitoring_status()
        logger.info("✅ Memory monitoring status retrieved successfully")
        return {
            "success": True, 
            "status": status, 
            "timestamp": datetime.now().isoformat(),
            "monitoring_working": status.get("monitoring_working", False),
            "critical_components": {
                "memory_monitor": status.get("memory_monitor_available", False),
                "memory_cleanup": status.get("memory_cleanup_available", False),
                "metrics": status.get("metrics_available", False)
            }
        }
    except Exception as e:
        logger.error(f"🚨 CRITICAL: Error getting memory monitoring status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring status: {e}")

@app.get("/api/memory/monitoring/stats")
async def get_memory_stats(
    limit: int = 100,
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Получение статистики использования памяти"""
    try:
        if hasattr(orchestrator, 'memory_monitor') and orchestrator.memory_monitor:
            stats = orchestrator.memory_monitor.get_memory_stats(limit)
            return {"success": True, "stats": [stat.__dict__ for stat in stats], "count": len(stats)}
        else:
            return {"success": False, "error": "Memory monitoring not available"}
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/monitoring/alerts")
async def get_memory_alerts(
    limit: int = 50,
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Получение алертов памяти"""
    try:
        if hasattr(orchestrator, 'memory_monitor') and orchestrator.memory_monitor:
            alerts = orchestrator.memory_monitor.get_memory_alerts(limit)
            return {"success": True, "alerts": [alert.__dict__ for alert in alerts], "count": len(alerts)}
        else:
            return {"success": False, "error": "Memory monitoring not available"}
    except Exception as e:
        logger.error(f"Error getting memory alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/monitoring/cleanup")
async def force_memory_cleanup(
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Принудительная очистка памяти"""
    try:
        result = await orchestrator.force_memory_cleanup()
        return {"success": True, "result": result, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error performing memory cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/monitoring/start")
async def start_memory_monitoring(
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Запуск мониторинга памяти"""
    try:
        await orchestrator.start_memory_monitoring()
        return {"success": True, "message": "Memory monitoring started", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error starting memory monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/monitoring/stop")
async def stop_memory_monitoring(
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Остановка мониторинга памяти"""
    try:
        await orchestrator.stop_memory_monitoring()
        return {"success": True, "message": "Memory monitoring stopped", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error stopping memory monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== ЭНДПОИНТЫ АВТОМАТИЧЕСКОГО ВОССТАНОВЛЕНИЯ =====

@app.get("/api/memory/recovery/status")
async def get_auto_recovery_status(
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Получение статуса автоматического восстановления"""
    try:
        status = orchestrator.get_auto_recovery_status()
        return {"success": True, "status": status, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting auto recovery status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/recovery/components")
async def get_components_status(
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Получение статуса всех компонентов"""
    try:
        components = orchestrator.get_components_status()
        return {"success": True, "components": components, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting components status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/recovery/history")
async def get_recovery_history(
    limit: int = 50,
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Получение истории восстановления"""
    try:
        history = orchestrator.get_recovery_history(limit)
        return {"success": True, "history": history, "count": len(history), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting recovery history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/recovery/start")
async def start_auto_recovery(
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Запуск автоматического восстановления"""
    try:
        await orchestrator.start_auto_recovery()
        return {"success": True, "message": "Auto recovery started", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error starting auto recovery: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/recovery/stop")
async def stop_auto_recovery(
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Остановка автоматического восстановления"""
    try:
        await orchestrator.stop_auto_recovery()
        return {"success": True, "message": "Auto recovery stopped", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error stopping auto recovery: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/recovery/force/{component_name}")
async def force_component_recovery(
    component_name: str,
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Принудительное восстановление компонента"""
    try:
        result = await orchestrator.force_component_recovery(component_name)
        return {"success": True, "result": result, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error forcing component recovery: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== ЭНДПОИНТЫ ПОИСКА =====

@app.get("/memories/graph/search")
async def graph_search(
    user_id: str,
    query: str,
    limit: int = 10,
    node_types: Optional[str] = None,
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Поиск по графу памяти"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        # Получаем граф-движок
        get_graph_engine = get_graph_engine_lazy()
        graph_engine = await get_graph_engine(orchestrator.graph_memory)
        if not graph_engine:
            raise HTTPException(status_code=503, detail="Graph engine not available")
        
        # Преобразуем node_types из строки в список
        node_types_list = [node_types] if node_types else None
        
        # Создаем запрос для графового поиска
        from ..search.graph_search import GraphSearchQuery
        search_query = GraphSearchQuery(
            query=query,
            user_id=user_id,
            search_type="simple",
            limit=limit,
            node_types=node_types_list
        )
        
        # Выполняем поиск
        search_results = await graph_engine.search(search_query)
        
        # Преобразуем результаты в нужный формат
        results = []
        for result in search_results:
            results.append({
                "id": result.node_id,
                "name": result.node_name,
                "type": result.node_type,
                "relevance_score": result.relevance_score,
                "path_length": result.path_length,
                "relationship_strength": result.relationship_strength
            })
        
        return {
            "success": True,
            "user_id": user_id,
            "query": query,
            "nodes": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in graph search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# DEPRECATED: Use /memories/hybrid/search instead
# @app.get("/memories/semantic/search") - REMOVED: This was a stub endpoint
# Semantic search is now available through /memories/hybrid/search

@app.get("/memories/hybrid/search")
async def hybrid_search(
    user_id: str,
    query: str,
    limit: int = 10,
    min_confidence: float = 0.0,
    memory_levels: Optional[str] = None,  # Стандартизировано на memory_levels
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator)
):
    """Гибридный поиск по памяти"""
    # Валидация параметров (ДО try-except)
    if limit <= 0:
        raise HTTPException(status_code=422, detail="Limit must be positive")
    if not (0.0 <= min_confidence <= 1.0):
        raise HTTPException(status_code=422, detail="Min confidence must be between 0.0 and 1.0")
    
    try:
        
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        
        # Получаем гибридный движок
        get_hybrid_engine = get_hybrid_engine_lazy()
        hybrid_engine = await get_hybrid_engine(orchestrator)
        if not hybrid_engine:
            raise HTTPException(status_code=503, detail="Hybrid engine not available")
        
        # Обрабатываем memory_levels параметр
        memory_levels_list = None
        if memory_levels:
            memory_levels_list = [level.strip() for level in memory_levels.split(',') if level.strip()]
        
        # Выполняем поиск
        results = await hybrid_engine.search(query, user_id, limit=limit)
        
        return {
            "success": True,
            "user_id": user_id,
            "query": query,
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== МЕЖУРОВНЕВЫЕ ССЫЛКИ API ====================

@app.get("/memories/interlevel/references/{user_id}/{level}/{item_id}")
async def get_interlevel_references(
    user_id: str,
    level: str,
    item_id: str
):
    """Получение межуровневых ссылок для элемента памяти"""
    try:
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        references = await memory_orchestrator.get_interlevel_references(item_id, level, user_id)
        return {
            "user_id": user_id,
            "level": level,
            "item_id": item_id,
            "references": references,
            "total_references": len(references),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting interlevel references: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/interlevel/consolidate/{user_id}")
async def consolidate_with_interlevel_tracking(user_id: str):
    """Консолидация памяти с отслеживанием межуровневых ссылок"""
    try:
        if not memory_orchestrator:
            raise HTTPException(status_code=503, detail="Memory Orchestrator not initialized")
        result = await memory_orchestrator.consolidate_with_interlevel_tracking(user_id)
        return {
            "user_id": user_id,
            "consolidation_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in interlevel consolidation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/interlevel/duplicates/{user_id}")
async def check_interlevel_duplicates(
    user_id: str,
    content: str,
    exclude_level: Optional[str] = None
):
    """Проверка дубликатов между уровнями памяти"""
    try:
        if memory_orchestrator and hasattr(memory_orchestrator, '_check_interlevel_duplicates'):
            duplicates = await memory_orchestrator._check_interlevel_duplicates(
                content, user_id, exclude_level or ""
            )
        else:
            duplicates = []
        return {
            "user_id": user_id,
            "content": content[:100] + "..." if len(content) > 100 else content,
            "exclude_level": exclude_level,
            "duplicates": duplicates,
            "total_duplicates": len(duplicates),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking interlevel duplicates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CACHE MONITORING ENDPOINTS
# ============================================================================

@app.get("/api/monitoring/cache/stats")
async def get_cache_stats():
    """Получить статистику кэша"""
    try:
        from ..cache.sqlite_cache import get_sqlite_cache
        
        sqlite_cache = await get_sqlite_cache()
        stats = await sqlite_cache.get_stats()
        
        return {
            "cache_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/monitoring/cache/cleanup")
async def cleanup_cache():
    """Очистить истекшие записи кэша"""
    try:
        from ..cache.sqlite_cache import get_sqlite_cache
        
        sqlite_cache = await get_sqlite_cache()
        cleaned_count = await sqlite_cache.cleanup_expired()
        
        return {
            "cleaned_entries": cleaned_count,
            "timestamp": datetime.now().isoformat(),
            "message": f"Cleaned up {cleaned_count} expired cache entries"
        }
    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/monitoring/cache/clear")
async def clear_cache():
    """Полная очистка кэша"""
    try:
        from ..cache.memory_cache import cache_delete_prefix
        
        # Очищаем все типы кэша
        total_deleted = 0
        prefixes = ["search:", "user_context:", "memory:", "embedding:", "ml_search:", "optimization:", "concepts:", "entities:"]
        
        for prefix in prefixes:
            deleted = await cache_delete_prefix(prefix)
            total_deleted += deleted
            logger.info(f"Cleared {deleted} entries with prefix '{prefix}'")
        
        return {
            "total_deleted": total_deleted,
            "timestamp": datetime.now().isoformat(),
            "message": f"Cleared {total_deleted} cache entries"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Запуск сервера
    uvicorn.run(
        "memory_api:app",
        host="0.0.0.0",
        port=8005,
        reload=False,
        log_level="info"
    )
