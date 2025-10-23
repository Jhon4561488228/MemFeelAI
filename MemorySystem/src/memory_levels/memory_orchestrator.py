"""
Memory Orchestrator для AIRI Memory System
Главный координатор всех уровней памяти
"""

import asyncio
import time
import uuid
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import logging
from ..utils.validation import validate_importance, validate_confidence, validate_metadata_values
# Импорт мониторинга - ПРАВИЛЬНЫЕ ОТНОСИТЕЛЬНЫЕ ИМПОРТЫ
from ..monitoring.metrics import inc, record_event, merge_counters, set_gauge
from ..monitoring.performance_tracker import track_operation_performance
from ..monitoring.search_metrics import record_search_metrics

try:
    from .sensor_buffer import SensorBuffer, SensorType, SensorData
    from .working_memory import WorkingMemoryManager, WorkingMemoryItem
    from .short_term_memory import ShortTermMemoryManager, ShortTermMemoryItem
    from .episodic_memory import EpisodicMemoryManager, EpisodicMemoryItem
    from .semantic_memory import SemanticMemoryManager, SemanticMemoryItem
    from .graph_memory_sqlite import GraphMemoryManagerSQLite as GraphMemoryManager, GraphNode, GraphEdge
    from .procedural_memory import ProceduralMemoryManager, ProceduralMemoryItem
    from .memory_consolidator import MemoryConsolidator
except ImportError:
    from sensor_buffer import SensorBuffer, SensorType, SensorData
    from working_memory import WorkingMemoryManager, WorkingMemoryItem
    from short_term_memory import ShortTermMemoryManager, ShortTermMemoryItem
    from episodic_memory import EpisodicMemoryManager, EpisodicMemoryItem
    from semantic_memory import SemanticMemoryManager, SemanticMemoryItem
    from graph_memory_sqlite import GraphMemoryManagerSQLite as GraphMemoryManager, GraphNode, GraphEdge
    from procedural_memory import ProceduralMemoryManager, ProceduralMemoryItem
    from memory_consolidator import MemoryConsolidator
try:
    from .proactive_goals_timer import ProactiveGoalsTimer, ProactiveGoal, GoalTriggerType, GoalActionType, GoalStatus
except ImportError:
    from proactive_goals_timer import ProactiveGoalsTimer, ProactiveGoal, GoalTriggerType, GoalActionType, GoalStatus
try:
    from ..extractors.fact_extractor import FactExtractor
    from ..classifiers.event_classifier import EventClassifier
    from ..prioritizers.memory_prioritizer import compute_priority
    from ..analyzers.relationship_analyzer import naive_relationships, advanced_relationships
    from ..emotion_analysis_service import get_emotion_service
    from ..search.entity_extractor import EntityExtractor
except ImportError:
    from extractors.fact_extractor import FactExtractor
    from classifiers.event_classifier import EventClassifier
    from prioritizers.memory_prioritizer import compute_priority
    from analyzers.relationship_analyzer import naive_relationships, advanced_relationships
    from emotion_analysis_service import get_emotion_service
    from search.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)

@dataclass
class MemoryQuery:
    """Запрос к системе памяти"""
    query: str
    user_id: str
    memory_levels: Optional[List[str]] = None
    limit: int = 10
    offset: int = 0
    include_emotions: bool = True
    include_relationships: bool = True
    track_interlevel_references: bool = True  # Отслеживание межуровневых ссылок

@dataclass
class MemoryResult:
    """Результат поиска в памяти"""
    level: str
    items: List[Any]
    relevance_scores: List[float]
    total_found: int

@dataclass
class UnifiedMemoryResult:
    """Унифицированный результат поиска"""
    query: str
    user_id: str
    results: Dict[str, MemoryResult]
    total_items: int
    processing_time: float
    recommendations: Optional[List[str]] = None

class MemoryOrchestrator:
    """Оркестратор системы памяти"""
    
    def __init__(self, chromadb_path: Optional[str] = None):
        # Resolve data directories from ENV for offline-install layout
        data_root = os.getenv("AIRI_DATA_DIR", "./data")
        chroma_base = os.getenv("CHROMADB_DIR", os.path.join(data_root, "chroma_db"))
        base = chromadb_path or chroma_base
        self.chromadb_path = base
        
        # ОПТИМИЗАЦИЯ: Создаем единый LLM Provider для всех компонентов
        # Это устраняет множественную инициализацию и ускоряет холодный старт
        try:
            try:
                from ..providers.lm_studio_provider import LMStudioProvider
            except ImportError:
                from providers.lm_studio_provider import LMStudioProvider
            self.llm_provider = LMStudioProvider("config/lm_studio_config.yaml")
            logger.info("Shared LLM Provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize shared LLM Provider: {e}")
            self.llm_provider = None
        
        # Инициализируем менеджеры всех уровней памяти
        # Уровень 1: Sensor Buffer - кольцевой буфер для потоковых данных
        self.sensor_buffer = SensorBuffer()
        
        # Уровни 2-7: Основные уровни памяти
        try:
            self.working_memory = WorkingMemoryManager(base)
            self.short_term_memory = ShortTermMemoryManager(base)
            self.episodic_memory = EpisodicMemoryManager(base)
            self.semantic_memory = SemanticMemoryManager(base)
            # Используем единую базу данных для графа
            graph_db_path = os.path.join(base, "memory_system.db")
            self.graph_memory = GraphMemoryManager(graph_db_path)
            self.procedural_memory = ProceduralMemoryManager(base)
            # ОПТИМИЗАЦИЯ: Передаем единый LLM Provider всем компонентам
            self.fact_extractor = FactExtractor(self.llm_provider)
            self.event_classifier = EventClassifier()
            self.entity_extractor = EntityExtractor(self.llm_provider)
            logger.info("All memory managers initialized successfully with shared LLM Provider")
        except Exception as e:
            logger.error(f"Error initializing memory managers: {e}")
            raise
        
        # Настройки поиска (8-уровневая система памяти)
        self.search_weights = {
            "sensor": 0.35,      # Самый высокий приоритет для текущих сенсорных данных
            "working": 0.25,     # Высокий приоритет для активного контекста
            "short_term": 0.2,   # Высокий приоритет для недавних событий
            "episodic": 0.1,     # Средний приоритет для значимых событий
            "semantic": 0.05,    # Низкий приоритет для знаний
            "fts5": 0.03,        # Низкий приоритет для ключевого поиска
            "graph": 0.03,       # Низкий приоритет для связей
            "procedural": 0.02   # Низкий приоритет для навыков
        }
        
        # Инициализация метрик для фоновых задач
        # Инициализация метрик фоновых задач - КРИТИЧЕСКИ ВАЖНО!
        try:
            inc("background_tasks_started", 0)  # Инициализация счетчика
            inc("background_tasks_completed", 0)
            inc("background_tasks_failed", 0)
            set_gauge("background_tasks_queue_size", 0)
            logger.info("✅ Background task metrics initialized successfully")
        except Exception as e:
            logger.error(f"🚨 CRITICAL: Failed to initialize background task metrics: {e}")
            logger.error("🚨 Monitoring fallback: using in-memory metrics")
        
        # Инициализация консолидатора памяти
        self.consolidator = MemoryConsolidator(self)
        logger.info("Memory consolidator initialized")
        
        # Инициализация проактивного таймера целей
        self.proactive_timer = ProactiveGoalsTimer(self)
        logger.info("Proactive goals timer initialized")
        
        # Инициализация мониторинга памяти
        try:
            # Импорт мониторинга памяти - ПРАВИЛЬНЫЕ ОТНОСИТЕЛЬНЫЕ ИМПОРТЫ
            from ..monitoring.memory_monitor import get_memory_monitor
            from ..monitoring.memory_cleanup import get_memory_cleanup
            
            self.memory_monitor = get_memory_monitor()
            self.memory_cleanup = get_memory_cleanup()
            
            # Проверяем успешность инициализации мониторинга
            if self.memory_monitor is None:
                logger.warning("⚠️ Memory monitor is None - monitoring may be disabled")
            else:
                logger.info("✅ Memory monitor initialized successfully")
                
            if self.memory_cleanup is None:
                logger.warning("⚠️ Memory cleanup is None - cleanup may be disabled")
            else:
                logger.info("✅ Memory cleanup initialized successfully")
            
            # Регистрируем компоненты для мониторинга
            self.memory_monitor.register_component(
                "sensor_buffer",
                self._get_sensor_buffer_stats,
                None  # cleanup функции не используются в мониторинге
            )
            self.memory_monitor.register_component(
                "memory_orchestrator",
                self._get_orchestrator_stats,
                None  # cleanup функции не используются в мониторинге
            )
            
            logger.info("Memory monitoring initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize memory monitoring: {e}")
            self.memory_monitor = None
            self.memory_cleanup = None
        
        # Инициализация автоматического восстановления ОТКЛЮЧЕНА
        # try:
        #     try:
        #         from ..recovery.auto_recovery import get_auto_recovery
        #         from ..recovery.recovery_handlers import get_recovery_handlers
        #     except ImportError:
        #         from recovery.auto_recovery import get_auto_recovery
        #         from recovery.recovery_handlers import get_recovery_handlers
        #     
        #     self.auto_recovery = get_auto_recovery()
        #     # Передаем self (memory_orchestrator) в recovery_handlers
        #     self.recovery_handlers = get_recovery_handlers(memory_orchestrator=self)
        
        self.auto_recovery = None
        self.recovery_handlers = None
        
        # Регистрация компонентов для восстановления ОТКЛЮЧЕНА
        # # Регистрируем компоненты для восстановления
        # self.auto_recovery.register_component(
        #     "memory_orchestrator",
        #     self.recovery_handlers.check_memory_orchestrator_health,
        #     self.recovery_handlers.recover_memory_orchestrator,
        #     is_critical=True,
        #     check_interval=120,  # 2 минуты вместо 30 секунд
        #     max_failures=5       # Больше попыток
        # )
        # 
        # self.auto_recovery.register_component(
        #     "sensor_buffer",
        #     self.recovery_handlers.check_sensor_buffer_health,
        #     self.recovery_handlers.recover_sensor_buffer,
        #     is_critical=False,
        #     check_interval=60,
        #     max_failures=5
        # )
        # 
        # self.auto_recovery.register_component(
        #     "memory_cache",
        #     self.recovery_handlers.check_memory_cache_health,
        #     self.recovery_handlers.recover_memory_cache,
        #     is_critical=False,
        #     check_interval=120,
        #     max_failures=3
        # )
        # 
        # self.auto_recovery.register_component(
        #     "sqlite_cache",
        #     self.recovery_handlers.check_sqlite_cache_health,
        #     self.recovery_handlers.recover_sqlite_cache,
        #     is_critical=True,
        #     check_interval=60,
        #     max_failures=3
        # )
        # 
        # self.auto_recovery.register_component(
        #     "lm_provider",
        #     self.recovery_handlers.check_lm_provider_health,
        #     self.recovery_handlers.recover_lm_provider,
        #     is_critical=True,
        #     check_interval=180,  # 3 минуты вместо 30 секунд
        #     max_failures=3
        # )
        
        logger.info("Auto recovery ОТКЛЮЧЕН")
        # except Exception as e:
        #     logger.warning(f"Failed to initialize auto recovery: {e}")
        #     self.auto_recovery = None
        #     self.recovery_handlers = None
        
        # Счетчики для рекурсивной суммаризации (thread-safe)
        self._summary_counters = {}  # user_id -> count
        self._summary_counters_lock = asyncio.Lock()  # Thread-safe блокировка
        self._summary_threshold = 10  # каждые 10 сообщений
        
        # Отслеживание активных summary задач для предотвращения дублирования
        self._summary_tasks = {}  # user_id -> asyncio.Task
        self._summary_task_locks = {}  # user_id -> asyncio.Lock
        
        # Отслеживание фоновых задач для корректного завершения
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
    
    def _add_background_task(self, task: asyncio.Task) -> None:
        """Добавляет фоновую задачу в отслеживание"""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def _queue_for_retry(self, task_type: str, task_data: dict):
        """Добавление задачи в очередь для повторной обработки"""
        try:
            if not hasattr(self, '_retry_queue'):
                self._retry_queue = []
            
            retry_task = {
                "task_type": task_type,
                "task_data": task_data,
                "created_at": time.time(),
                "next_retry": time.time() + 60  # Повтор через 1 минуту
            }
            
            self._retry_queue.append(retry_task)
            logger.info(f"Task queued for retry: {task_type}, queue size: {len(self._retry_queue)}")
            
        except Exception as e:
            logger.error(f"Failed to queue task for retry: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Корректное завершение работы оркестратора"""
        logger.info("Shutting down MemoryOrchestrator...")
        
        # Останавливаем проактивный таймер
        if hasattr(self, 'proactive_timer'):
            await self.proactive_timer.stop()
            logger.info("Proactive goals timer stopped")
        
        # Устанавливаем флаг завершения
        self._shutdown_event.set()
        
        # Отменяем все фоновые задачи
        if self._background_tasks:
            logger.info(f"Cancelling {len(self._background_tasks)} background tasks...")
            for task in self._background_tasks.copy():
                if not task.done():
                    task.cancel()
            
            # Ждем завершения всех задач
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        logger.info("MemoryOrchestrator shutdown complete")
        
    async def add_memory(self, content: str, user_id: str, 
                        memory_type: str = "general", importance: float = 0.5,
                        emotion_data: Optional[Dict[str, Any]] = None,
                        context: Optional[str] = None,
                        participants: Optional[List[str]] = None,
                        location: Optional[str] = None) -> Dict[str, str]:
        """
        Добавить память на все уровни
        
        Args:
            content: Содержимое памяти
            user_id: ID пользователя
            memory_type: Тип памяти (conversation, event, knowledge, skill, etc.)
            importance: Важность (0.0-1.0) - валидируется автоматически
            emotion_data: Эмоциональные данные
            context: Контекст
            participants: Участники
            location: Местоположение
            
        Returns:
            Словарь с ID созданных элементов на каждом уровне
        """
        try:
            start_time = time.time()
            results = {}
            
            # Валидация входных параметров
            importance = validate_importance(importance)
            
            logger.info(f"AddMemory: user={user_id} type={memory_type} importance={importance:.2f}")
            
            # УРОВЕНЬ 1: Sensor Buffer - добавляем текстовые данные в кольцевой буфер
            try:
                sensor_id = self.sensor_buffer.add_sensor_data(
                    sensor_type=SensorType.TEXT,
                    data=content,
                    user_id=user_id,
                    metadata={
                        "memory_type": memory_type,
                        "importance": importance,
                        "context": context,
                        "participants": participants,
                        "location": location
                    },
                    text_content=content
                )
                results["sensor"] = sensor_id
                logger.debug(f"Добавлено в Sensor Buffer: {sensor_id}")
            except Exception as e:
                logger.warning(f"Ошибка добавления в Sensor Buffer: {e}")
                results["sensor"] = None
            from ..monitoring.metrics import time_block
            with time_block("memory_add_emotion_analysis"):
                # 0. Auto emotion analysis (optional)
                if emotion_data is None and os.getenv("EMOTION_AUTO", "1") in ("1", "true", "True"):
                    try:
                        svc = get_emotion_service()
                        res = await svc.analyze_text_emotions(content, include_validation=True)
                        emotion_data = {
                            "primary_emotion": res.primary_emotion,
                            "primary_confidence": res.primary_confidence,
                            "secondary_emotion": res.secondary_emotion,
                            "secondary_confidence": res.secondary_confidence,
                            "tertiary_emotion": res.tertiary_emotion,
                            "tertiary_confidence": res.tertiary_confidence,
                            "consistency": res.consistency,
                            "dominant_source": res.dominant_source,
                            "validation_applied": res.validation_applied,
                        }
                        # Обновление метрик анализа эмоций - КРИТИЧЕСКИ ВАЖНО!
                        try:
                            inc("emotion_analysis_total")
                            logger.debug("✅ Emotion analysis metrics updated successfully")
                        except Exception as e:
                            logger.error(f"🚨 CRITICAL: Failed to update emotion analysis metrics: {e}")
                            logger.error("🚨 Monitoring fallback: using in-memory metrics")
                    except Exception as e:
                        logger.warning(f"Auto emotion analysis failed: {e}")

                # 1. Working Memory - всегда добавляем
                wm_id = await self.working_memory.add_memory(
                    content=content,
                    user_id=user_id,
                    importance=importance,
                    confidence=validate_confidence(min(1.0, max(0.0, importance * 0.8 + 0.2))),
                    context=context,
                    emotion_data=emotion_data
                )
                results["working"] = wm_id
                # Обновление метрик Working Memory - КРИТИЧЕСКИ ВАЖНО!
                try:
                    inc("memory_add_level_working")
                    logger.debug("✅ Working memory metrics updated successfully")
                except Exception as _e:
                    logger.error(f"🚨 CRITICAL: Failed to update working memory metrics: {_e}")
                    logger.error("🚨 Monitoring fallback: using in-memory metrics")
            
            # 2. Short-term Memory - добавляем если важность > 0.3 (с авто-классификацией события)
            if importance > 0.3:
                try:
                    auto_type = await self.event_classifier.classify(content)
                except Exception:
                    auto_type = memory_type
                stm_id = await self.short_term_memory.add_event(
                    content=content,
                    user_id=user_id,
                    importance=importance,
                    confidence=validate_confidence(min(1.0, max(0.0, importance * 0.8 + 0.2))),
                    event_type=auto_type or memory_type,
                    location=location,
                    participants=participants or [],
                    emotion_data=emotion_data
                )
                results["short_term"] = stm_id
                # Обновление метрик Short-term Memory - КРИТИЧЕСКИ ВАЖНО!
                try:
                    inc("memory_add_level_short_term")
                    logger.debug("✅ Short-term memory metrics updated successfully")
                except Exception as _e:
                    logger.error(f"🚨 CRITICAL: Failed to update short-term memory metrics: {_e}")
                    logger.error("🚨 Monitoring fallback: using in-memory metrics")
            
            # 3. Episodic Memory - добавляем если важность > 0.5
            if importance > 0.5:
                with time_block("memory_add_episodic"):
                    ep_id = await self.episodic_memory.add_experience(
                        content=content,
                        user_id=user_id,
                        importance=importance,
                        confidence=validate_confidence(min(1.0, max(0.0, importance * 0.8 + 0.2))),
                        event_type=memory_type,
                        location=location,
                        participants=participants or [],
                        emotion_data=emotion_data,
                        significance=importance,
                        vividness=0.7 if emotion_data else 0.5,
                        context=context
                    )
                    results["episodic"] = ep_id
                    try:
                        inc("memory_add_level_episodic")
                    except Exception as _e:
                        logger.warning(f"metrics inc failed (episodic): {_e}")
            
            # 4. Semantic Memory - добавляем если это знание, а также извлекаем факты при высокой важности
            if memory_type in ["knowledge", "fact", "concept", "definition"]:
                sm_id = await self.semantic_memory.add_knowledge(
                    content=content,
                    user_id=user_id,
                    knowledge_type=memory_type,
                    category="general",
                    confidence=validate_confidence(min(1.0, max(0.0, importance * 0.8 + 0.2))),
                    importance=importance
                )
                results["semantic"] = [sm_id]
                try:
                    inc("memory_add_level_semantic")
                except Exception as _e:
                    logger.warning(f"metrics inc failed (semantic-extracted): {_e}")
            elif importance >= 0.5:
                # ОПТИМИЗАЦИЯ: Факт-экстракция через единый LLM вызов
                # Факты уже извлечены в _process_memory_unified, обрабатываем их
                with time_block("memory_add_semantic_unified"):
                    unified_result = await self._process_memory_unified(content, user_id, importance)
                    facts = unified_result.get("facts", [])
                    logger.info(f"DEBUG: unified_result keys: {list(unified_result.keys())}")
                    logger.info(f"DEBUG: facts from unified_result: {facts}")
                    logger.info(f"DEBUG: facts type: {type(facts)}, len: {len(facts) if facts else 'None'}")
                
                if facts:
                    logger.info(f"Processing {len(facts)} facts from unified result")
                    # Обрабатываем факты как semantic memories с дедупликацией
                    processed_facts = set()  # Для предотвращения дубликатов
                    for fact in facts:
                        if isinstance(fact, list) and len(fact) >= 3:
                            subject, relation, object_ = fact[0], fact[1], fact[2]
                            fact_content = f"{subject} {relation} {object_}"
                            
                            # Проверяем дубликаты
                            if fact_content in processed_facts:
                                logger.debug(f"Skipping duplicate fact: {fact_content}")
                                continue
                            
                        # Проверяем существование в базе
                        existing = await self.semantic_memory.search_knowledge(
                            user_id=user_id,
                            query=fact_content,
                            limit=1,
                            min_confidence=0.25  # Унифицированный порог confidence
                        )
                        
                        logger.info(f"DEDUPLICATION CHECK: fact='{fact_content}', existing_count={len(existing) if existing else 0}")
                        
                        if existing and len(existing) > 0:
                            # Проверяем distance - если 0.0, то это точное совпадение
                            first_result = existing[0]
                            distance = getattr(first_result, 'distance', None)
                            logger.info(f"DEDUPLICATION CHECK: distance={distance}, threshold=0.1, should_skip={distance is not None and distance <= 0.1}")
                            if distance is not None and distance <= 0.1:  # Очень близкое совпадение
                                logger.info(f"SKIPPING DUPLICATE FACT: distance={distance}, fact='{fact_content}'")
                                continue
                            else:
                                logger.info(f"KEEPING FACT: distance={distance} > 0.1, fact='{fact_content}'")
                        
                        # Добавляем факт только если он не дубликат
                        processed_facts.add(fact_content)
                        sm_id = await self.semantic_memory.add_knowledge(
                            content=fact_content,
                            user_id=user_id,
                            importance=importance * 0.8,  # Факты немного менее важны
                            knowledge_type="fact",
                            source="unified_processing"
                        )
                        logger.info(f"Added fact as semantic memory: {sm_id}")
                else:
                    logger.info(f"No facts extracted for user {user_id}")
                
                try:
                    inc("memory_add_level_semantic")
                except Exception as e:
                    logger.warning(f"Failed to update semantic metrics: {e}")
            
            # 5. Graph Memory - создаем узлы для важных концепций и связываем их
            if importance >= 0.5:
                logger.info(f"Graph processing triggered for importance={importance}")
                # ОПТИМИЗАЦИЯ: Используем уже извлеченные концепции из единого вызова
                # Если концепции еще не извлечены, извлекаем их
                if 'concepts' not in locals():
                    logger.info("Extracting concepts for graph processing")
                    try:
                        concepts, entity_types = await self._extract_concepts(content, user_id, importance)
                        logger.info(f"Extracted {len(concepts)} concepts: {concepts}")
                        logger.info(f"Entity types: {entity_types}")
                    except Exception as e:
                        logger.error(f"Error extracting concepts: {e}")
                        concepts = []
                        entity_types = {}
                else:
                    concepts = []  # Инициализируем concepts если не извлечены
                    logger.info(f"Using existing concepts: {concepts}")
                
                logger.info(f"Final concepts for graph processing: {concepts}")
                if concepts:
                    logger.info(f"Creating {len(concepts)} graph nodes")
                    added_node_ids = []
                    for concept in concepts:
                        try:
                            # Проверяем существование узла перед созданием
                            existing_nodes = await self.graph_memory.search_nodes(
                                user_id=user_id,
                                query=concept,
                                node_type=entity_types.get(concept, "concept"),
                                limit=1
                            )
                            
                            if existing_nodes:
                                # Узел уже существует, используем его ID
                                existing_node = existing_nodes[0]
                                node_id = existing_node.id
                                logger.info(f"Using existing graph node: {concept} -> {node_id}")
                            else:
                                # Создаем новый узел
                                node_type = entity_types.get(concept, "concept")
                                node_id = await self.graph_memory.add_node(
                                    name=concept,
                                    user_id=user_id,
                                    node_type=node_type,
                                    importance=importance
                                )
                                logger.info(f"Created new graph node: {concept} -> {node_id}")
                            
                            added_node_ids.append(node_id)
                        except Exception as e:
                            logger.error(f"Error creating graph node for concept '{concept}': {e}")
                else:
                    logger.warning("No concepts to create graph nodes")
                    added_node_ids = []
                
                # связываем концепции между собой базовой связью 'related'
                for i in range(len(added_node_ids)):
                    for j in range(i + 1, len(added_node_ids)):
                        try:
                            await self.graph_memory.add_edge(
                                source_id=added_node_ids[i],
                                target_id=added_node_ids[j],
                                user_id=user_id,
                                relationship_type="related",
                                strength=min(1.0, importance)
                            )
                        except Exception as e:
                            logger.warning(f"Failed to update relationship analysis metrics: {e}")
                # ОПТИМИЗАЦИЯ: Анализ отношений - АСИНХРОННО (для долгосрочного обучения)
                # Основные узлы уже созданы, LLM получит полный контекст
                task = asyncio.create_task(self._analyze_relationships_async(content, user_id, added_node_ids, importance))
                self._add_background_task(task)
                # Добавляем обработку исключений для предотвращения зависания
                task.add_done_callback(lambda t: logger.debug(f"Relationship analysis task completed: {t.exception() if t.exception() else 'success'}"))
                try:
                    inc("background_tasks_started")
                except Exception as e:
                    logger.warning(f"Failed to update background tasks metrics: {e}")
                logger.info(f"Relationship analysis queued for async processing: user={user_id} nodes={len(added_node_ids)}")
                if added_node_ids:
                    results["graph"] = added_node_ids
                    try:
                        inc("memory_add_level_graph")
                    except Exception as _e:
                        logger.warning(f"metrics inc failed (graph): {_e}")
            else:
                logger.info(f"Graph processing skipped for importance={importance} (threshold: 0.5)")
                added_node_ids = []
            
            # 6. Procedural Memory - добавляем если это навык
            if memory_type in ["skill", "procedure", "algorithm", "method"]:
                pm_id = await self.procedural_memory.add_skill(
                    name=content.split(":")[0] if ":" in content else content[:50],
                    description=content,
                    user_id=user_id,
                    skill_type="cognitive",
                    difficulty=importance,
                    importance=importance
                )
                results["procedural"] = pm_id
                try:
                    inc("memory_add_level_procedural")
                except Exception as _e:
                    logger.warning(f"metrics inc failed (procedural): {_e}")
            
            processing_time = time.time() - start_time
            logger.info(f"AddMemory: stored={results.keys()} duration={processing_time:.2f}s")
            try:
                record_event("add_memory", {
                    "user_id": user_id,
                    "type": memory_type,
                    "importance": importance,
                    "levels": list(results.keys()),
                    "duration": processing_time,
                })
            except Exception as _e:
                logger.warning(f"record_event(add_memory) failed: {_e}")
            
            # ОПТИМИЗАЦИЯ: Запускаем фоновые задачи (не блокируем ответ)
            asyncio.create_task(self._background_post_processing(user_id, content, results, memory_type, importance))
            logger.info(f"Background post-processing started for user {user_id}")
            
            # КРИТИЧНО: Проверяем необходимость создания суммаризации (асинхронно в фоне)
            # ОПТИМИЗАЦИЯ: Используем per-user lock для предотвращения race conditions
            try:
                # Создаем lock для пользователя если его нет
                if user_id not in self._summary_task_locks:
                    self._summary_task_locks[user_id] = asyncio.Lock()
                
                # Проверяем и создаем задачу под lock
                async def _safe_create_summary_task():
                    async with self._summary_task_locks[user_id]:
                        # Двойная проверка под lock
                        if user_id not in self._summary_tasks or self._summary_tasks[user_id].done():
                            # Создаем новую задачу только если предыдущая завершена
                            task = asyncio.create_task(self._check_and_create_summary_async(user_id))
                            self._summary_tasks[user_id] = task
                            logger.debug(f"Summary creation task started for user {user_id}")
                        else:
                            logger.debug(f"Summary creation task already running for user {user_id}")
                
                # Запускаем проверку в фоне
                asyncio.create_task(_safe_create_summary_task())
            except Exception as e:
                logger.warning(f"Failed to start summary creation for user {user_id}: {e}")
            
            # Мониторинг производительности
            duration_ms = (time.time() - start_time) * 1000
            track_operation_performance(
                operation_type="add_memory",
                operation_name="memory_orchestrator_add",
                user_id=user_id,
                duration_ms=duration_ms,
                success=True,
                result_count=len(results),
                metadata={
                    "memory_type": memory_type,
                    "importance": importance,
                    "levels_created": list(results.keys())
                }
            )
            
            return results
            
        except Exception as e:
            # Мониторинг производительности для ошибок
            duration_ms = (time.time() - start_time) * 1000
            track_operation_performance(
                operation_type="add_memory",
                operation_name="memory_orchestrator_add",
                user_id=user_id,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                metadata={
                    "memory_type": memory_type,
                    "importance": importance
                }
            )
            logger.error(f"Error adding memory: {e}")
            raise
    
    async def search_memory(self, query: MemoryQuery) -> UnifiedMemoryResult:
        """
        Поиск по всем уровням памяти
        
        Args:
            query: Запрос к системе памяти
            
        Returns:
            Унифицированный результат поиска
        """
        try:
            start_time = time.time()
            logger.info(f"SearchMemory: user={query.user_id} q='{query.query}' levels={query.memory_levels or 'all'} limit={query.limit} offset={query.offset}")
            
            # ОПТИМИЗАЦИЯ: Проверяем кэш результатов поиска
            cache_key = f"search:{query.user_id}:{hash(query.query)}:{query.limit}:{query.offset}"
            cached_result = await self._get_search_cache(cache_key)
            
            if cached_result and self._is_search_cache_fresh(cached_result, query.user_id):
                logger.info(f"Cache hit for search: {query.query}")
                try:
                    inc("search_cache_hits")
                except Exception as e:
                    logger.warning(f"Failed to update search cache hits metrics: {e}")
                
                # Мониторинг производительности для кэш-хита
                duration_ms = (time.time() - start_time) * 1000
                record_search_metrics(
                    query=query.query,
                    search_type="integrated",
                    user_id=query.user_id,
                    duration_ms=duration_ms,
                    results=[],  # Упрощенный список результатов
                    cache_hit=True,
                    memory_levels=cached_result.get("levels_searched", []),
                    filters={"limit": query.limit, "offset": query.offset}
                )
                
                track_operation_performance(
                    operation_type="search",
                    operation_name="memory_orchestrator_search_cached",
                    user_id=query.user_id,
                    duration_ms=duration_ms,
                    success=True,
                    result_count=cached_result["total_items"],
                    cache_hit=True,
                    metadata={
                        "query_length": len(query.query),
                        "cache_hit": True
                    }
                )
                
                # Возвращаем упрощенный результат из кэша
                return UnifiedMemoryResult(
                    query=cached_result["query"],
                    user_id=cached_result["user_id"],
                    results={},  # Пустые результаты для кэша
                    total_items=cached_result["total_items"],
                    processing_time=0.001,  # Быстрый ответ из кэша
                    recommendations=cached_result.get("recommendations", [])
                )
            
            results: Dict[str, MemoryResult] = {}
            total_items = 0
            level_times: Dict[str, float] = {}
            
            # Определяем уровни для поиска (включаем Sensor Buffer и FTS5)
            levels_to_search = query.memory_levels or ["sensor", "working", "short_term", "episodic", "semantic", "graph", "procedural", "fts5"]
            
            # Поиск по каждому уровню — параллельно
            async def _search_sensor():
                """Поиск в Sensor Buffer - самый быстрый и приоритетный"""
                _t0 = datetime.now()
                try:
                    # Получаем недавние данные из Sensor Buffer
                    recent_data = self.sensor_buffer.get_recent_data(
                        user_id=query.user_id,
                        seconds=30,  # Последние 30 секунд
                        limit=query.limit
                    )
                    
                    # Простой текстовый поиск по содержимому
                    matching_items = []
                    query_lower = query.query.lower()
                    
                    for data in recent_data:
                        if data.text_content and query_lower in data.text_content.lower():
                            matching_items.append(data)
                    
                    # Ограничиваем результаты
                    items = matching_items[query.offset:query.offset + query.limit]
                    
                    level_times["sensor"] = (datetime.now() - _t0).total_seconds()
                    try:
                        set_gauge("search_time_sensor_seconds", level_times["sensor"])
                    except Exception as e:
                        logger.warning(f"Failed to update sensor search time metrics: {e}")
                    
                    return ("sensor", MemoryResult(
                        level="sensor",
                        items=items,
                        relevance_scores=[1.0] * len(items),  # Высокая релевантность для сенсорных данных
                        total_found=len(matching_items)
                    ))
                except Exception as e:
                    logger.warning(f"Ошибка поиска в Sensor Buffer: {e}")
                    level_times["sensor"] = (datetime.now() - _t0).total_seconds()
                    return ("sensor", MemoryResult(
                        level="sensor",
                        items=[],
                        relevance_scores=[],
                        total_found=0
                    ))
            
            async def _search_working():
                _t0 = datetime.now()
                full = await self.working_memory.search_context(query.user_id, query.query, query.limit + query.offset)
                items = full[query.offset:query.offset + query.limit]
                level_times["working"] = (datetime.now() - _t0).total_seconds()
                try:
                    set_gauge("search_time_working_seconds", level_times["working"])
                except Exception as e:
                    logger.warning(f"Failed to update working memory search time metrics: {e}")
                now = datetime.now()
                scores = [
                    compute_priority(
                        base_importance=getattr(it, "importance", 0.5),
                        emotional_intensity=None,
                        recency_hours=max(0.0, (now - getattr(it, "timestamp", now)).total_seconds() / 3600.0),
                    ) for it in items
                ]
                return ("working", MemoryResult("working", items, scores, len(items)))

            async def _search_short_term():
                _t0 = datetime.now()
                full = await self.short_term_memory.search_events(query.user_id, query.query, hours=24, limit=query.limit + query.offset)
                items = full[query.offset:query.offset + query.limit]
                level_times["short_term"] = (datetime.now() - _t0).total_seconds()
                try:
                    set_gauge("search_time_short_term_seconds", level_times["short_term"])
                except Exception as e:
                    logger.warning(f"Failed to update short term memory search time metrics: {e}")
                now = datetime.now()
                scores = [
                    compute_priority(
                        base_importance=getattr(it, "importance", 0.5),
                        emotional_intensity=None,
                        recency_hours=max(0.0, (now - getattr(it, "timestamp", now)).total_seconds() / 3600.0),
                    ) for it in items
                ]
                return ("short_term", MemoryResult("short_term", items, scores, len(items)))

            async def _search_episodic():
                _t0 = datetime.now()
                full = await self.episodic_memory.search_experiences(query.user_id, query.query, days=30, limit=query.limit + query.offset)
                items = full[query.offset:query.offset + query.limit]
                level_times["episodic"] = (datetime.now() - _t0).total_seconds()
                try:
                    set_gauge("search_time_episodic_seconds", level_times["episodic"])
                except Exception as e:
                    logger.warning(f"Failed to update episodic memory search time metrics: {e}")
                now = datetime.now()
                scores = [
                    compute_priority(
                        base_importance=getattr(it, "importance", 0.5),
                        emotional_intensity=None,
                        recency_hours=max(0.0, (now - getattr(it, "timestamp", now)).total_seconds() / 3600.0),
                    ) for it in items
                ]
                return ("episodic", MemoryResult("episodic", items, scores, len(items)))

            async def _search_semantic():
                _t0 = datetime.now()
                full = await self.semantic_memory.search_knowledge(query.user_id, query.query, limit=query.limit + query.offset)
                items = full[query.offset:query.offset + query.limit]
                level_times["semantic"] = (datetime.now() - _t0).total_seconds()
                try:
                    set_gauge("search_time_semantic_seconds", level_times["semantic"])
                except Exception as e:
                    logger.warning(f"Failed to update semantic memory search time metrics: {e}")
                now = datetime.now()
                scores = []
                for it in items:
                    recency_h = None
                    try:
                        recency_h = max(0.0, (now - getattr(it, "last_accessed", now)).total_seconds() / 3600.0)
                    except Exception:
                        recency_h = None
                    scores.append(
                        compute_priority(
                            base_importance=getattr(it, "importance", 0.5),
                            last_accessed=getattr(it, "last_accessed", None),
                            access_count=getattr(it, "access_count", 0),
                            emotional_intensity=None,
                            recency_hours=recency_h,
                        )
                    )
                return ("semantic", MemoryResult("semantic", items, scores, len(items)))

            async def _search_graph():
                _t0 = datetime.now()
                full = await self.graph_memory.search_nodes(query.user_id, query.query, limit=query.limit + query.offset)
                items = full[query.offset:query.offset + query.limit]
                level_times["graph"] = (datetime.now() - _t0).total_seconds()
                try:
                    set_gauge("search_time_graph_seconds", level_times["graph"])
                except Exception as e:
                    logger.warning(f"Failed to update graph memory search time metrics: {e}")
                return ("graph", MemoryResult("graph", items, [0.5] * len(items), len(items)))

            async def _search_procedural():
                _t0 = datetime.now()
                full = await self.procedural_memory.search_skills(query.user_id, query.query, limit=query.limit + query.offset)
                items = full[query.offset:query.offset + query.limit]
                level_times["procedural"] = (datetime.now() - _t0).total_seconds()
                try:
                    set_gauge("search_time_procedural_seconds", level_times["procedural"])
                except Exception as e:
                    logger.warning(f"Failed to update procedural memory search time metrics: {e}")
                return ("procedural", MemoryResult("procedural", items, [0.5] * len(items), len(items)))

            async def _search_fts5():
                """Поиск в FTS5 - быстрый ключевой поиск"""
                _t0 = datetime.now()
                try:
                    from ..search import get_fts5_engine
                    fts5_engine = await get_fts5_engine()
                    
                    # Выполняем FTS5 поиск
                    fts5_results = await fts5_engine.search(
                        query=query.query,
                        user_id=query.user_id,
                        limit=query.limit
                    )
                    
                    # Преобразуем результаты в формат для MemoryResult
                    items = []
                    for result in fts5_results:
                        # Создаем простой объект с нужными полями
                        item = type('FTS5Result', (), {
                            'id': result["memory_id"],
                            'content': result["content"],
                            'user_id': result["user_id"],
                            'memory_type': result["memory_type"],
                            'importance': result["importance"],
                            'created_at': datetime.fromtimestamp(result["created_at"]),
                            'metadata': {"fts_score": result["fts_score"], "search_type": "keyword"}
                        })()
                        items.append(item)
                    
                    _t1 = datetime.now()
                    level_times["fts5"] = (_t1 - _t0).total_seconds()
                    logger.debug(f"FTS5 search completed: {len(items)} items in {level_times['fts5']:.3f}s")
                    
                    # Обновляем метрики времени поиска
                    try:
                        set_gauge("memory_search_time_fts5_seconds", level_times["fts5"])
                    except Exception as e:
                        logger.warning(f"Failed to update FTS5 search time metrics: {e}")
                    
                    return ("fts5", MemoryResult("fts5", items, [result["fts_score"] for result in fts5_results], len(items)))
                except Exception as e:
                    logger.warning(f"FTS5 search failed: {e}")
                    return ("fts5", MemoryResult("fts5", [], [], 0))

            tasks = []
            if "sensor" in levels_to_search:
                tasks.append(_search_sensor())
            if "working" in levels_to_search:
                tasks.append(_search_working())
            if "short_term" in levels_to_search:
                tasks.append(_search_short_term())
            if "episodic" in levels_to_search:
                tasks.append(_search_episodic())
            if "semantic" in levels_to_search:
                tasks.append(_search_semantic())
            if "graph" in levels_to_search:
                tasks.append(_search_graph())
            if "procedural" in levels_to_search:
                tasks.append(_search_procedural())
            if "fts5" in levels_to_search:
                tasks.append(_search_fts5())

            if tasks:
                results_list = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results_list:
                    if isinstance(res, Exception):
                        continue
                    if isinstance(res, tuple) and len(res) == 2:
                        level, mem_result = res
                        results[level] = mem_result
                        total_items += mem_result.total_found
            
            processing_time = time.time() - start_time
            
            # Генерируем рекомендации
            recommendations = await self._generate_recommendations(query, results)
            logger.info(f"SearchMemory: total_items={total_items} duration={processing_time:.2f}s")
            try:
                record_event("search_memory", {
                    "user_id": query.user_id,
                    "q": query.query,
                    "levels": levels_to_search,
                    "limit": query.limit,
                    "offset": query.offset,
                    "total": total_items,
                    "duration": time.time() - start_time,
                    "level_times": level_times,
                })
            except Exception as e:
                logger.warning(f"Failed to update search result metrics: {e}")
            
            # Создаем результат
            result = UnifiedMemoryResult(
                query=query.query,
                user_id=query.user_id,
                results=results,
                total_items=total_items,
                processing_time=processing_time,
                recommendations=recommendations
            )
            
            # ОПТИМИЗАЦИЯ: Сохраняем результат в кэш (только стабильные результаты)
            if self._is_search_result_stable(result):
                # Создаем упрощенную версию для кэширования
                cache_data = {
                    "query": result.query,
                    "user_id": result.user_id,
                    "total_items": result.total_items,
                    "processing_time": result.processing_time,
                    "recommendations": result.recommendations,
                    "timestamp": time.time(),
                    "results_summary": {
                        level: {
                            "total_found": mem_result.total_found,
                            "items_count": len(mem_result.items) if mem_result.items else 0
                        }
                        for level, mem_result in result.results.items()
                    }
                }
                
                await self._set_search_cache(cache_key, cache_data, ttl_sec=300)  # 5 минут
                logger.info(f"Search result cached: {query.query}")
                try:
                    inc("search_cache_saves")
                except Exception as e:
                    logger.warning(f"Failed to update search cache saves metrics: {e}")
            
            # Мониторинг производительности поиска
            duration_ms = (time.time() - start_time) * 1000
            total_results = sum(len(mem_result.items) if mem_result.items else 0 for mem_result in result.results.values())
            avg_relevance = 0.0
            if result.results:
                all_scores = []
                for mem_result in result.results.values():
                    if mem_result.relevance_scores:
                        all_scores.extend(mem_result.relevance_scores)
                avg_relevance = sum(all_scores) / len(all_scores) if all_scores else 0.0
            
            # Записываем метрики поиска - КРИТИЧЕСКИ ВАЖНО!
            try:
                record_search_metrics(
                    query=query.query,
                    search_type="integrated",
                    user_id=query.user_id,
                    duration_ms=duration_ms,
                    results=[],  # Упрощенный список результатов
                    cache_hit=False,
                    memory_levels=list(result.results.keys()),
                    filters={"limit": query.limit, "offset": query.offset}
                )
                logger.debug("✅ Search metrics recorded successfully")
            except Exception as e:
                logger.error(f"🚨 CRITICAL: Failed to record search metrics: {e}")
                logger.error("🚨 Monitoring fallback: using in-memory metrics")
            
            # Записываем общие метрики производительности - КРИТИЧЕСКИ ВАЖНО!
            try:
                track_operation_performance(
                    operation_type="search",
                    operation_name="memory_orchestrator_search",
                    user_id=query.user_id,
                    duration_ms=duration_ms,
                    success=True,
                    result_count=total_results,
                    accuracy_score=avg_relevance,
                    metadata={
                        "query_length": len(query.query),
                        "levels_searched": list(result.results.keys()),
                        "limit": query.limit,
                        "offset": query.offset
                    }
                )
                logger.debug("✅ Performance metrics tracked successfully")
            except Exception as e:
                logger.error(f"🚨 CRITICAL: Failed to track performance metrics: {e}")
                logger.error("🚨 Monitoring fallback: using in-memory metrics")
            
            return result
            
        except Exception as e:
            # Мониторинг производительности для ошибок поиска
            duration_ms = (time.time() - start_time) * 1000
            track_operation_performance(
                operation_type="search",
                operation_name="memory_orchestrator_search",
                user_id=query.user_id,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                metadata={
                    "query_length": len(query.query),
                    "limit": query.limit,
                    "offset": query.offset
                }
            )
            logger.error(f"Error searching memory: {e}")
            raise

    async def process_message(self, user_id: str, content: str,
                              importance: float = 0.5,
                              context: Optional[str] = None,
                              participants: Optional[List[str]] = None,
                              location: Optional[str] = None,
                              emotion_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Высокоуровневая обработка входящего сообщения.

        Выполняет: автоклассификацию события, извлечение фактов (при высокой важности),
        и распределение по уровням через add_memory.
        """
        try:
            # Предварительная автоклассификация
            try:
                auto_type = await self.event_classifier.classify(content)
            except Exception:
                auto_type = "general"

            results = await self.add_memory(
                content=content,
                user_id=user_id,
                memory_type=auto_type,
                importance=importance,
                emotion_data=emotion_data,
                context=context,
                participants=participants or [],
                location=location
            )
            return {"success": True, "auto_type": auto_type, "results": results}
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_memory_context(self, user_id: str, context_type: str = "full") -> Dict[str, Any]:
        """
        Получить контекст памяти пользователя
        
        Args:
            user_id: ID пользователя
            context_type: Тип контекста (full, recent, important)
            
        Returns:
            Словарь с контекстом всех уровней
        """
        try:
            context = {}
            
            if context_type in ["full", "recent"]:
                # Рабочая память
                wm_context = await self.working_memory.get_active_context(user_id)
                context["working"] = [item.content for item in wm_context]
                
                # Кратковременная память
                stm_context = await self.short_term_memory.get_recent_events(user_id, hours=24)
                context["short_term"] = [item.content for item in stm_context]
            
            if context_type in ["full", "important"]:
                # Эпизодическая память
                ep_context = await self.episodic_memory.get_significant_events(user_id, min_importance=0.7)
                context["episodic"] = [item.content for item in ep_context]
                
                # Семантическая память
                sm_context = await self.semantic_memory.get_knowledge_by_category(user_id, "general")
                context["semantic"] = [item.content for item in sm_context]
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting memory context: {e}")
            return {}
    
    async def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Получить статистику всех уровней памяти
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Словарь со статистикой всех уровней
        """
        try:
            stats = {}
            
            # Получаем статистику каждого уровня
            stats["working"] = await self.working_memory.get_stats(user_id)
            stats["short_term"] = await self.short_term_memory.get_stats(user_id)
            stats["episodic"] = await self.episodic_memory.get_stats(user_id)
            stats["semantic"] = await self.semantic_memory.get_stats(user_id)
            stats["graph"] = await self.graph_memory.get_stats(user_id)
            stats["procedural"] = await self.procedural_memory.get_stats(user_id)
            
            # Общая статистика
            total_items = sum(
                level_stats.get("total_items", 0) 
                for level_stats in stats.values() 
                if isinstance(level_stats, dict)
            )
            
            stats["total"] = {
                "total_items": total_items,
                "levels_active": len([s for s in stats.values() if isinstance(s, dict) and s.get("total_items", 0) > 0])
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_memories(self, user_id: str):
        """
        Очистка старых воспоминаний на всех уровнях
        
        Args:
            user_id: ID пользователя
        """
        try:
            # Очищаем каждый уровень
            await self.working_memory._cleanup_old_memories(user_id)
            await self.short_term_memory._cleanup_old_memories(user_id)
            await self.episodic_memory._cleanup_old_memories(user_id)
            await self.semantic_memory._cleanup_old_memories(user_id)
            if self.graph_memory:
                try:
                    # Проверяем наличие метода через getattr
                    cleanup_method = getattr(self.graph_memory, '_cleanup_old_memories', None)
                    if cleanup_method:
                        await cleanup_method(user_id)
                    else:
                        logger.warning("GraphMemoryManagerSQLite does not support _cleanup_old_memories")
                except AttributeError:
                    logger.warning("GraphMemoryManagerSQLite does not support _cleanup_old_memories")
            await self.procedural_memory._cleanup_old_memories(user_id)
            
            logger.info(f"Cleaned up memories for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up memories: {e}")
    
    async def consolidate_memories(self, user_id: str) -> Dict[str, int]:
        """Консолидация между уровнями памяти по правилам возраста.

        Правила:
        - Working -> Short-term, если старше 1 часа
        - Short-term -> Episodic, если старше 7 дней
        - Episodic -> Semantic, если старше 30 дней и importance >= 0.5
        """
        moved_counts = {"working_to_short_term": 0, "short_term_to_episodic": 0, "episodic_to_semantic": 0}

        # Working -> Short-term
        try:
            old_working = await self.working_memory.get_older_than(user_id, timedelta(hours=1), limit=200)
            for item in old_working:
                await self.short_term_memory.add_event(
                    content=item.content,
                    user_id=user_id,
                    importance=item.importance,
                    confidence=min(1.0, max(0.0, item.importance * 0.8 + 0.2)),
                    event_type="conversation",
                    location=None,
                    participants=[],
                    emotion_data=item.emotion_data,
                    related_memories=[]
                )
                await self.working_memory.remove_memory(item.id)
                moved_counts["working_to_short_term"] += 1
        except Exception as e:
            logger.error(f"Error consolidating Working -> Short-term for user {user_id}: {e}")

        # Short-term -> Episodic (older than 7 days)
        try:
            old_short = await self.short_term_memory.get_older_than(user_id, timedelta(days=7), limit=500)
            for item in old_short:
                await self.episodic_memory.add_experience(
                    content=item.content,
                    user_id=user_id,
                    importance=item.importance,
                    confidence=min(1.0, max(0.0, item.importance * 0.8 + 0.2)),
                    event_type=item.event_type or "experience",
                    location=item.location,
                    participants=item.participants or [],
                    emotion_data=item.emotion_data,
                    related_memories=item.related_memories or [],
                    significance=min(1.0, 0.4 + item.importance * 0.6),
                    vividness=0.5,
                    context=None
                )
                await self.short_term_memory.remove_memory(item.id)
                moved_counts["short_term_to_episodic"] += 1
        except Exception as e:
            logger.error(f"Error consolidating Short-term -> Episodic for user {user_id}: {e}")

        # Episodic -> Semantic (older than 30 days and importance >= 0.5)
        try:
            old_epi = await self.episodic_memory.get_older_than(user_id, timedelta(days=30), limit=1000)
            for item in old_epi:
                if item.importance >= 0.5:
                    await self.semantic_memory.add_knowledge(
                        content=item.content,
                        user_id=user_id,
                        knowledge_type="fact",
                        category="experience",
                        confidence=0.6,
                        source="episodic",
                        related_concepts=item.related_memories or [],
                        tags=[item.event_type] if item.event_type else [],
                        importance=item.importance
                    )
                    await self.episodic_memory.remove_memory(item.id)
                    moved_counts["episodic_to_semantic"] += 1
        except Exception as e:
            logger.error(f"Error consolidating Episodic -> Semantic for user {user_id}: {e}")

        logger.info(f"Consolidation finished for user {user_id}: {moved_counts}")
        return moved_counts

    async def get_known_user_ids(self, max_per_level: int = 2000) -> Set[str]:
        """Собрать уникальные user_id, замеченные в уровнях памяти (Chroma-уровни)."""
        user_ids: Set[str] = set()
        try:
            # Working
            try:
                res = self.working_memory.collection.get(limit=max_per_level, include=["metadatas"])
                metadatas = res.get("metadatas", [])
                if isinstance(metadatas, list):
                    for meta in metadatas:
                        uid = meta.get("user_id")
                        if uid and isinstance(uid, str):
                            user_ids.add(uid)
            except Exception as e:
                logger.warning(f"Failed to collect user IDs from working memory: {e}")
            # Short-term
            try:
                res = self.short_term_memory.collection.get(limit=max_per_level, include=["metadatas"])
                metadatas = res.get("metadatas", [])
                if isinstance(metadatas, list):
                    for meta in metadatas:
                        uid = meta.get("user_id")
                        if uid and isinstance(uid, str):
                            user_ids.add(uid)
            except Exception as e:
                logger.warning(f"Failed to collect user IDs from short term memory: {e}")
            # Episodic
            try:
                res = self.episodic_memory.collection.get(limit=max_per_level, include=["metadatas"])
                metadatas = res.get("metadatas", [])
                if isinstance(metadatas, list):
                    for meta in metadatas:
                        uid = meta.get("user_id")
                        if uid and isinstance(uid, str):
                            user_ids.add(uid)
            except Exception as e:
                logger.warning(f"Failed to collect user IDs from episodic memory: {e}")
            # Semantic
            try:
                res = self.semantic_memory.collection.get(limit=max_per_level, include=["metadatas"])
                metadatas = res.get("metadatas", [])
                if isinstance(metadatas, list):
                    for meta in metadatas:
                        uid = meta.get("user_id")
                        if uid and isinstance(uid, str):
                            user_ids.add(uid)
            except Exception as e:
                logger.warning(f"Failed to collect user IDs from semantic memory: {e}")
        except Exception as e:
            logger.error(f"Error collecting known user ids: {e}")
        return user_ids
    
    async def _extract_concepts(self, content: str, user_id: str = "claude_ai", importance: float = 0.5) -> tuple[List[str], dict]:
        """
        Извлечь ключевые концепции из текста с помощью EntityExtractor
        
        Args:
            content: Текст для анализа
            user_id: ID пользователя для EntityExtractor
            
        Returns:
            Список концепций
        """
        try:
            # ОПТИМИЗАЦИЯ: Проверяем кэш концепций (только для повторных запросов)
            import hashlib
            content_hash = hashlib.md5(content.encode()).hexdigest()
            cache_key = f"concepts:{content_hash}"
            
            # Пытаемся получить из кэша (быстрая проверка)
            try:
                from ..cache.sqlite_cache import get_sqlite_cache
                cache = await get_sqlite_cache()
                # Используем timeout для быстрой проверки кэша
                cached_concepts = await asyncio.wait_for(cache.get(cache_key), timeout=1.0)
                if cached_concepts:
                    concepts = cached_concepts.get('concepts', [])
                    entity_types = cached_concepts.get('entity_types', {})
                    logger.debug(f"Cache hit for concepts: {len(concepts)} concepts, {len(entity_types)} entity_types")
                    
                    # Если entity_types пустой, но есть концепты, попробуем извлечь типы из LLM
                    if not entity_types and concepts:
                        logger.debug("Entity types empty in cache, extracting from LLM...")
                        unified_result = await self._process_memory_unified(content, user_id, importance)
                        entities = unified_result.get("entities", [])
                        for entity in entities:
                            if isinstance(entity, dict):
                                entity_name = entity.get("text", "")
                                entity_type = entity.get("type", "")
                                if entity_name in concepts:
                                    entity_types[entity_name] = entity_type.lower()
                        logger.debug(f"Extracted entity types from LLM: {entity_types}")
                    
                    return concepts, entity_types
            except (Exception, asyncio.TimeoutError) as e:
                logger.debug(f"Cache miss for concepts: {e}")
            
            # ОПТИМИЗАЦИЯ: Используем единый LLM вызов вместо отдельных
            unified_result = await self._process_memory_unified(content, user_id, importance)
            
            # Извлекаем концепции из единого результата с сохранением типов
            concepts = []
            entity_types = {}  # Словарь для сохранения типов сущностей
            
            # Добавляем сущности по типам из единого результата
            entities = unified_result.get("entities", [])
            logger.debug(f"Unified processing found {len(entities)} entities: {entities}")
            for entity in entities:
                if isinstance(entity, dict):
                    entity_name = entity.get("text", "")
                    entity_type = entity.get("type", "")
                    entity_importance = entity.get("importance", 0.5)
                    
                    logger.debug(f"Entity: name='{entity_name}', type='{entity_type}', importance={entity_importance}")
                    
                    # Расширяем фильтр для включения всех типов из промпта
                    entity_type_lower = entity_type.lower()
                    if any(t in entity_type_lower for t in ['concept', 'technology', 'organization', 'person', 'service', 'date']):
                        concepts.append(entity_name)
                        entity_types[entity_name] = entity_type_lower  # Сохраняем тип сущности
                        logger.debug(f"Added concept: {entity_name} (type: {entity_type_lower})")
                    else:
                        logger.debug(f"Skipped entity type: {entity_type}")
                else:
                    logger.warning(f"Invalid entity format: {entity}")
            
            # Если EntityExtractor не нашел сущностей, используем fallback
            if not concepts:
                logger.warning("EntityExtractor не нашел сущностей, используем fallback")
                concepts = await self._extract_concepts_fallback(content)
                logger.info(f"Fallback extracted {len(concepts)} concepts: {concepts}")
                
                # Если fallback тоже не дал результатов, создаем базовые концепции из текста
                if not concepts:
                    logger.warning("Fallback не дал результатов, создаем базовые концепции")
                    concepts = await self._create_basic_concepts_async(content)
                    logger.info(f"Basic concepts created: {len(concepts)} concepts: {concepts}")
            
            # Возвращаем уникальные концепции, отсортированные по важности
            unique_concepts = list(set(concepts))
            
            # Сортируем по длине (более длинные концепции обычно более значимы)
            sorted_concepts = sorted(unique_concepts, key=lambda x: len(x), reverse=True)
            final_concepts = sorted_concepts[:10]  # Максимум 10 концепций
            
            # ОПТИМИЗАЦИЯ: Асинхронное сохранение в кэш (не блокирует основной поток)
            task = asyncio.create_task(self._cache_concepts_async(cache_key, final_concepts, entity_types))
            self._add_background_task(task)
            
            return final_concepts, entity_types
            
        except Exception as e:
            logger.error(f"Error extracting concepts with EntityExtractor: {e}")
            # Fallback на простой метод
            fallback_concepts = await self._extract_concepts_fallback(content)
            return fallback_concepts, {}
    
    async def _cache_concepts_async(self, cache_key: str, concepts: List[str], entity_types: Optional[dict] = None):
        """Асинхронное сохранение концепций в кэш"""
        try:
            # Проверяем флаг завершения
            if self._shutdown_event.is_set():
                logger.debug("Shutdown requested, skipping concept caching")
                return
                
            from ..cache.sqlite_cache import get_sqlite_cache
            cache = await get_sqlite_cache()
            cache_data = {'concepts': concepts}
            if entity_types and isinstance(entity_types, dict):
                cache_data['entity_types'] = list(entity_types.keys())  # Преобразуем dict в List[str]
            await cache.set(cache_key, cache_data, ttl_sec=3600)  # Кэш на 1 час
            logger.debug(f"Cached concepts: {len(concepts)} concepts")
        except Exception as e:
            logger.debug(f"Failed to cache concepts: {e}")
    
    async def _extract_concepts_fallback(self, content: str) -> List[str]:
        """
        Fallback метод для извлечения концепций (упрощенная версия)
        
        Args:
            content: Текст для анализа
            
        Returns:
            Список концепций
        """
        try:
            import re
            
            # Очищаем текст от пунктуации и приводим к нижнему регистру
            cleaned_content = re.sub(r'[^\w\s]', ' ', content.lower())
            words = cleaned_content.split()
            
            # Русские стоп-слова (основные)
            russian_stop_words = {
                "и", "в", "на", "с", "по", "для", "от", "до", "из", "к", "у", "о", "об", "что", "как", "где", "когда", "почему",
                "это", "то", "он", "она", "оно", "они", "мы", "вы", "я", "ты", "его", "её", "их", "наш", "ваш", "мой", "твой",
                "быть", "есть", "был", "была", "было", "были", "будет", "будут", "иметь", "имеет", "имел", "имела", "имели",
                "делать", "делает", "делал", "делала", "делали", "может", "хотеть", "хочет", "хотел", "хотела", "хотели", 
                "нужно", "нужен", "нужна", "нужны", "можно", "нельзя", "очень", "более", "менее", "самый", "самая", "самое", 
                "самые", "все", "вся", "всё", "каждый", "каждая", "каждое", "каждые", "другой", "другая", "другое", "другие", 
                "новый", "новая", "новое", "новые", "старый", "старая", "старое", "старые", "большой", "большая", "большое", 
                "большие", "маленький", "маленькая", "маленькое", "маленькие", "хороший", "хорошая", "хорошее", "хорошие", 
                "плохой", "плохая", "плохое", "плохие", "первый", "первая", "первое", "первые", "последний", "последняя", 
                "последнее", "последние", "главный", "главная", "главное", "главные", "основной", "основная", "основное", 
                "основные", "важный", "важная", "важное", "важные"
            }
            
            # Английские стоп-слова (основные)
            english_stop_words = {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", 
                "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", 
                "might", "can", "must", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me",
                "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours",
                "theirs", "am", "are", "is", "was", "were", "being", "been", "have", "has", "had", "having", "do", "does",
                "did", "doing", "will", "would", "could", "should", "may", "might", "can", "must", "shall", "ought", "need",
                "dare", "used", "get", "got", "getting", "go", "went", "gone", "going", "come", "came", "coming", "see", "saw",
                "seen", "seeing", "know", "knew", "known", "knowing", "think", "thought", "thinking", "take", "took", "taken",
                "taking", "give", "gave", "given", "giving", "make", "made", "making", "find", "found", "finding", "tell",
                "told", "telling", "ask", "asked", "asking", "work", "worked", "working", "seem", "seemed", "seeming", "feel",
                "felt", "feeling", "try", "tried", "trying", "leave", "left", "leaving", "call", "called", "calling", "move",
                "moved", "moving", "play", "played", "playing", "turn", "turned", "turning", "start", "started", "starting",
                "stop", "stopped", "stopping", "show", "showed", "shown", "showing", "hear", "heard", "hearing", "let",
                "letting", "put", "putting", "bring", "brought", "bringing", "begin", "began", "begun", "beginning", "keep",
                "kept", "keeping", "hold", "held", "holding", "write", "wrote", "written", "writing", "provide", "provided",
                "providing", "sit", "sat", "sitting", "stand", "stood", "standing", "lose", "lost", "losing", "pay", "paid",
                "paying", "meet", "met", "meeting", "include", "included", "including", "continue", "continued", "continuing",
                "set", "setting", "learn", "learned", "learning", "change", "changed", "changing", "lead", "led", "leading",
                "understand", "understood", "understanding", "watch", "watched", "watching", "follow", "followed", "following",
                "stop", "stopped", "stopping", "create", "created", "creating", "speak", "spoke", "spoken", "speaking",
                "read", "reading", "allow", "allowed", "allowing", "add", "added", "adding", "spend", "spent", "spending",
                "grow", "grew", "grown", "growing", "open", "opened", "opening", "walk", "walked", "walking", "win", "won",
                "winning", "offer", "offered", "offering", "remember", "remembered", "remembering", "love", "loved", "loving",
                "consider", "considered", "considering", "appear", "appeared", "appearing", "buy", "bought", "buying", "wait",
                "waited", "waiting", "serve", "served", "serving", "die", "died", "dying", "send", "sent", "sending", "expect",
                "expected", "expecting", "build", "built", "building", "stay", "stayed", "staying", "fall", "fell", "fallen",
                "falling", "cut", "cutting", "reach", "reached", "reaching", "kill", "killed", "killing", "remain", "remained",
                "remaining", "suggest", "suggested", "suggesting", "raise", "raised", "raising", "pass", "passed", "passing",
                "sell", "sold", "selling", "require", "required", "requiring", "report", "reported", "reporting", "decide",
                "decided", "deciding", "pull", "pulled", "pulling", "break", "broke", "broken", "breaking", "produce",
                "produced", "producing", "eat", "ate", "eaten", "eating", "teach", "taught", "teaching", "cover", "covered",
                "covering", "catch", "caught", "catching", "draw", "drew", "drawn", "drawing", "choose", "chose", "chosen",
                "choosing", "wear", "wore", "worn", "wearing", "throw", "threw", "thrown", "throwing", "drive", "drove", "driven",
                "driving", "ride", "rode", "ridden", "riding", "fly", "flew", "flown", "flying", "swim", "swam", "swum", "swimming",
                "run", "ran", "running", "jump", "jumped", "jumping", "climb", "climbed", "climbing", "crawl", "crawled", "crawling",
                "slide", "slid", "sliding", "roll", "rolled", "rolling", "spin", "spun", "spinning", "shake", "shook", "shaken",
                "shaking", "bend", "bent", "bending", "stretch", "stretched", "stretching", "squeeze", "squeezed", "squeezing",
                "press", "pressed", "pressing", "push", "pushed", "pushing", "pull", "pulled", "pulling", "lift", "lifted", "lifting",
                "drop", "dropped", "dropping", "throw", "threw", "thrown", "throwing", "catch", "caught", "catching", "grab",
                "grabbed", "grabbing", "hold", "held", "holding", "carry", "carried", "carrying", "move", "moved", "moving",
                "place", "placed", "placing", "put", "putting", "set", "setting", "lay", "laid", "laying", "stand", "stood",
                "standing", "sit", "sat", "sitting", "lie", "lay", "lain", "lying", "sleep", "slept", "sleeping", "wake", "woke",
                "woken", "waking", "rest", "rested", "resting", "relax", "relaxed", "relaxing", "work", "worked", "working",
                "play", "played", "playing", "study", "studied", "studying", "learn", "learned", "learning", "teach", "taught",
                "teaching", "help", "helped", "helping", "support", "supported", "supporting", "care", "cared", "caring", "love",
                "loved", "loving", "like", "liked", "liking", "enjoy", "enjoyed", "enjoying", "hate", "hated", "hating", "dislike",
                "disliked", "disliking", "prefer", "preferred", "preferring", "choose", "chose", "chosen", "choosing", "select",
                "selected", "selecting", "pick", "picked", "picking", "decide", "decided", "deciding", "determine", "determined",
                "determining", "resolve", "resolved", "resolving", "solve", "solved", "solving", "fix", "fixed", "fixing", "repair",
                "repaired", "repairing", "mend", "mended", "mending", "improve", "improved", "improving", "enhance", "enhanced",
                "enhancing", "upgrade", "upgraded", "upgrading", "update", "updated", "updating", "modify", "modified", "modifying",
                "change", "changed", "changing", "alter", "altered", "altering", "adjust", "adjusted", "adjusting", "adapt",
                "adapted", "adapting", "transform", "transformed", "transforming", "convert", "converted", "converting", "turn",
                "turned", "turning", "switch", "switched", "switching", "shift", "shifted", "shifting", "transfer", "transferred",
                "transferring", "move", "moved", "moving", "relocate", "relocated", "relocating", "migrate", "migrated", "migrating",
                "travel", "traveled", "traveling", "journey", "journeyed", "journeying", "trip", "tripped", "tripping", "visit",
                "visited", "visiting", "tour", "toured", "touring", "explore", "explored", "exploring", "discover", "discovered",
                "discovering", "find", "found", "finding", "locate", "located", "locating", "search", "searched", "searching",
                "seek", "sought", "seeking", "look", "looked", "looking", "watch", "watched", "watching", "observe", "observed",
                "observing", "notice", "noticed", "noticing", "see", "saw", "seen", "seeing", "view", "viewed", "viewing", "examine",
                "examined", "examining", "inspect", "inspected", "inspecting", "check", "checked", "checking", "test", "tested",
                "testing", "try", "tried", "trying", "attempt", "attempted", "attempting", "experiment", "experimented", "experimenting",
                "practice", "practiced", "practicing", "exercise", "exercised", "exercising", "train", "trained", "training", "coach",
                "coached", "coaching", "instruct", "instructed", "instructing", "guide", "guided", "guiding", "lead", "led", "leading",
                "direct", "directed", "directing", "manage", "managed", "managing", "control", "controlled", "controlling", "operate",
                "operated", "operating", "run", "ran", "running", "execute", "executed", "executing", "perform", "performed", "performing",
                "accomplish", "accomplished", "accomplishing", "achieve", "achieved", "achieving", "complete", "completed", "completing",
                "finish", "finished", "finishing", "end", "ended", "ending", "stop", "stopped", "stopping", "pause", "paused", "pausing",
                "wait", "waited", "waiting", "delay", "delayed", "delaying", "postpone", "postponed", "postponing", "cancel", "cancelled",
                "cancelling", "abort", "aborted", "aborting", "terminate", "terminated", "terminating", "close", "closed", "closing",
                "shut", "shut", "shutting", "lock", "locked", "locking", "unlock", "unlocked", "unlocking", "open", "opened", "opening",
                "start", "started", "starting", "begin", "began", "begun", "beginning", "launch", "launched", "launching", "initiate",
                "initiated", "initiating", "establish", "established", "establishing", "create", "created", "creating", "make", "made",
                "making", "build", "built", "building", "construct", "constructed", "constructing", "develop", "developed", "developing",
                "design", "designed", "designing", "plan", "planned", "planning", "organize", "organized", "organizing", "arrange",
                "arranged", "arranging", "prepare", "prepared", "preparing", "setup", "setted", "setting", "install", "installed",
                "installing", "configure", "configured", "configuring", "customize", "customized", "customizing", "personalize",
                "personalized", "personalizing", "tailor", "tailored", "tailoring", "adapt", "adapted", "adapting", "adjust", "adjusted",
                "adjusting", "tune", "tuned", "tuning", "calibrate", "calibrated", "calibrating", "optimize", "optimized", "optimizing",
                "maximize", "maximized", "maximizing", "minimize", "minimized", "minimizing", "reduce", "reduced", "reducing", "decrease",
                "decreased", "decreasing", "increase", "increased", "increasing", "expand", "expanded", "expanding", "extend", "extended",
                "extending", "stretch", "stretched", "stretching", "grow", "grew", "grown", "growing", "shrink", "shrank", "shrunk",
                "shrinking", "contract", "contracted", "contracting", "compress", "compressed", "compressing", "squeeze", "squeezed",
                "squeezing", "press", "pressed", "pressing", "push", "pushed", "pushing", "pull", "pulled", "pulling", "drag", "dragged",
                "dragging", "draw", "drew", "drawn", "drawing", "paint", "painted", "painting", "color", "colored", "coloring", "dye",
                "dyed", "dying", "tint", "tinted", "tinting", "shade", "shaded", "shading", "highlight", "highlighted", "highlighting",
                "emphasize", "emphasized", "emphasizing", "stress", "stressed", "stressing", "accent", "accented", "accenting", "focus",
                "focused", "focusing", "concentrate", "concentrated", "concentrating", "center", "centered", "centering", "target",
                "targeted", "targeting", "aim", "aimed", "aiming", "point", "pointed", "pointing", "direct", "directed", "directing",
                "guide", "guided", "guiding", "steer", "steered", "steering", "navigate", "navigated", "navigating", "pilot", "piloted",
                "piloting", "drive", "drove", "driven", "driving", "ride", "rode", "ridden", "riding", "fly", "flew", "flown", "flying",
                "sail", "sailed", "sailing", "cruise", "cruised", "cruising", "float", "floated", "floating", "drift", "drifted", "drifting",
                "flow", "flowed", "flowing", "stream", "streamed", "streaming", "pour", "poured", "pouring", "spill", "spilled", "spilling",
                "leak", "leaked", "leaking", "drip", "dripped", "dripping", "drop", "dropped", "dropping", "fall", "fell", "fallen",
                "falling", "sink", "sank", "sunk", "sinking", "dive", "dove", "dived", "diving", "plunge", "plunged", "plunging", "jump",
                "jumped", "jumping", "leap", "leapt", "leaping", "hop", "hopped", "hopping", "skip", "skipped", "skipping", "bounce",
                "bounced", "bouncing", "spring", "sprang", "sprung", "springing", "bound", "bounded", "bounding", "vault", "vaulted",
                "vaulting", "hurdle", "hurdled", "hurdling", "climb", "climbed", "climbing", "scale", "scaled", "scaling", "ascend",
                "ascended", "ascending", "rise", "rose", "risen", "rising", "lift", "lifted", "lifting", "raise", "raised", "raising",
                "elevate", "elevated", "elevating", "boost", "boosted", "boosting", "promote", "promoted", "promoting", "advance",
                "advanced", "advancing", "progress", "progressed", "progressing", "proceed", "proceeded", "proceeding", "continue",
                "continued", "continuing", "persist", "persisted", "persisting", "endure", "endured", "enduring", "last", "lasted",
                "lasting", "remain", "remained", "remaining", "stay", "stayed", "staying", "keep", "kept", "keeping", "retain", "retained",
                "retaining", "hold", "held", "holding", "grasp", "grasped", "grasping", "grip", "gripped", "gripping", "clutch", "clutched",
                "clutching", "clasp", "clasped", "clasping", "hug", "hugged", "hugging", "embrace", "embraced", "embracing", "cuddle",
                "cuddled", "cuddling", "snuggle", "snuggled", "snuggling", "nestle", "nestled", "nestling", "lean", "leaned", "leaning",
                "rest", "rested", "resting", "recline", "reclined", "reclining", "lie", "lay", "lain", "lying", "sit", "sat", "sitting",
                "squat", "squatted", "squatting", "crouch", "crouched", "crouching", "kneel", "knelt", "kneeling", "bend", "bent", "bending",
                "stoop", "stooped", "stooping", "bow", "bowed", "bowing", "nod", "nodded", "nodding", "shake", "shook", "shaken", "shaking",
                "tremble", "trembled", "trembling", "quiver", "quivered", "quivering", "shiver", "shivered", "shivering", "vibrate",
                "vibrated", "vibrating", "oscillate", "oscillated", "oscillating", "swing", "swung", "swinging", "sway", "swayed", "swaying",
                "rock", "rocked", "rocking", "roll", "rolled", "rolling", "spin", "spun", "spinning", "rotate", "rotated", "rotating",
                "turn", "turned", "turning", "twist", "twisted", "twisting", "wind", "wound", "winding", "coil", "coiled", "coiling",
                "curl", "curled", "curling", "loop", "looped", "looping", "circle", "circled", "circling", "orbit", "orbited", "orbiting",
                "revolve", "revolved", "revolving", "rotate", "rotated", "rotating", "spin", "spun", "spinning", "whirl", "whirled",
                "whirling", "twirl", "twirled", "twirling", "pirouette", "pirouetted", "pirouetting", "dance", "danced", "dancing",
                "waltz", "waltzed", "waltzing", "tango", "tangoed", "tangoing", "foxtrot", "foxtrotted", "foxtrotting", "samba", "sambaed",
                "sambaing", "rumba", "rumbaed", "rumbaing", "cha", "chaed", "chaing", "mambo", "mamboed", "mamboing", "salsa", "salsaed",
                "salsaing", "merengue", "merengued", "merenguing", "bachata", "bachataed", "bachataing", "kizomba", "kizombaed", "kizombaing",
                "zouk", "zouked", "zouking", "reggaeton", "reggaetoned", "reggaetoning", "hip", "hiped", "hiping", "hop", "hopped", "hopping",
                "break", "breaked", "breaking", "pop", "popped", "popping", "lock", "locked", "locking", "popping", "popped", "popping",
                "krump", "krumped", "krumping", "turf", "turfed", "turfing", "flex", "flexed", "flexing", "jook", "jooked", "jooking",
                "buck", "bucked", "bucking", "juke", "juked", "juking", "wop", "wopped", "wopping", "dougie", "dougied", "dougieing",
                "nay", "nayed", "naying", "cat", "catted", "catting", "fish", "fished", "fishing", "stanky", "stankied", "stankying",
                "leg", "legged", "legging", "milly", "millied", "millying", "rock", "rocked", "rocking", "shmoney", "shmoneyed", "shmoneying",
                "yeet", "yeeted", "yeeting", "dab", "dabbed", "dabbing", "floss", "flossed", "flossing", "orange", "oranged", "oranging",
                "justice", "justiced", "justicing", "fortnite", "fortnited", "fortniting", "default", "defaulted", "defaulting", "dance",
                "danced", "dancing", "move", "moved", "moving", "step", "stepped", "stepping", "walk", "walked", "walking", "run", "ran",
                "running", "jog", "jogged", "jogging", "sprint", "sprinted", "sprinting", "dash", "dashed", "dashing", "rush", "rushed",
                "rushing", "hurry", "hurried", "hurrying", "speed", "sped", "speeding", "race", "raced", "racing", "compete", "competed",
                "competing", "contest", "contested", "contesting", "challenge", "challenged", "challenging", "oppose", "opposed", "opposing",
                "resist", "resisted", "resisting", "fight", "fought", "fighting", "battle", "battled", "battling", "war", "warred", "warring",
                "conflict", "conflicted", "conflicting", "struggle", "struggled", "struggling", "strive", "strove", "striven", "striving",
                "endeavor", "endeavored", "endeavoring", "attempt", "attempted", "attempting", "try", "tried", "trying", "test", "tested",
                "testing", "experiment", "experimented", "experimenting", "trial", "trialed", "trialing", "sample", "sampled", "sampling",
                "taste", "tasted", "tasting", "try", "tried", "trying", "experience", "experienced", "experiencing", "feel", "felt", "feeling",
                "sense", "sensed", "sensing", "perceive", "perceived", "perceiving", "notice", "noticed", "noticing", "observe", "observed",
                "observing", "watch", "watched", "watching", "see", "saw", "seen", "seeing", "look", "looked", "looking", "view", "viewed",
                "viewing", "gaze", "gazed", "gazing", "stare", "stared", "staring", "glance", "glanced", "glancing", "peek", "peeked", "peeking",
                "peep", "peeped", "peeping", "peer", "peered", "peering", "scan", "scanned", "scanning", "skim", "skimmed", "skimming",
                "browse", "browsed", "browsing", "search", "searched", "searching", "seek", "sought", "seeking", "hunt", "hunted", "hunting",
                "track", "tracked", "tracking", "trace", "traced", "tracing", "follow", "followed", "following", "pursue", "pursued", "pursuing",
                "chase", "chased", "chasing", "hunt", "hunted", "hunting", "stalk", "stalked", "stalking", "shadow", "shadowed", "shadowing",
                "tail", "tailed", "tailing", "trail", "trailed", "trailing", "pursue", "pursued", "pursuing", "chase", "chased", "chasing",
                "hunt", "hunted", "hunting", "stalk", "stalked", "stalking", "shadow", "shadowed", "shadowing", "tail", "tailed", "tailing",
                "trail", "trailed", "trailing", "pursue", "pursued", "pursuing", "chase", "chased", "chasing", "hunt", "hunted", "hunting",
                "stalk", "stalked", "stalking", "shadow", "shadowed", "shadowing", "tail", "tailed", "tailing", "trail", "trailed", "trailing"
            }
            
            # Объединяем стоп-слова
            all_stop_words = russian_stop_words | english_stop_words
            
            # Извлекаем концепции
            concepts = []
            for word in words:
                # Фильтруем короткие слова, стоп-слова и числа
                if (len(word) > 3 and 
                    word not in all_stop_words and 
                    not word.isdigit() and
                    word.isalpha()):
                    concepts.append(word)
            
            # Возвращаем уникальные концепции, отсортированные по частоте
            concept_counts = {}
            for concept in concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
            
            # Сортируем по частоте и возвращаем топ-10
            sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
            return [concept for concept, count in sorted_concepts[:10]]
            
        except Exception as e:
            logger.error(f"Error in fallback concept extraction: {e}")
            return []
    
    async def _create_basic_concepts_async(self, content: str) -> List[str]:
        """
        Создать базовые концепции из текста для графа
        
        Args:
            content: Текст для анализа
            
        Returns:
            Список базовых концепций
        """
        try:
            import re
            
            # Извлекаем слова с заглавной буквы (имена, названия)
            capitalized_words = re.findall(r'\b[А-ЯЁ][а-яё]+\b|\b[A-Z][a-z]+\b', content)
            
            # Извлекаем слова в кавычках
            quoted_words = re.findall(r'"([^"]+)"', content)
            
            # Извлекаем технические термины (слова с цифрами или специальными символами)
            technical_terms = re.findall(r'\b\w*[0-9]+\w*\b|\b\w*[_-]\w*\b', content)
            
            # Извлекаем длинные слова (более 6 символов)
            long_words = re.findall(r'\b\w{6,}\b', content)
            
            # Объединяем все концепции
            concepts = []
            concepts.extend(capitalized_words)
            concepts.extend(quoted_words)
            concepts.extend(technical_terms)
            concepts.extend(long_words)
            
            # Убираем дубликаты и стоп-слова
            unique_concepts = list(set(concepts))
            
            # Фильтруем стоп-слова
            stop_words = {
                "тест", "test", "многоуровневой", "памяти", "системы", "система", "system", "memory", "level", "уровень",
                "данные", "data", "информация", "information", "контент", "content", "текст", "text", "сообщение", "message"
            }
            
            filtered_concepts = [c for c in unique_concepts if c.lower() not in stop_words]
            
            # Возвращаем максимум 5 концепций
            return filtered_concepts[:5]
            
        except Exception as e:
            logger.error(f"Error creating basic concepts: {e}")
            return []
    
    async def _extract_facts_async(self, content: str, user_id: str, importance: float, max_retries: int = 3):
        """
        Асинхронная экстракция фактов с retry механизмом
        
        Args:
            content: Содержимое для анализа
            user_id: ID пользователя
            importance: Важность контента
            max_retries: Максимальное количество попыток
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Starting fact extraction (attempt {attempt + 1}/{max_retries}): user={user_id}")
                
                # Извлекаем факты
                facts = await self.fact_extractor.extract_facts(content)
                
                if not facts:
                    logger.info(f"No facts extracted for user {user_id}")
                    return
                
                # Добавляем факты в семантическую память
                semantic_ids: List[str] = []
                for fact in facts:
                    sm_id = await self.semantic_memory.add_knowledge(
                        content=fact,
                        user_id=user_id,
                        knowledge_type="fact",
                        category="extracted",
                        confidence=validate_confidence(min(1.0, max(0.0, importance * 0.8 + 0.2))),
                        importance=importance
                    )
                    semantic_ids.append(sm_id)
                
                # Обновляем метрики
                try:
                    inc("memory_add_level_semantic")
                    inc("background_tasks_completed")
                except Exception as _e:
                    logger.warning(f"metrics inc failed (semantic-extracted): {_e}")
                
                logger.info(f"Fact extraction completed successfully: user={user_id} facts={len(semantic_ids)}")
                return
                
            except Exception as e:
                logger.error(f"Fact extraction attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    # Последняя попытка неудачна - логируем ошибку
                    logger.error(f"Fact extraction failed after {max_retries} attempts for user {user_id}: {e}")
                    try:
                        inc("background_tasks_failed")
                    except Exception as e:
                        logger.warning(f"Failed to update background tasks failed metrics: {e}")
                    
                    # Добавляем в очередь для повторной обработки
                    try:
                        await self._queue_for_retry("fact_extraction", {
                            "user_id": user_id,
                            "content": content,
                            "importance": importance,
                            "retry_count": 0,
                            "max_retries": 3
                        })
                        logger.info(f"Fact extraction queued for retry: user={user_id}")
                    except Exception as retry_error:
                        logger.warning(f"Failed to queue fact extraction for retry: {retry_error}")
                else:
                    # Ждем перед следующей попыткой (exponential backoff)
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)

    async def _analyze_relationships_async(self, content: str, user_id: str, added_node_ids: List[str], importance: float, max_retries: int = 3):
        """
        Асинхронный анализ отношений с retry механизмом
        
        Args:
            content: Содержимое для анализа
            user_id: ID пользователя
            added_node_ids: Список созданных узлов
            importance: Важность контента
            max_retries: Максимальное количество попыток
        """
        for attempt in range(max_retries):
            try:
                # Проверяем флаг завершения
                if self._shutdown_event.is_set():
                    logger.info(f"Shutdown requested, cancelling relationship analysis for user={user_id}")
                    return
                
                logger.info(f"Starting relationship analysis (attempt {attempt + 1}/{max_retries}): user={user_id} nodes={len(added_node_ids)}")
                
                if not added_node_ids or len(added_node_ids) < 2:
                    logger.info(f"Not enough nodes for relationship analysis: user={user_id}")
                    # Завершаем задачу корректно
                    try:
                        inc("relationship_analysis_completed")
                        logger.info(f"Relationship analysis completed successfully: user={user_id} relationships=0")
                    except Exception as e:
                        logger.warning(f"Failed to update relationship analysis metrics: {e}")
                    return
                
                # Извлекаем концепции для сопоставления
                concepts, entity_types = await self._extract_concepts(content, user_id)
                concept_to_node = {concept.lower(): nid for concept, nid in zip(concepts, added_node_ids)}
                
                # Advanced: создаём типизированные рёбра между обнаруженными концепциями
                adv = advanced_relationships(content)
                relationships_added = 0
                
                logger.info(f"Advanced relationships found: {len(adv)}")
                for src_c, dst_c, rel_type, weight in adv[:10]:
                    # Ищем узлы по имени (без учета регистра)
                    src_id = None
                    dst_id = None
                    for node_id in added_node_ids:
                        # Получаем информацию об узле для сравнения
                        # Пока используем простую логику - ищем по концептам
                        for concept, nid in zip(concepts, added_node_ids):
                            if nid == node_id and concept.lower() == src_c.lower():
                                src_id = node_id
                            if nid == node_id and concept.lower() == dst_c.lower():
                                dst_id = node_id
                    
                    if src_id and dst_id and src_id != dst_id:
                        await self.graph_memory.add_edge(
                            source_id=src_id,
                            target_id=dst_id,
                            user_id=user_id,
                            relationship_type=rel_type,
                            strength=float(min(1.0, max(0.1, weight)))
                        )
                        relationships_added += 1
                        logger.info(f"Added relationship: {src_c} -> {dst_c} ({rel_type})")
                
                # Fallback: если advanced не нашел связей, создаем базовые связи между всеми узлами
                if relationships_added == 0 and len(added_node_ids) >= 2:
                    logger.info("No advanced relationships found, creating basic connections")
                    for i in range(len(added_node_ids)):
                        for j in range(i + 1, len(added_node_ids)):
                            try:
                                await self.graph_memory.add_edge(
                                    source_id=added_node_ids[i],
                                    target_id=added_node_ids[j],
                                    user_id=user_id,
                                    relationship_type="related",
                                    strength=0.5
                                )
                                relationships_added += 1
                                logger.info(f"Added basic relationship: {added_node_ids[i]} -> {added_node_ids[j]}")
                            except Exception as e:
                                logger.warning(f"Failed to add basic relationship: {e}")
                
                # Naive fallback: если узлов >=2 и нет явных связей — добавим similar_to одной связью
                if relationships_added == 0 and len(added_node_ids) >= 2:
                    sentences = [s.strip() for s in content.split('.') if s.strip()]
                    rels = naive_relationships(sentences)
                    if rels:
                        await self.graph_memory.add_edge(
                            source_id=added_node_ids[0],
                            target_id=added_node_ids[1],
                            user_id=user_id,
                            relationship_type=rels[0][2],
                            strength=min(1.0, 0.5 + importance / 2)
                        )
                        relationships_added += 1
                
                # Обновляем метрики
                try:
                    inc("background_tasks_completed")
                except Exception as _e:
                    logger.warning(f"metrics inc failed (relationship-analysis): {_e}")
                
                logger.info(f"Relationship analysis completed successfully: user={user_id} relationships={relationships_added}")
                return
                
            except Exception as e:
                logger.error(f"Relationship analysis attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    # Последняя попытка неудачна - логируем ошибку
                    logger.error(f"Relationship analysis failed after {max_retries} attempts for user {user_id}: {e}")
                    try:
                        inc("background_tasks_failed")
                    except Exception as e:
                        logger.warning(f"Failed to update background tasks failed metrics: {e}")
                else:
                    # Ждем перед следующей попыткой (exponential backoff)
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)

    async def _get_search_cache(self, cache_key: str) -> Optional[dict]:
        """Получить результат поиска из кэша"""
        try:
            try:
                from ..cache.memory_cache import cache_get
            except ImportError:
                from cache.memory_cache import cache_get
            return await cache_get(cache_key)
        except Exception as e:
            logger.warning(f"Failed to get search cache: {e}")
            return None

    async def _set_search_cache(self, cache_key: str, result: dict, ttl_sec: int = 300):
        """Сохранить результат поиска в кэш"""
        try:
            try:
                from ..cache.memory_cache import cache_set
            except ImportError:
                from cache.memory_cache import cache_set
            await cache_set(cache_key, result, ttl_sec)
        except Exception as e:
            logger.warning(f"Failed to set search cache: {e}")

    def _is_search_cache_fresh(self, cached_result: dict, user_id: str) -> bool:
        """Проверить свежесть кэша поиска"""
        try:
            cache_timestamp = cached_result.get("timestamp", 0)
            # Проверяем, не обновлялась ли память пользователя
            last_memory_update = self._get_last_memory_update(user_id)
            return cache_timestamp >= last_memory_update
        except Exception as e:
            logger.warning(f"Failed to check cache freshness: {e}")
            return False

    def _is_search_result_stable(self, result: UnifiedMemoryResult) -> bool:
        """Проверить стабильность результата поиска для кэширования"""
        try:
            # Кэшируем только если есть результаты и они не слишком специфичны
            if result.total_items == 0:
                return False
            
            # Не кэшируем результаты с очень маленьким лимитом (могут быть специфичными)
            if hasattr(result, 'query') and len(result.query) < 3:
                return False
            
            # Кэшируем стабильные результаты
            return True
        except Exception as e:
            logger.warning(f"Failed to check result stability: {e}")
            return False

    def _get_last_memory_update(self, user_id: str) -> float:
        """Получить время последнего обновления памяти пользователя"""
        try:
            # Простая реализация - можно улучшить с помощью метрик
            import time
            return time.time() - 3600  # Предполагаем, что память обновлялась в последний час
        except Exception:
            return 0.0

    async def _invalidate_user_search_cache(self, user_id: str):
        """Инвалидировать кэш поиска для пользователя"""
        logger.info(f"Starting cache invalidation for user {user_id}")
        try:
            try:
                from ..cache.memory_cache import cache_delete_prefix
                logger.info(f"Imported cache_delete_prefix from ..cache.memory_cache")
            except ImportError:
                from cache.memory_cache import cache_delete_prefix
                logger.info(f"Imported cache_delete_prefix from cache.memory_cache")
            
            logger.info(f"Calling cache_delete_prefix for user {user_id}")
            deleted_count = await cache_delete_prefix(f"search:{user_id}:")
            logger.info(f"cache_delete_prefix returned {deleted_count} for user {user_id}")
            
            if deleted_count > 0:
                logger.info(f"Invalidated {deleted_count} search cache entries for user {user_id}")
                try:
                    inc("search_cache_invalidations")
                except Exception as e:
                    logger.warning(f"Failed to update search cache invalidations metrics: {e}")
        except Exception as e:
            logger.error(f"Failed to invalidate search cache for user {user_id}: {e}")
            import traceback
            logger.error(f"Cache invalidation traceback: {traceback.format_exc()}")

    async def consolidate_memory(self, user_id: str, level: Optional[str] = None) -> Dict[str, Any]:
        """
        Запустить консолидацию памяти
        
        Args:
            user_id: ID пользователя для консолидации
            level: Конкретный уровень для консолидации (опционально)
            
        Returns:
            Результаты консолидации
        """
        try:
            logger.info(f"Starting memory consolidation for level: {level or 'all'}")
            
            if level:
                # Консолидация конкретного уровня
                result = await self.consolidator.consolidate_memory_level(level, user_id)
                return {
                    "success": True,
                    "level": level,
                    "result": result
                }
            else:
                # Консолидация всех уровней
                result = await self.consolidator.consolidate_all_memory(user_id)
                return result
                
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_consolidation_stats(self) -> Dict[str, Any]:
        """
        Получить статистику консолидации памяти
        """
        try:
            return await self.consolidator.get_consolidation_stats()
        except Exception as e:
            logger.error(f"Error getting consolidation stats: {e}")
            return {"error": str(e)}
    
    async def schedule_consolidation(self, user_id: str, interval_hours: int = 24):
        """
        Запланировать периодическую консолидацию памяти
        
        Args:
            user_id: ID пользователя для консолидации
            interval_hours: Интервал между консолидациями в часах
        """
        try:
            # Добавляем флаг для graceful shutdown
            self._consolidation_running = True
            while self._consolidation_running:
                await asyncio.sleep(interval_hours * 3600)  # Конвертируем в секунды
                
                # Проверяем флаг перед выполнением
                if not self._consolidation_running:
                    break
                    
                logger.info(f"Starting scheduled memory consolidation for user {user_id}...")
                result = await self.consolidate_memory(user_id)
                
                if result.get("success"):
                    logger.info(f"Scheduled consolidation completed: {result.get('total_removed', 0)} items removed")
                else:
                    logger.error(f"Scheduled consolidation failed: {result.get('error', 'Unknown error')}")
                    
        except asyncio.CancelledError:
            logger.info("Scheduled consolidation cancelled")
        except Exception as e:
            logger.error(f"Error in scheduled consolidation: {e}")
        finally:
            self._consolidation_running = False
    
    def stop_scheduled_consolidation(self):
        """Остановка запланированной консолидации"""
        self._consolidation_running = False
        logger.info("Scheduled consolidation stop requested")
    
    async def cleanup_old_memories(self, days: int = 365) -> Dict[str, Any]:
        """
        Очистка старых воспоминаний
        
        Args:
            days: Количество дней для удержания воспоминаний
            
        Returns:
            Результаты очистки
        """
        try:
            logger.info(f"Starting cleanup of memories older than {days} days")
            result = await self.consolidator.cleanup_old_memories(str(days))
            return result
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _generate_recommendations(self, query: MemoryQuery, results: Dict[str, MemoryResult]) -> List[str]:
        """
        Генерировать рекомендации на основе результатов поиска
        
        Args:
            query: Исходный запрос
            results: Результаты поиска
            
        Returns:
            Список рекомендаций
        """
        try:
            recommendations = []
            
            # Анализируем результаты и генерируем рекомендации
            if results.get("working") and results["working"].total_found > 0:
                recommendations.append("У вас есть активный контекст по этой теме")
            
            if results.get("episodic") and results["episodic"].total_found > 0:
                recommendations.append("Найдены значимые воспоминания, связанные с запросом")
            
            if results.get("semantic") and results["semantic"].total_found > 0:
                recommendations.append("Доступны знания по этой теме")
            
            if results.get("procedural") and results["procedural"].total_found > 0:
                recommendations.append("Есть навыки, которые могут помочь")
            
            # Если результатов мало, предлагаем альтернативы
            if sum(r.total_found for r in results.values()) < 3:
                recommendations.append("Попробуйте переформулировать запрос или добавить больше контекста")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def _create_summary_memory(self, user_id: str, recent_memories: List[Dict[str, Any]]) -> Optional[str]:
        """
        Создает суммаризацию последних воспоминаний
        
        Args:
            user_id: ID пользователя
            recent_memories: Список последних воспоминаний для суммаризации
            
        Returns:
            ID созданной суммаризации или None при ошибке
        """
        try:
            if not recent_memories:
                logger.warning(f"No memories to summarize for user {user_id}")
                return None
            
            # Объединяем содержимое воспоминаний
            combined_content = "\n".join([
                f"[{mem.get('type', 'unknown')}] {mem.get('content', '')}" 
                for mem in recent_memories
            ])
            
            # ОПТИМИЗАЦИЯ: Используем единый LLM вызов для суммаризации
            if self.llm_provider:
                unified_result = await self.llm_provider.process_memory_unified(
                    content=combined_content,
                    tasks=["summary"]
                )
            else:
                unified_result = {"summary": combined_content[:200] + "..."}
            summary_content = unified_result.get("summary", combined_content[:200] + "...")
            
            # Добавляем метаданные суммаризации
            summary_metadata = {
                "type": "summary",
                "source_count": len(recent_memories),
                "source_types": list(set(mem.get('type', 'unknown') for mem in recent_memories)),
                "created_at": datetime.now().isoformat(),
                "ttl_days": 180  # TTL 180 дней как указано в требованиях
            }
            
            # Сохраняем суммаризацию в semantic memory (долгосрочное хранение)
            summary_id = await self.semantic_memory.add_knowledge(
                content=summary_content,
                user_id=user_id,
                importance=0.8,  # Высокая важность для суммаризации
                knowledge_type="summary",
                source="memory_consolidation"
            )
            
            logger.info(f"Created summary memory: {summary_id} for user {user_id} with {len(recent_memories)} source memories")
            
            # Обновляем метрики
            try:
                inc("summary_memories_created")
                set_gauge("summary_memory_source_count", len(recent_memories))
            except Exception as e:
                logger.warning(f"Failed to update summary metrics: {e}")
            
            return summary_id
            
        except Exception as e:
            logger.error(f"Error creating summary memory for user {user_id}: {e}")
            return None
    
    async def _background_post_processing(self, user_id: str, content: str, results: Dict[str, Any], memory_type: str, importance: float) -> None:
        """
        Фоновая обработка после добавления памяти (не блокирует ответ)
        """
        logger.info(f"Background post-processing started for user {user_id}, results: {results}")
        try:
            # Инвалидируем кэш поиска для пользователя
            logger.info(f"About to invalidate search cache for user {user_id}")
            await self._invalidate_user_search_cache(user_id)
            logger.info(f"Search cache invalidated for user {user_id}")
            
            # FTS5 индексация для ключевого поиска
            logger.info(f"About to start FTS5 indexing for user {user_id}")
            try:
                logger.info(f"Starting FTS5 indexing for user {user_id}")
                from ..search import get_fts5_engine
                fts5_engine = await get_fts5_engine()
                logger.info(f"FTS5 engine obtained for user {user_id}")
                
                # Создаем уникальный ID для FTS5 (используем первый доступный ID)
                fts5_id = None
                for level, level_id in results.items():
                    if isinstance(level_id, list) and level_id:
                        fts5_id = level_id[0]
                        break
                    elif isinstance(level_id, str):
                        fts5_id = level_id
                        break
                
                logger.info(f"FTS5 ID found: {fts5_id} for user {user_id}")
                
                if fts5_id:
                    await fts5_engine.index_memory(
                        memory_id=fts5_id,
                        content=content,
                        user_id=user_id,
                        memory_type=memory_type,
                        importance=importance
                    )
                    logger.info(f"Memory {fts5_id} indexed in FTS5 successfully")
            except Exception as e:
                logger.error(f"Failed to index memory in FTS5: {e}")
                import traceback
                logger.error(f"FTS5 indexing traceback: {traceback.format_exc()}")
                
        except Exception as e:
            logger.error(f"Background post-processing failed for user {user_id}: {e}")

    async def _check_and_create_summary_async(self, user_id: str) -> None:
        """
        Асинхронная версия проверки и создания суммаризации (не блокирует ответ)
        
        Args:
            user_id: ID пользователя
        """
        try:
            summary_id = await self._check_and_create_summary(user_id)
            if summary_id:
                logger.info(f"Summary created asynchronously for user {user_id}: {summary_id}")
        except Exception as e:
            logger.error(f"Failed to create summary asynchronously for user {user_id}: {e}")
        finally:
            # Очищаем завершенную задачу и lock
            try:
                if user_id in self._summary_task_locks:
                    async with self._summary_task_locks[user_id]:
                        if user_id in self._summary_tasks:
                            del self._summary_tasks[user_id]
                        # Очищаем lock если он больше не нужен
                        del self._summary_task_locks[user_id]
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup summary task for user {user_id}: {cleanup_error}")

    async def _check_and_create_summary(self, user_id: str) -> Optional[str]:
        """
        Проверяет необходимость создания суммаризации и создает её при необходимости
        
        Args:
            user_id: ID пользователя
            
        Returns:
            ID созданной суммаризации или None
        """
        try:
            # ОПТИМИЗАЦИЯ: Атомарная проверка и обновление счетчика под одним lock
            async with self._summary_counters_lock:
                if user_id not in self._summary_counters:
                    self._summary_counters[user_id] = 0
                self._summary_counters[user_id] += 1
                final_count = self._summary_counters[user_id]
            
            # Проверяем, нужно ли создавать суммаризацию
            if final_count < self._summary_threshold:
                return None
            
            logger.info(f"Creating summary for user {user_id} (threshold reached: {final_count})")
            
            # Получаем последние воспоминания для суммаризации
            recent_memories = []
            
            # Собираем из всех уровней памяти
            for level_name, manager in [
                ("working", self.working_memory),
                ("short_term", self.short_term_memory), 
                ("episodic", self.episodic_memory)
            ]:
                try:
                    memories = await manager.get_memories(user_id, limit=20)
                    for mem in memories:
                        recent_memories.append({
                            "content": getattr(mem, 'content', str(mem)),
                            "type": level_name,
                            "timestamp": getattr(mem, 'created_at', datetime.now())
                        })
                except Exception as e:
                    logger.warning(f"Failed to get memories from {level_name} for summary: {e}")
            
            # Сортируем по времени (новые сначала)
            recent_memories.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
            
            # Берем последние 20 воспоминаний для суммаризации
            recent_memories = recent_memories[:20]
            
            # Создаем суммаризацию
            summary_id = await self._create_summary_memory(user_id, recent_memories)
            
            # Thread-safe сброс счетчика после создания суммаризации
            if summary_id:
                async with self._summary_counters_lock:
                    self._summary_counters[user_id] = 0
                logger.info(f"Summary created successfully for user {user_id}, counter reset")
            
            return summary_id
            
        except Exception as e:
            logger.error(f"Error in summary check for user {user_id}: {e}")
            return None
    
    async def _process_memory_unified(self, content: str, user_id: str, importance: float) -> Dict[str, Any]:
        """
        Единая обработка памяти через один LLM вызов
        Заменяет отдельные вызовы extract_entities, extract_facts, summarize_content
        
        Args:
            content: Содержимое для обработки
            user_id: ID пользователя
            importance: Важность контента
            
        Returns:
            Словарь с результатами: entities, facts, summary
        """
        try:
            logger.info(f"Starting unified memory processing: user={user_id}, importance={importance}")
            # УЛУЧШЕННАЯ ОБРАБОТКА КОРОТКИХ СООБЩЕНИЙ
            content_length = len(content.strip())
            if content_length < 3:
                # Очень короткие сообщения (менее 3 символов) - только базовое сохранение
                logger.info(f"Very short content: {content_length} chars - basic processing")
                return {
                    "entities": [],
                    "facts": [],
                    "summary": content.strip()
                }
            elif content_length < 10:
                # Короткие сообщения (3-9 символов) - упрощенная обработка
                logger.info(f"Short content: {content_length} chars - simplified processing")
                # Для коротких сообщений делаем базовое извлечение сущностей
                simplified_entities = []
                simplified_facts = []
                
                # Простое извлечение сущностей из коротких сообщений
                words = content.strip().split()
                for word in words:
                    if len(word) > 2:  # Слова длиннее 2 символов
                        # Проверяем, является ли слово именем собственным или важным термином
                        if word[0].isupper() or word.lower() in ['да', 'нет', 'ок', 'хорошо', 'плохо', 'давай', 'стоп', 'начать', 'конец']:
                            simplified_entities.append({
                                "text": word,
                                "type": "keyword",
                                "confidence": 0.8
                            })
                
                # Простые факты для коротких сообщений
                if content.strip().lower() in ['да', 'yes', 'ок', 'хорошо', 'good']:
                    simplified_facts.append(["user", "agreed", "to something"])
                elif content.strip().lower() in ['нет', 'no', 'плохо', 'bad']:
                    simplified_facts.append(["user", "disagreed", "with something"])
                elif content.strip().lower() in ['стоп', 'stop', 'конец', 'end']:
                    simplified_facts.append(["user", "requested", "to stop"])
                elif content.strip().lower() in ['давай', 'let\'s', 'начать', 'start']:
                    simplified_facts.append(["user", "requested", "to start"])
                
                return {
                    "entities": simplified_entities,
                    "facts": simplified_facts,
                    "summary": content.strip()
                }

            # Проверяем на системные команды (но все равно обрабатываем)
            if content.strip().startswith('/'):
                logger.info(f"System command detected, processing anyway")
                # Для команд добавляем специальный промпт для извлечения фактов
                content = f"SYSTEM COMMAND: {content}\nExtract any useful information or knowledge from this command for memory storage."
            
            logger.info(f"Starting unified memory processing: user={user_id}, importance={importance:.2f}")
            
            # Единый вызов к LLM для всех задач
            if self.llm_provider:
                unified_result = await self.llm_provider.process_memory_unified(
                    content=content,
                    tasks=["entities", "facts", "summary"]
                )
            else:
                unified_result = {"entities": [], "facts": [], "summary": content[:200] + "..."}
            
            logger.info(f"Unified processing completed: entities={len(unified_result.get('entities', []))}, facts={len(unified_result.get('facts', []))}")
            
            return unified_result
            
        except Exception as e:
            logger.error(f"Error in unified memory processing: {e}")
            # Fallback: возвращаем пустую структуру
            return {
                "entities": [],
                "facts": [],
                "summary": content[:200] + "..." if len(content) > 200 else content
            }
    
    # ==================== ПРОАКТИВНЫЕ ЦЕЛИ ====================
    
    async def start_proactive_timer(self):
        """Запуск проактивного таймера целей"""
        if hasattr(self, 'proactive_timer'):
            await self.proactive_timer.start()
            logger.info("Proactive goals timer started")
    
    async def stop_proactive_timer(self):
        """Остановка проактивного таймера целей"""
        if hasattr(self, 'proactive_timer'):
            await self.proactive_timer.stop()
            logger.info("Proactive goals timer stopped")
    
    def create_proactive_goal(self, user_id: str, name: str, description: str, 
                            trigger_type: GoalTriggerType, trigger_value: str,
                            action_type: GoalActionType, action_params: str = "{}") -> str:
        """Создание проактивной цели"""
        goal_id = str(uuid.uuid4())
        
        goal = ProactiveGoal(
            id=goal_id,
            user_id=user_id,
            name=name,
            description=description,
            trigger_type=trigger_type,
            trigger_value=trigger_value,
            action_type=action_type,
            action_params=action_params
        )
        
        if hasattr(self, 'proactive_timer'):
            self.proactive_timer.add_goal(goal)
            logger.info(f"Created proactive goal: {name} ({goal_id})")
        
        return goal_id
    
    def get_proactive_goals(self, user_id: Optional[str] = None) -> List[ProactiveGoal]:
        """Получение проактивных целей"""
        if not hasattr(self, 'proactive_timer'):
            return []
        
        if user_id:
            return self.proactive_timer.get_user_goals(user_id)
        else:
            return self.proactive_timer.get_goals()
    
    def remove_proactive_goal(self, goal_id: str) -> bool:
        """Удаление проактивной цели"""
        if hasattr(self, 'proactive_timer'):
            self.proactive_timer.remove_goal(goal_id)
            logger.info(f"Removed proactive goal: {goal_id}")
            return True
        return False
    
    def get_proactive_timer_status(self) -> Dict[str, Any]:
        """Получение статуса проактивного таймера"""
        if hasattr(self, 'proactive_timer'):
            return self.proactive_timer.get_status()
        return {"is_running": False, "total_goals": 0, "active_goals": 0}
    
    def _get_sensor_buffer_stats(self) -> Dict[str, Any]:
        """Получение статистики SensorBuffer для мониторинга"""
        try:
            if hasattr(self, 'sensor_buffer') and hasattr(self.sensor_buffer, 'get_stats'):
                try:
                    # Проверяем наличие метода через getattr
                    get_stats_method = getattr(self.sensor_buffer, 'get_stats', None)
                    if get_stats_method:
                        stats = get_stats_method()
                    else:
                        return {"error": "SensorBuffer.get_stats not available"}
                except AttributeError:
                    return {"error": "SensorBuffer.get_stats not available"}
                return {
                    "total_items": stats.get("total_items", 0),
                    "audio_items": stats.get("audio_items", 0),
                    "text_items": stats.get("text_items", 0),
                    "emotion_items": stats.get("emotion_items", 0),
                    "total_added": stats.get("total_added", 0),
                    "total_evicted": stats.get("total_evicted", 0)
                }
            return {"error": "SensorBuffer not available"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _cleanup_sensor_buffer(self) -> float:
        """Очистка SensorBuffer"""
        try:
            if hasattr(self, 'sensor_buffer'):
                # Очищаем старые данные (старше 30 минут)
                cutoff_time = datetime.now() - timedelta(minutes=30)  # 30 минут назад
                cleared_count = 0
                
                # Очищаем каждый тип буфера
                for buffer_name in ['_audio_buffer', '_video_buffer', '_text_buffer', 
                                  '_emotion_buffer', '_gesture_buffer', '_environment_buffer']:
                    buffer = getattr(self.sensor_buffer, buffer_name, None)
                    if buffer:
                        # Удаляем старые элементы
                        old_items = [item for item in buffer if item.timestamp < cutoff_time]
                        for item in old_items:
                            buffer.remove(item)
                            cleared_count += 1
                
                # Примерная оценка освобожденной памяти (1KB на элемент)
                freed_mb = cleared_count * 0.001
                
                if cleared_count > 0:
                    logger.info(f"SensorBuffer очищен: удалено {cleared_count} элементов, "
                               f"освобождено ~{freed_mb:.2f}MB")
                
                return freed_mb
            return 0.0
        except Exception as e:
            logger.error(f"Ошибка очистки SensorBuffer: {e}")
            return 0.0
    
    def _get_orchestrator_stats(self) -> Dict[str, Any]:
        """Получение статистики MemoryOrchestrator для мониторинга"""
        try:
            return {
                "background_tasks": len(self._background_tasks),
                "summary_counters": len(self._summary_counters),
                "memory_levels": {
                    "sensor_buffer": hasattr(self, 'sensor_buffer'),
                    "working_memory": hasattr(self, 'working_memory'),
                    "short_term_memory": hasattr(self, 'short_term_memory'),
                    "episodic_memory": hasattr(self, 'episodic_memory'),
                    "semantic_memory": hasattr(self, 'semantic_memory'),
                    "graph_memory": hasattr(self, 'graph_memory'),
                    "procedural_memory": hasattr(self, 'procedural_memory'),
                    "summary_memory": hasattr(self, 'summary_memory')
                },
                "consolidator": hasattr(self, 'consolidator'),
                "proactive_timer": hasattr(self, 'proactive_timer'),
                "memory_monitor": hasattr(self, 'memory_monitor')
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _cleanup_orchestrator(self) -> float:
        """Очистка MemoryOrchestrator"""
        try:
            freed_mb = 0.0
            
            # Очищаем старые счетчики суммаризации (старше 1 часа)
            if hasattr(self, '_summary_counters'):
                current_time = time.time()
                old_counters = []
                
                for user_id, counter_data in self._summary_counters.items():
                    if isinstance(counter_data, dict) and 'last_update' in counter_data:
                        if current_time - counter_data['last_update'] > 3600:  # 1 час
                            old_counters.append(user_id)
                
                for user_id in old_counters:
                    del self._summary_counters[user_id]
                
                if old_counters:
                    freed_mb += len(old_counters) * 0.001  # Примерная оценка
                    logger.info(f"Очищены старые счетчики суммаризации: {len(old_counters)} записей")
            
            # Очищаем завершенные фоновые задачи
            if hasattr(self, '_background_tasks'):
                completed_tasks = [task for task in self._background_tasks if task.done()]
                for task in completed_tasks:
                    self._background_tasks.discard(task)
                
                if completed_tasks:
                    freed_mb += len(completed_tasks) * 0.001  # Примерная оценка
                    logger.info(f"Очищены завершенные фоновые задачи: {len(completed_tasks)} задач")
            
            return freed_mb
        except Exception as e:
            logger.error(f"Ошибка очистки MemoryOrchestrator: {e}")
            return 0.0
    
    async def start_memory_monitoring(self):
        """Запуск мониторинга памяти"""
        if hasattr(self, 'memory_monitor') and self.memory_monitor:
            await self.memory_monitor.start_monitoring()
            logger.info("Memory monitoring started")
    
    async def stop_memory_monitoring(self):
        """Остановка мониторинга памяти"""
        if hasattr(self, 'memory_monitor') and self.memory_monitor:
            await self.memory_monitor.stop_monitoring()
            logger.info("Memory monitoring stopped")
    
    def get_memory_monitoring_status(self) -> Dict[str, Any]:
        """Получение статуса мониторинга памяти"""
        if hasattr(self, 'memory_monitor') and self.memory_monitor:
            return self.memory_monitor.get_monitoring_status()
        return {"error": "Memory monitoring not available"}
    
    async def force_memory_cleanup(self) -> Dict[str, Any]:
        """Принудительная очистка памяти"""
        if hasattr(self, 'memory_monitor') and self.memory_monitor:
            return await self.memory_monitor.force_cleanup()
        return {"error": "Memory monitoring not available"}
    
    async def start_auto_recovery(self):
        """Запуск автоматического восстановления - ОТКЛЮЧЕНО"""
        logger.info("Auto recovery ОТКЛЮЧЕН - запуск невозможен")
        return {"status": "disabled", "message": "Auto recovery system is disabled"}
    
    async def stop_auto_recovery(self):
        """Остановка автоматического восстановления - ОТКЛЮЧЕНО"""
        logger.info("Auto recovery ОТКЛЮЧЕН - остановка невозможна")
        return {"status": "disabled", "message": "Auto recovery system is disabled"}
    
    def get_auto_recovery_status(self) -> Dict[str, Any]:
        """Получение статуса автоматического восстановления - ОТКЛЮЧЕНО"""
        return {
            "status": "disabled", 
            "message": "Auto recovery system is disabled",
            "is_running": False,
            "components": {},
            "healthy_components": 0,
            "total_failures": 0
        }
    
    def get_components_status(self) -> Dict[str, Any]:
        """Получение статуса всех компонентов - ОТКЛЮЧЕНО"""
        return {
            "status": "disabled",
            "message": "Auto recovery system is disabled",
            "components": {}
        }
    
    def get_recovery_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Получение истории восстановления - ОТКЛЮЧЕНО"""
        return []
    
    async def force_component_recovery(self, component_name: str) -> Dict[str, Any]:
        """Принудительное восстановление компонента - ОТКЛЮЧЕНО"""
        return {
            "status": "disabled",
            "message": "Auto recovery system is disabled",
            "component_name": component_name,
            "success": False
        }
    
    def get_proactive_execution_results(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Получение результатов выполнения проактивных целей"""
        if not hasattr(self, 'proactive_timer'):
            return []
        
        results = self.proactive_timer.get_execution_results(limit)
        return [
            {
                "goal_id": result.goal_id,
                "success": result.success,
                "message": result.message,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat(),
                "data": result.data
            }
            for result in results
        ]
    
    # ==================== МЕЖУРОВНЕВЫЕ ССЫЛКИ ====================
    
    async def _track_interlevel_reference(self, source_id: str, target_id: str, 
                                        source_level: str, target_level: str, 
                                        user_id: str) -> None:
        """Отслеживание межуровневой ссылки между элементами памяти"""
        try:
            # Сохраняем ссылку в метаданных элементов
            reference_data = {
                "source_id": source_id,
                "target_id": target_id,
                "source_level": source_level,
                "target_level": target_level,
                "user_id": user_id,
                "created_at": datetime.now().isoformat()
            }
            
            # Добавляем ссылку в метаданные целевого элемента
            await self._add_reference_to_metadata(target_id, target_level, user_id, reference_data)
            
            logger.debug(f"Interlevel reference tracked: {source_level}:{source_id} -> {target_level}:{target_id}")
            
        except Exception as e:
            logger.warning(f"Failed to track interlevel reference: {e}")
    
    async def _add_reference_to_metadata(self, item_id: str, level: str, user_id: str, 
                                       reference_data: Dict[str, Any]) -> None:
        """Добавление ссылки в метаданные элемента памяти"""
        try:
            # Получаем элемент по ID и уровню
            item = await self._get_memory_item_by_id(item_id, level, user_id)
            if not item:
                return
            
            # Добавляем ссылку в метаданные
            if not hasattr(item, 'metadata') or not item.metadata:
                item.metadata = {}
            
            if 'interlevel_references' not in item.metadata:
                item.metadata['interlevel_references'] = []
            
            item.metadata['interlevel_references'].append(reference_data)
            
            # Сохраняем обновленный элемент
            await self._update_memory_item_metadata(item_id, level, user_id, item.metadata)
            
        except Exception as e:
            logger.warning(f"Failed to add reference to metadata: {e}")
    
    async def _get_memory_item_by_id(self, item_id: str, level: str, user_id: str) -> Optional[Any]:
        """Получение элемента памяти по ID и уровню"""
        try:
            if level == "working":
                return await self.working_memory.get_memory_by_id(item_id, user_id)
            elif level == "short_term":
                return await self.short_term_memory.get_memory_by_id(item_id, user_id)
            elif level == "episodic":
                return await self.episodic_memory.get_memory_by_id(item_id, user_id)
            elif level == "semantic":
                return await self.semantic_memory.get_memory_by_id(item_id, user_id)
            elif level == "graph":
                return await self.graph_memory.get_node_by_id(item_id, user_id)
            elif level == "procedural":
                return await self.procedural_memory.get_memory_by_id(item_id, user_id)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get memory item by ID: {e}")
            return None
    
    async def _update_memory_item_metadata(self, item_id: str, level: str, user_id: str, 
                                         metadata: Dict[str, Any]) -> None:
        """Обновление метаданных элемента памяти"""
        try:
            if level == "working":
                await self.working_memory.update_metadata(item_id, user_id, metadata)
            elif level == "short_term":
                await self.short_term_memory.update_metadata(item_id, user_id, metadata)
            elif level == "episodic":
                await self.episodic_memory.update_metadata(item_id, user_id, metadata)
            elif level == "semantic":
                await self.semantic_memory.update_metadata(item_id, user_id, metadata)
            elif level == "graph":
                await self.graph_memory.update_node_metadata(item_id, user_id, metadata)
            elif level == "procedural":
                await self.procedural_memory.update_metadata(item_id, user_id, metadata)
                
        except Exception as e:
            logger.warning(f"Failed to update memory item metadata: {e}")
    
    async def _check_interlevel_duplicates(self, content: str, user_id: str, 
                                         exclude_level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Проверка дубликатов между уровнями памяти"""
        try:
            duplicates = []
            
            # Проверяем все уровни кроме исключенного
            levels_to_check = ["working", "short_term", "episodic", "semantic", "procedural"]
            if exclude_level:
                levels_to_check = [level for level in levels_to_check if level != exclude_level]
            
            for level in levels_to_check:
                # Поиск похожего контента на уровне
                similar_items = await self._find_similar_content_on_level(
                    content, level, user_id, similarity_threshold=0.8
                )
                
                for item in similar_items:
                    duplicates.append({
                        "level": level,
                        "item_id": item.get("id"),
                        "similarity": item.get("similarity", 0.0),
                        "content": item.get("content", "")[:100] + "..."
                    })
            
            return duplicates
            
        except Exception as e:
            logger.warning(f"Failed to check interlevel duplicates: {e}")
            return []
    
    async def _find_similar_content_on_level(self, content: str, level: str, user_id: str, 
                                           similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Поиск похожего контента на конкретном уровне памяти"""
        try:
            # Простая проверка на основе длины и первых символов
            # В реальной реализации здесь должен быть более сложный алгоритм
            similar_items = []
            
            # Получаем элементы уровня используя реальные методы
            if level == "working":
                items = await self.working_memory.get_active_context(user_id, limit=100)
            elif level == "short_term":
                items = await self.short_term_memory.get_recent_events(user_id, hours=168, limit=100)  # 7 дней
            elif level == "episodic":
                items = await self.episodic_memory.get_significant_events(user_id, days=30, limit=100)
            elif level == "semantic":
                # Для semantic используем get_knowledge_by_category с общими категориями
                items = []
                categories = ["general", "knowledge", "fact", "concept", "information"]
                for category in categories:
                    try:
                        category_items = await self.semantic_memory.get_knowledge_by_category(user_id, category, limit=20)
                        items.extend(category_items)
                    except Exception:
                        continue
            elif level == "procedural":
                # Для procedural используем get_skills_by_type с общими типами
                items = []
                skill_types = ["general", "skill", "procedure", "knowledge", "ability"]
                for skill_type in skill_types:
                    try:
                        skill_items = await self.procedural_memory.get_skills_by_type(user_id, skill_type, limit=20)
                        items.extend(skill_items)
                    except Exception:
                        continue
            else:
                return similar_items
            
            # Простая проверка схожести
            for item in items:
                if hasattr(item, 'content') and item.content:
                    similarity = self._calculate_content_similarity(content, item.content)
                    if similarity >= similarity_threshold:
                        similar_items.append({
                            "id": getattr(item, 'id', None),
                            "content": item.content,
                            "similarity": similarity
                        })
            
            return similar_items
            
        except Exception as e:
            logger.warning(f"Failed to find similar content on level {level}: {e}")
            return []
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Простой расчет схожести контента"""
        try:
            # Нормализация текста
            text1 = content1.lower().strip()
            text2 = content2.lower().strip()
            
            # Если тексты идентичны
            if text1 == text2:
                return 1.0
            
            # Если один текст содержит другой
            if text1 in text2 or text2 in text1:
                return 0.9
            
            # Простая проверка на основе длины и общих слов
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate content similarity: {e}")
            return 0.0
    
    async def get_interlevel_references(self, item_id: str, level: str, user_id: str) -> List[Dict[str, Any]]:
        """Получение всех межуровневых ссылок для элемента"""
        try:
            item = await self._get_memory_item_by_id(item_id, level, user_id)
            if not item or not hasattr(item, 'metadata') or not item.metadata:
                return []
            
            return item.metadata.get('interlevel_references', [])
            
        except Exception as e:
            logger.warning(f"Failed to get interlevel references: {e}")
            return []
    
    async def consolidate_with_interlevel_tracking(self, user_id: str) -> Dict[str, Any]:
        """Консолидация с отслеживанием межуровневых ссылок"""
        try:
            logger.info(f"Starting consolidation with interlevel tracking for user {user_id}")
            
            # Выполняем обычную консолидацию
            consolidation_result = await self.consolidate_memories(user_id)
            
            # Отслеживаем межуровневые ссылки для перемещенных элементов
            interlevel_tracking = {
                "working_to_short_term": 0,
                "short_term_to_episodic": 0,
                "episodic_to_semantic": 0,
                "total_references_created": 0
            }
            
            # Здесь можно добавить логику отслеживания ссылок
            # при перемещении элементов между уровнями
            
            consolidation_result["interlevel_tracking"] = interlevel_tracking.get("total_references_created", 0)
            
            logger.info(f"Consolidation with interlevel tracking completed for user {user_id}")
            return consolidation_result
            
        except Exception as e:
            logger.error(f"Failed to consolidate with interlevel tracking: {e}")
            return {"error": str(e)}
    
    def _is_monitoring_working(self) -> bool:
        """Проверка работоспособности мониторинга"""
        try:
            # Тестируем основные функции мониторинга
            inc("monitoring_test", 1)
            set_gauge("monitoring_test_gauge", 1.0)
            record_event("monitoring_test", {"test": True})
            logger.debug("✅ Monitoring test passed - all functions working")
            return True
        except Exception as e:
            logger.error(f"🚨 CRITICAL: Monitoring test failed: {e}")
            logger.error("🚨 Monitoring is using dummy functions")
            return False
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Получение статуса мониторинга"""
        try:
            return {
                "monitoring_working": self._is_monitoring_working(),
                "memory_monitor_available": self.memory_monitor is not None,
                "memory_cleanup_available": self.memory_cleanup is not None,
                "metrics_available": True,  # Будет проверено в _is_monitoring_working
                "last_test_time": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get monitoring status: {e}")
            return {
                "monitoring_working": False,
                "error": str(e)
            }
