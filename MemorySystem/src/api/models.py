"""
Pydantic модели для AIRI Memory System API
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class MemoryResult(BaseModel):
    """Результат поиска воспоминания"""
    id: str = Field(..., description="ID воспоминания")
    memory: str = Field(..., description="Содержимое воспоминания")
    metadata: Dict[str, Any] = Field(..., description="Метаданные воспоминания")
    similarity: float = Field(..., description="Схожесть с запросом", ge=0.0, le=1.0)
    created_at: Optional[str] = Field(default=None, description="Время создания")


class GetMemoryResponse(BaseModel):
    """Ответ на получение воспоминания"""
    id: str = Field(..., description="ID воспоминания")
    content: str = Field(..., description="Содержимое воспоминания")
    metadata: Dict[str, Any] = Field(..., description="Метаданные воспоминания")
    created_at: Optional[str] = Field(default=None, description="Время создания")


class UpdateMemoryRequest(BaseModel):
    """Запрос на обновление воспоминания"""
    content: Optional[str] = Field(default=None, description="Новое содержимое воспоминания")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Новые метаданные")


class UpdateMemoryResponse(BaseModel):
    """Ответ на обновление воспоминания"""
    success: bool = Field(..., description="Успешность операции")
    message: str = Field(..., description="Сообщение о результате")


class DeleteMemoryResponse(BaseModel):
    """Ответ на удаление воспоминания"""
    success: bool = Field(..., description="Успешность операции")
    message: str = Field(..., description="Сообщение о результате")


class UserMemory(BaseModel):
    """Воспоминание пользователя"""
    id: str = Field(..., description="ID воспоминания")
    content: str = Field(..., description="Содержимое воспоминания")
    metadata: Dict[str, Any] = Field(..., description="Метаданные воспоминания")
    created_at: Optional[str] = Field(default=None, description="Время создания")


class GetUserMemoriesResponse(BaseModel):
    """Ответ на получение воспоминаний пользователя"""
    memories: List[UserMemory] = Field(..., description="Воспоминания пользователя")
    total: int = Field(..., description="Общее количество воспоминаний")


class MemoryStats(BaseModel):
    """Статистика памяти"""
    total_memories: int = Field(..., description="Общее количество воспоминаний")
    memory_types: Dict[str, int] = Field(..., description="Количество воспоминаний по типам")
    recent_memories: int = Field(..., description="Количество недавних воспоминаний")


class SystemStats(BaseModel):
    """Статистика системы"""
    vector_store: Dict[str, Any] = Field(..., description="Статистика векторного хранилища")
    embeddings: Dict[str, Any] = Field(..., description="Статистика эмбеддингов")
    llm_provider: str = Field(..., description="Провайдер LLM")
    embedder_provider: str = Field(..., description="Провайдер эмбеддингов")
    vector_store_provider: str = Field(..., description="Провайдер векторного хранилища")
    user_stats: Optional[MemoryStats] = Field(default=None, description="Статистика пользователя")


class HealthCheckResponse(BaseModel):
    """Ответ проверки здоровья системы"""
    status: str = Field(..., description="Статус системы (healthy/unhealthy)")
    overall: bool = Field(..., description="Общий статус системы")
    llm: bool = Field(..., description="Статус LLM провайдера")
    embeddings: bool = Field(..., description="Статус провайдера эмбеддингов")
    vector_store: bool = Field(..., description="Статус векторного хранилища")
    service: str = Field(..., description="Название сервиса")
    version: str = Field(..., description="Версия сервиса")
    timestamp: str = Field(..., description="Время проверки")


class CleanupRequest(BaseModel):
    """Запрос на очистку старых воспоминаний"""
    days: Optional[int] = Field(default=None, description="Количество дней для сохранения воспоминаний")


class CleanupResponse(BaseModel):
    """Ответ на очистку старых воспоминаний"""
    deleted_count: int = Field(..., description="Количество удаленных воспоминаний")
    message: str = Field(..., description="Сообщение о результате")


class ErrorResponse(BaseModel):
    """Ответ об ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(default=None, description="Детали ошибки")
    timestamp: str = Field(..., description="Время ошибки")


# ===== МОДЕЛИ ДЛЯ ЦЕЛЕЙ (Procedural Goals / SQLite) =====

class GoalItem(BaseModel):
    id: str
    name: str
    description: str
    status: str
    progress: float


class GoalsListResponse(BaseModel):
    goals: List[GoalItem] = []


class CreateGoalRequest(BaseModel):
    user_id: str
    name: str
    description: Optional[str] = ""
    next_run: Optional[datetime] = None


class CreateGoalResponse(BaseModel):
    id: str
    status: str = "created"


# ===== МОДЕЛИ ДЛЯ ПРОАКТИВНЫХ ЦЕЛЕЙ =====

class GoalTriggerType(str, Enum):
    """Типы триггеров для проактивных целей"""
    TIME_INTERVAL = "time_interval"  # Каждые N минут
    SCHEDULED_TIME = "scheduled_time"  # В определенное время
    EVENT = "event"  # По событию
    CONDITION = "condition"  # По условию


class GoalActionType(str, Enum):
    """Типы действий для проактивных целей"""
    ANALYZE_MEMORY = "analyze_memory"
    GENERATE_IDEAS = "generate_ideas"
    HEALTH_CHECK = "health_check"
    CLEANUP_OLD_DATA = "cleanup_old_data"
    CREATE_REMINDER = "create_reminder"
    ANALYZE_EMOTIONS = "analyze_emotions"
    CONSOLIDATE_MEMORY = "consolidate_memory"
    DEDUPLICATE_MEMORY = "deduplicate_memory"


class CreateProactiveGoalRequest(BaseModel):
    """Запрос создания проактивной цели"""
    user_id: str = Field(..., description="ID пользователя")
    name: str = Field(..., description="Название цели")
    description: str = Field(..., description="Описание цели")
    trigger_type: GoalTriggerType = Field(..., description="Тип триггера")
    trigger_value: str = Field(..., description="Значение триггера")
    action_type: GoalActionType = Field(..., description="Тип действия")
    action_params: str = Field(default="{}", description="Параметры действия в JSON")


class CreateProactiveGoalResponse(BaseModel):
    """Ответ создания проактивной цели"""
    success: bool = Field(default=True, description="Статус успешности")
    goal_id: str = Field(..., description="ID созданной цели")
    timestamp: str = Field(..., description="Время создания")


class ProactiveGoalItem(BaseModel):
    """Элемент проактивной цели"""
    id: str = Field(..., description="ID цели")
    user_id: str = Field(..., description="ID пользователя")
    name: str = Field(..., description="Название цели")
    description: str = Field(..., description="Описание цели")
    trigger_type: str = Field(..., description="Тип триггера")
    trigger_value: str = Field(..., description="Значение триггера")
    action_type: str = Field(..., description="Тип действия")
    action_params: str = Field(..., description="Параметры действия")
    status: str = Field(..., description="Статус цели")
    created_at: str = Field(..., description="Время создания")
    last_triggered: Optional[str] = Field(None, description="Время последнего срабатывания")


class ProactiveGoalsListResponse(BaseModel):
    """Ответ списка проактивных целей"""
    goals: List[ProactiveGoalItem] = Field(default=[], description="Список целей")
    total_count: int = Field(default=0, description="Общее количество целей")


class DeleteProactiveGoalResponse(BaseModel):
    """Ответ удаления проактивной цели"""
    success: bool = Field(default=True, description="Статус успешности")
    message: str = Field(..., description="Сообщение о результате")
    timestamp: str = Field(..., description="Время операции")


class ProactiveGoalsStatusResponse(BaseModel):
    """Ответ статуса проактивных целей"""
    status: str = Field(..., description="Статус системы")
    active_goals: int = Field(default=0, description="Количество активных целей")
    timestamp: str = Field(..., description="Время получения статуса")


class ProactiveGoalsResultsResponse(BaseModel):
    """Ответ результатов проактивных целей"""
    results: List[dict] = Field(default=[], description="Результаты выполнения")
    total_executed: int = Field(default=0, description="Общее количество выполненных целей")
    timestamp: str = Field(..., description="Время получения результатов")


# ===== МОДЕЛИ ДЛЯ МУЛЬТИМОДАЛЬНОГО АНАЛИЗА =====

class VisionAnalyzeRequest(BaseModel):
    prompt: str = Field(default="Опиши изображение")
    image_b64: str = Field(..., description="Изображение в base64")


class VisionAnalyzeResponse(BaseModel):
    result: str


# ===== МОДЕЛИ ДЛЯ МНОГОУРОВНЕВОЙ ПАМЯТИ =====

class MemoryLevel(str, Enum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    GRAPH = "graph"
    PROCEDURAL = "procedural"


class MultiLevelMemoryRequest(BaseModel):
    content: str
    user_id: str
    level: MemoryLevel
    importance: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Важность воспоминания (0.0-1.0)")
    metadata: Optional[Dict[str, Any]] = None


# ===== МОДЕЛИ ДЛЯ ЭМОЦИОНАЛЬНЫХ ДАННЫХ =====

class EmotionComplexity(str, Enum):
    """Уровни сложности эмоций"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class EmotionData(BaseModel):
    """Данные об эмоциях"""
    primary_emotion: str = Field(..., description="Основная эмоция")
    primary_confidence: float = Field(..., description="Уверенность в основной эмоции", ge=0.0, le=1.0)
    secondary_emotion: Optional[str] = Field(default=None, description="Вторичная эмоция")
    secondary_confidence: Optional[float] = Field(default=None, description="Уверенность во вторичной эмоции", ge=0.0, le=1.0)


class MultiLevelAddRequest(BaseModel):
    """Тело запроса для добавления многоуровневой памяти (JSON)."""
    content: str
    user_id: str
    level: Optional[str] = "working"  # Стандартизировано на level
    importance: float = 0.5
    emotion_data: Optional["EmotionData"] = None
    context: Optional[str] = None
    participants: Optional[List[str]] = None
    location: Optional[str] = None


class MultiLevelSearchRequest(BaseModel):
    """Тело запроса для поиска по многоуровневой памяти (JSON)."""
    query: str
    user_id: str
    memory_levels: Optional[List[str]] = None
    limit: int = 10
    offset: int = 0
    include_emotions: bool = True
    include_relationships: bool = True
    consistency: str = Field(default="high", description="Согласованность между источниками")
    dominant_source: str = Field(default="voice", description="Доминирующий источник эмоций")
    validation_applied: bool = Field(default=False, description="Применена ли валидация")


class FormattedEmotion(BaseModel):
    """Отформатированная эмоция для ИИ"""
    emotion_prompt: str = Field(..., description="Промпт с эмоциональным контекстом")
    complexity: EmotionComplexity = Field(..., description="Уровень сложности эмоций")
    context_hints: List[str] = Field(default=[], description="Контекстные подсказки")
    metadata: Dict[str, Any] = Field(default={}, description="Метаданные эмоций")


class EmotionEnhancedMemoryRequest(BaseModel):
    """Запрос на добавление воспоминания с эмоциональными данными"""
    content: str = Field(..., description="Содержимое воспоминания", min_length=1)
    user_id: str = Field(default="default_user", description="ID пользователя")
    emotion_data: Optional[EmotionData] = Field(default=None, description="Данные об эмоциях")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Дополнительные метаданные")


class EmotionEnhancedMemoryResponse(BaseModel):
    """Ответ на добавление воспоминания с эмоциональными данными"""
    id: str = Field(..., description="ID созданного воспоминания")
    content: str = Field(..., description="Содержимое воспоминания")
    user_id: str = Field(..., description="ID пользователя")
    emotion_data: Optional[EmotionData] = Field(default=None, description="Данные об эмоциях")
    formatted_emotion: Optional[FormattedEmotion] = Field(default=None, description="Отформатированная эмоция")
    metadata: Dict[str, Any] = Field(..., description="Метаданные воспоминания")
    created_at: str = Field(..., description="Время создания")
    emotion_enhancement_applied: bool = Field(default=False, description="Применено ли эмоциональное усиление")


class EmotionAnalysisRequest(BaseModel):
    """Запрос на анализ эмоций в тексте"""
    text: str = Field(..., description="Текст для анализа эмоций", min_length=1)
    user_id: str = Field(default="default_user", description="ID пользователя")
    include_validation: bool = Field(default=True, description="Включить ли валидацию")


class VoiceEmotionAnalysisRequest(BaseModel):
    """Запрос на анализ эмоций в голосе"""
    audio_b64: str = Field(..., description="Аудио данные в формате base64")
    user_id: str = Field(default="default_user", description="ID пользователя")


class EmotionAnalysisResponse(BaseModel):
    """Ответ на анализ эмоций"""
    emotion_data: EmotionData = Field(..., description="Данные об эмоциях")
    formatted_emotion: FormattedEmotion = Field(..., description="Отформатированная эмоция")
    analysis_time: float = Field(..., description="Время анализа в секундах")
    validation_applied: bool = Field(default=False, description="Применена ли валидация")


# ===== МУЛЬТИМОДАЛЬНЫЙ ВХОД (АУДИО) =====

class EmotionLabel(BaseModel):
    label: str
    confidence: float
    category: Optional[str] = None


class IngestAudioRequest(BaseModel):
    user_id: str
    audio_b64: str
    metadata: Optional[Dict[str, Any]] = None


class IngestAudioResponse(BaseModel):
    text: str
    emotions: List[EmotionLabel]
    sentiment: Optional[str] = None
    dominant_source: Optional[str] = None
    consistency: Optional[str] = None
    memory: Dict[str, Any]
    # Extended AI-ready payload
    merged_top3: Optional[List[EmotionLabel]] = None
    voice_top3: Optional[List[EmotionLabel]] = None
    text_top3: Optional[List[EmotionLabel]] = None
    conflict: Optional[Dict[str, Any]] = None
    guidance: Optional[str] = None


# ==================== A/B TESTING MODELS ====================

class ClickRequest(BaseModel):
    """Запрос на запись клика"""
    user_id: str
    experiment_id: str
    variant_name: str
    query: str
    result_id: str


class SatisfactionRequest(BaseModel):
    """Запрос на запись удовлетворенности"""
    user_id: str
    experiment_id: str
    variant_name: str
    query: str
    satisfaction_score: float


class SearchMetricsRequest(BaseModel):
    """Запрос на запись метрик поиска"""
    user_id: str
    experiment_id: str
    variant_name: str
    query: str
    response_time_ms: float
    results_count: int
