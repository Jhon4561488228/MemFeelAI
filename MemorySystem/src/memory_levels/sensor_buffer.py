"""
Sensor Buffer для AIRI Memory System
Уровень 1: Кольцевой буфер для последних 20 секунд audio/video и 5 сообщений
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Deque
from dataclasses import dataclass, field
from collections import deque
import logging
import threading
from enum import Enum

try:
    from ..monitoring.metrics import inc, set_gauge, record_event
except ImportError:
    try:
        from monitoring.metrics import inc, set_gauge, record_event
    except ImportError:
        # Fallback для случаев когда модуль недоступен - используем реальные функции
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'monitoring'))
        try:
            from metrics import inc, set_gauge, record_event
        except ImportError:
            # Последний fallback - создаем простые функции мониторинга
            _counters = {}
            _gauges = {}
            _events = []
            
            def inc(name: str, value: int = 1) -> None:
                _counters[name] = _counters.get(name, 0) + value
            
            def set_gauge(name: str, value: float) -> None:
                _gauges[name] = float(value)
            
            def record_event(kind: str, data: dict) -> None:
                _events.append({"kind": kind, "data": data, "ts": time.time()})
                if len(_events) > 200:
                    _events.pop(0)

logger = logging.getLogger(__name__)

class SensorType(Enum):
    """Типы сенсорных данных"""
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    EMOTION = "emotion"
    GESTURE = "gesture"
    ENVIRONMENT = "environment"

@dataclass
class SensorData:
    """Данные сенсора"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sensor_type: SensorType = SensorType.TEXT
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Для аудио/видео данных
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    
    # Для текстовых данных
    text_content: Optional[str] = None
    language: Optional[str] = None
    
    # Эмоциональные данные
    emotion: Optional[str] = None
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации"""
        return {
            "id": self.id,
            "sensor_type": self.sensor_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "metadata": self.metadata,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "text_content": self.text_content,
            "language": self.language,
            "emotion": self.emotion,
            "confidence": self.confidence
        }

class SensorBuffer:
    """
    Кольцевой буфер для сенсорных данных
    - Последние 20 секунд audio/video
    - Последние 5 текстовых сообщений
    - Эмоциональные данные в реальном времени
    """
    
    def __init__(self, 
                 audio_buffer_seconds: int = 20,
                 text_buffer_size: int = 5,
                 emotion_buffer_size: int = 10):
        """
        Инициализация Sensor Buffer
        
        Args:
            audio_buffer_seconds: Размер аудио буфера в секундах
            text_buffer_size: Количество текстовых сообщений
            emotion_buffer_size: Количество эмоциональных записей
        """
        self.audio_buffer_seconds = audio_buffer_seconds
        self.text_buffer_size = text_buffer_size
        self.emotion_buffer_size = emotion_buffer_size
        
        # Кольцевые буферы для разных типов данных
        self._audio_buffer: Deque[SensorData] = deque(maxlen=1000)  # ~20 сек при 50fps
        self._video_buffer: Deque[SensorData] = deque(maxlen=1000)  # ~20 сек при 50fps
        self._text_buffer: Deque[SensorData] = deque(maxlen=text_buffer_size)
        self._emotion_buffer: Deque[SensorData] = deque(maxlen=emotion_buffer_size)
        self._gesture_buffer: Deque[SensorData] = deque(maxlen=100)  # Жесты
        self._environment_buffer: Deque[SensorData] = deque(maxlen=50)  # Окружение
        
        # Потокобезопасность
        self._lock = threading.RLock()
        
        # Метрики
        self._total_added = 0
        self._total_evicted = 0
        self._last_cleanup = time.time()
        
        logger.info(f"SensorBuffer инициализирован: audio={audio_buffer_seconds}s, text={text_buffer_size}, emotion={emotion_buffer_size}")
    
    def add_sensor_data(self, 
                       sensor_type: SensorType,
                       data: Any,
                       user_id: str = "default",
                       metadata: Optional[Dict[str, Any]] = None,
                       **kwargs) -> str:
        """
        Добавление данных в соответствующий буфер
        
        Args:
            sensor_type: Тип сенсорных данных
            data: Данные сенсора
            user_id: ID пользователя
            metadata: Дополнительные метаданные
            **kwargs: Дополнительные параметры (duration_seconds, sample_rate, etc.)
            
        Returns:
            ID добавленного элемента
        """
        with self._lock:
            # Создаем объект SensorData
            sensor_data = SensorData(
                sensor_type=sensor_type,
                data=data,
                user_id=user_id,
                metadata=metadata or {},
                **kwargs
            )
            
            # Добавляем в соответствующий буфер
            if sensor_type == SensorType.AUDIO:
                self._audio_buffer.append(sensor_data)
                self._cleanup_old_audio()
            elif sensor_type == SensorType.VIDEO:
                self._video_buffer.append(sensor_data)
                self._cleanup_old_video()
            elif sensor_type == SensorType.TEXT:
                self._text_buffer.append(sensor_data)
            elif sensor_type == SensorType.EMOTION:
                self._emotion_buffer.append(sensor_data)
            elif sensor_type == SensorType.GESTURE:
                self._gesture_buffer.append(sensor_data)
            elif sensor_type == SensorType.ENVIRONMENT:
                self._environment_buffer.append(sensor_data)
            
            self._total_added += 1
            
            # Обновляем метрики
            try:
                inc("sensor_buffer_added_total")
                set_gauge("sensor_buffer_size", self.get_total_size())
            except Exception as e:
                logger.debug(f"Failed to update metrics: {e}")
            
            logger.debug(f"Добавлены данные в {sensor_type.value} буфер: {sensor_data.id}")
            return sensor_data.id
    
    def get_recent_data(self, 
                       sensor_type: Optional[SensorType] = None,
                       user_id: Optional[str] = None,
                       seconds: Optional[int] = None,
                       limit: Optional[int] = None) -> List[SensorData]:
        """
        Получение недавних данных из буфера
        
        Args:
            sensor_type: Тип сенсорных данных (None = все типы)
            user_id: ID пользователя (None = все пользователи)
            seconds: Количество секунд назад (None = все данные)
            limit: Максимальное количество записей
            
        Returns:
            Список недавних данных
        """
        with self._lock:
            results = []
            cutoff_time = None
            
            if seconds:
                cutoff_time = datetime.now() - timedelta(seconds=seconds)
            
            # Выбираем буферы для поиска
            buffers_to_search = []
            if sensor_type is None:
                buffers_to_search = [
                    self._audio_buffer,
                    self._video_buffer,
                    self._text_buffer,
                    self._emotion_buffer,
                    self._gesture_buffer,
                    self._environment_buffer
                ]
            else:
                buffer_map = {
                    SensorType.AUDIO: self._audio_buffer,
                    SensorType.VIDEO: self._video_buffer,
                    SensorType.TEXT: self._text_buffer,
                    SensorType.EMOTION: self._emotion_buffer,
                    SensorType.GESTURE: self._gesture_buffer,
                    SensorType.ENVIRONMENT: self._environment_buffer
                }
                if sensor_type in buffer_map:
                    buffers_to_search = [buffer_map[sensor_type]]
            
            # Собираем данные
            for buffer in buffers_to_search:
                for data in buffer:
                    # Фильтруем по пользователю
                    if user_id and data.user_id != user_id:
                        continue
                    
                    # Фильтруем по времени
                    if cutoff_time and data.timestamp < cutoff_time:
                        continue
                    
                    results.append(data)
            
            # Сортируем по времени (новые сначала)
            results.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Ограничиваем количество
            if limit:
                results = results[:limit]
            
            return results
    
    def get_audio_context(self, seconds: int = 20, user_id: str = "default") -> List[SensorData]:
        """Получение аудио контекста за последние N секунд"""
        return self.get_recent_data(
            sensor_type=SensorType.AUDIO,
            user_id=user_id,
            seconds=seconds
        )
    
    def get_text_context(self, limit: int = 5, user_id: str = "default") -> List[SensorData]:
        """Получение текстового контекста (последние N сообщений)"""
        return self.get_recent_data(
            sensor_type=SensorType.TEXT,
            user_id=user_id,
            limit=limit
        )
    
    def get_emotion_context(self, limit: int = 10, user_id: str = "default") -> List[SensorData]:
        """Получение эмоционального контекста"""
        return self.get_recent_data(
            sensor_type=SensorType.EMOTION,
            user_id=user_id,
            limit=limit
        )
    
    def get_current_emotion(self, user_id: str = "default") -> Optional[str]:
        """Получение текущей эмоции пользователя"""
        emotions = self.get_emotion_context(limit=1, user_id=user_id)
        if emotions:
            return emotions[0].emotion
        return None
    
    def get_combined_context(self, 
                           audio_seconds: int = 20,
                           text_limit: int = 5,
                           emotion_limit: int = 10,
                           user_id: str = "default") -> Dict[str, List[SensorData]]:
        """
        Получение комбинированного контекста для передачи в Working Memory
        
        Returns:
            Словарь с данными всех типов сенсоров
        """
        return {
            "audio": self.get_audio_context(audio_seconds, user_id),
            "text": self.get_text_context(text_limit, user_id),
            "emotion": self.get_emotion_context(emotion_limit, user_id),
            "gesture": self.get_recent_data(SensorType.GESTURE, user_id, limit=5),
            "environment": self.get_recent_data(SensorType.ENVIRONMENT, user_id, limit=3)
        }
    
    def _cleanup_old_audio(self):
        """Очистка старых аудио данных (старше audio_buffer_seconds)"""
        cutoff_time = datetime.now() - timedelta(seconds=self.audio_buffer_seconds)
        
        # Удаляем старые записи (deque автоматически ограничивает размер)
        while self._audio_buffer and self._audio_buffer[0].timestamp < cutoff_time:
            old_data = self._audio_buffer.popleft()
            self._total_evicted += 1
            logger.debug(f"Удален старый аудио фрагмент: {old_data.id}")
    
    def _cleanup_old_video(self):
        """Очистка старых видео данных"""
        cutoff_time = datetime.now() - timedelta(seconds=self.audio_buffer_seconds)
        
        while self._video_buffer and self._video_buffer[0].timestamp < cutoff_time:
            old_data = self._video_buffer.popleft()
            self._total_evicted += 1
            logger.debug(f"Удален старый видео фрагмент: {old_data.id}")
    
    def get_total_size(self) -> int:
        """Получение общего размера всех буферов"""
        with self._lock:
            return (len(self._audio_buffer) + 
                   len(self._video_buffer) + 
                   len(self._text_buffer) + 
                   len(self._emotion_buffer) + 
                   len(self._gesture_buffer) + 
                   len(self._environment_buffer))
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Получение статистики буферов"""
        with self._lock:
            return {
                "audio_buffer_size": len(self._audio_buffer),
                "video_buffer_size": len(self._video_buffer),
                "text_buffer_size": len(self._text_buffer),
                "emotion_buffer_size": len(self._emotion_buffer),
                "gesture_buffer_size": len(self._gesture_buffer),
                "environment_buffer_size": len(self._environment_buffer),
                "total_size": self.get_total_size(),
                "total_added": self._total_added,
                "total_evicted": self._total_evicted,
                "audio_buffer_seconds": self.audio_buffer_seconds,
                "text_buffer_size": self.text_buffer_size,
                "emotion_buffer_size": self.emotion_buffer_size
            }
    
    def clear_buffer(self, sensor_type: Optional[SensorType] = None, user_id: Optional[str] = None):
        """Очистка буферов"""
        with self._lock:
            if sensor_type is None:
                # Очищаем все буферы
                self._audio_buffer.clear()
                self._video_buffer.clear()
                self._text_buffer.clear()
                self._emotion_buffer.clear()
                self._gesture_buffer.clear()
                self._environment_buffer.clear()
                logger.info("Все буферы очищены")
            else:
                # Очищаем конкретный буфер
                buffer_map = {
                    SensorType.AUDIO: self._audio_buffer,
                    SensorType.VIDEO: self._video_buffer,
                    SensorType.TEXT: self._text_buffer,
                    SensorType.EMOTION: self._emotion_buffer,
                    SensorType.GESTURE: self._gesture_buffer,
                    SensorType.ENVIRONMENT: self._environment_buffer
                }
                
                if sensor_type in buffer_map:
                    if user_id:
                        # Удаляем только записи конкретного пользователя
                        buffer = buffer_map[sensor_type]
                        buffer[:] = [data for data in buffer if data.user_id != user_id]
                    else:
                        # Очищаем весь буфер
                        buffer_map[sensor_type].clear()
                    
                    logger.info(f"Буфер {sensor_type.value} очищен для пользователя {user_id or 'всех'}")
    
    def export_data(self, user_id: str = "default") -> Dict[str, List[Dict[str, Any]]]:
        """Экспорт данных для передачи в Working Memory"""
        with self._lock:
            context = self.get_combined_context(user_id=user_id)
            
            # Преобразуем в словари для сериализации
            export_data = {}
            for sensor_type, data_list in context.items():
                export_data[sensor_type] = [data.to_dict() for data in data_list]
            
            return export_data

# Глобальный экземпляр Sensor Buffer
_sensor_buffer_instance: Optional[SensorBuffer] = None

def get_sensor_buffer() -> SensorBuffer:
    """Получение глобального экземпляра Sensor Buffer"""
    global _sensor_buffer_instance
    if _sensor_buffer_instance is None:
        _sensor_buffer_instance = SensorBuffer()
    return _sensor_buffer_instance

def add_audio_data(data: Any, user_id: str = "default", **kwargs) -> str:
    """Быстрое добавление аудио данных"""
    return get_sensor_buffer().add_sensor_data(
        SensorType.AUDIO, data, user_id, **kwargs
    )

def add_text_data(text: str, user_id: str = "default", **kwargs) -> str:
    """Быстрое добавление текстовых данных"""
    return get_sensor_buffer().add_sensor_data(
        SensorType.TEXT, text, user_id, text_content=text, **kwargs
    )

def add_emotion_data(emotion: str, confidence: float, user_id: str = "default", **kwargs) -> str:
    """Быстрое добавление эмоциональных данных"""
    return get_sensor_buffer().add_sensor_data(
        SensorType.EMOTION, emotion, user_id, emotion=emotion, confidence=confidence, **kwargs
    )
