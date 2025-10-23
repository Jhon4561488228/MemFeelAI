"""
Улучшенный Emotion Combiner для AIRI Memory System
Перенесенная и улучшенная логика из Rust emotion-engine
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EmotionConsistency(Enum):
    """Уровни согласованности эмоций"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SINGLE_SOURCE = "single_source"

@dataclass
class VoiceEmotionResult:
    """Результат анализа голосовых эмоций"""
    emotion: str
    confidence: float
    source: str = "voice"

@dataclass
class TextEmotionResult:
    """Результат анализа текстовых эмоций"""
    primary_emotion: str
    primary_confidence: float
    secondary_emotion: str
    secondary_confidence: float
    consistency: str
    source: str = "text_combined"

@dataclass
class CombinedEmotionResult:
    """Финальный результат комбинирования эмоций"""
    emotion: str
    confidence: float
    source: str
    secondary_emotion: Optional[str]
    secondary_confidence: Optional[float]
    consistency: str
    dominant_source: str

class EmotionCombiner:
    """Улучшенный комбинер эмоций с логикой из Rust emotion-engine"""
    
    def __init__(self):
        # Константы весов из Rust emotion-engine
        self.VOICE_WEIGHT = 0.6        # Увеличено с 0.4 до 0.6
        self.TEXT_ANIEMORE_WEIGHT = 0.3  # Остается 0.3
        self.TEXT_DOSTOEVSKY_WEIGHT = 0.1  # Уменьшено с 0.3 до 0.1
        
        # Маппинг эмоций Aniemore в категории для сравнения с Dostoevsky
        self.emotion_categories = {
            # Положительные эмоции
            "радость": "positive", "joy": "positive", "happiness": "positive", "счастье": "positive",
            "энтузиазм": "positive", "enthusiasm": "positive", "восторг": "positive",
            "удивление": "positive", "surprise": "positive", "изумление": "positive",
            
            # Отрицательные эмоции
            "злость": "negative", "anger": "negative", "гнев": "negative", "ярость": "negative",
            "страх": "negative", "fear": "negative", "боязнь": "negative", "тревога": "negative",
            "грусть": "negative", "sadness": "negative", "печаль": "negative", "тоска": "negative",
            "отвращение": "negative", "disgust": "negative", "омерзение": "negative",
            
            # Нейтральные эмоции
            "нейтральная": "neutral", "neutral": "neutral", "спокойствие": "neutral", "calm": "neutral",
        }
        
        logger.info("EmotionCombiner инициализирован с весами из Rust emotion-engine")
    
    def map_aniemore_emotion_to_category(self, emotion: str) -> str:
        """Маппинг эмоций Aniemore в категории для сравнения с Dostoevsky"""
        emotion_lower = emotion.lower()
        return self.emotion_categories.get(emotion_lower, "neutral")
    
    def validate_emotion_tonality(
        self,
        aniemore_emotion: str,
        aniemore_confidence: float,
        dostoevsky_emotion: str,
        dostoevsky_confidence: float
    ) -> Tuple[bool, float, str]:
        """Валидация тональности между Aniemore и Dostoevsky"""
        aniemore_category = self.map_aniemore_emotion_to_category(aniemore_emotion)
        dostoevsky_category = dostoevsky_emotion.lower()
        
        # Проверяем совпадение категорий
        categories_match = aniemore_category == dostoevsky_category
        
        if categories_match:
            # Категории совпадают - валидация успешна
            confidence_boost = 0.1  # Небольшое повышение confidence
            return (
                True,
                confidence_boost,
                f"Тональность валидна: {aniemore_emotion} ({aniemore_category}) совпадает с {dostoevsky_emotion} ({dostoevsky_category})"
            )
        else:
            # Категории не совпадают - валидация неуспешна
            confidence_penalty = -0.2  # Понижение confidence
            return (
                False,
                confidence_penalty,
                f"Конфликт тональности: {aniemore_emotion} ({aniemore_category}) не совпадает с {dostoevsky_emotion} ({dostoevsky_category})"
            )
    
    def combine_emotions_final(
        self,
        voice_result: Optional[VoiceEmotionResult],
        text_result: Optional[TextEmotionResult],
        transcription: str = ""
    ) -> CombinedEmotionResult:
        """Финальное объединение эмоций с логикой из Rust emotion-engine"""
        logger.debug("Combining emotions from two sources with weights: voice + combined_text")
        
        if not voice_result and not text_result:
            # Нет данных - возвращаем нейтральную эмоцию
            return CombinedEmotionResult(
                emotion="нейтральная",
                confidence=0.0,
                source="none",
                secondary_emotion=None,
                secondary_confidence=None,
                consistency="single_source",
                dominant_source="none"
            )
        
        # Вычисляем взвешенные уверенности для двух источников
        voice_weighted_confidence = 0.0
        if voice_result:
            voice_weighted_confidence = voice_result.confidence * self.VOICE_WEIGHT
        
        text_weighted_confidence = 0.0
        if text_result:
            # Обновленная логика: Aniemore и Dostoevsky имеют разные веса
            aniemore_weighted_confidence = text_result.primary_confidence * self.TEXT_ANIEMORE_WEIGHT
            dostoevsky_weighted_confidence = text_result.secondary_confidence * self.TEXT_DOSTOEVSKY_WEIGHT
            text_weighted_confidence = aniemore_weighted_confidence + dostoevsky_weighted_confidence
        
        # Логируем веса для отладки
        logger.debug(f"Voice confidence: {voice_result.confidence if voice_result else 0.0:.3f} * {self.VOICE_WEIGHT:.1f} = {voice_weighted_confidence:.3f}")
        if text_result:
            logger.debug(f"Aniemore confidence: {text_result.primary_confidence:.3f} * {self.TEXT_ANIEMORE_WEIGHT:.1f} = {text_result.primary_confidence * self.TEXT_ANIEMORE_WEIGHT:.3f}")
            logger.debug(f"Dostoevsky confidence: {text_result.secondary_confidence:.3f} * {self.TEXT_DOSTOEVSKY_WEIGHT:.1f} = {text_result.secondary_confidence * self.TEXT_DOSTOEVSKY_WEIGHT:.3f}")
        logger.debug(f"Total text confidence: {text_weighted_confidence:.3f}")
        
        # Валидация тональности между Aniemore и Dostoevsky
        confidence_adjustment = 0.0
        if text_result:
            is_tonality_valid, confidence_adjustment, validation_message = self.validate_emotion_tonality(
                text_result.primary_emotion,
                text_result.primary_confidence,
                text_result.secondary_emotion,
                text_result.secondary_confidence
            )
            logger.debug(f"Валидация тональности: {validation_message}")
        
        # Применяем корректировку confidence к текстовому анализу
        adjusted_text_confidence = text_weighted_confidence + confidence_adjustment
        
        # Определяем доминирующий источник с учетом валидации
        if voice_weighted_confidence > adjusted_text_confidence:
            final_confidence = voice_weighted_confidence
            final_emotion = voice_result.emotion
            dominant_source = "voice"
        else:
            # Используем базовый text_weighted_confidence для финального результата
            # Валидация влияет только на выбор доминирующего источника
            final_confidence = text_weighted_confidence
            final_emotion = text_result.primary_emotion if text_result else "нейтральная"
            dominant_source = "text_combined"
        
        # Определяем вторичную эмоцию
        secondary_emotion = None
        secondary_confidence = None
        
        if voice_result and text_result:
            if voice_weighted_confidence > text_weighted_confidence:
                # Voice доминирует, вторичная - от текста
                if text_result.primary_emotion != final_emotion:
                    secondary_emotion = text_result.primary_emotion
                    secondary_confidence = text_result.primary_confidence
                elif text_result.primary_emotion != text_result.secondary_emotion:
                    secondary_emotion = text_result.secondary_emotion
                    secondary_confidence = text_result.secondary_confidence
            else:
                # Text доминирует, вторичная - от голоса
                if voice_result.emotion != final_emotion:
                    secondary_emotion = voice_result.emotion
                    secondary_confidence = voice_result.confidence
        elif text_result and not voice_result:
            # Только текст - вторичная от secondary
            if text_result.primary_emotion != text_result.secondary_emotion:
                secondary_emotion = text_result.secondary_emotion
                secondary_confidence = text_result.secondary_confidence
        
        # Определяем согласованность
        consistency = self.calculate_consistency_final(voice_result, text_result)
        
        return CombinedEmotionResult(
            emotion=final_emotion,
            confidence=final_confidence,
            source=dominant_source,
            secondary_emotion=secondary_emotion,
            secondary_confidence=secondary_confidence,
            consistency=consistency,
            dominant_source=dominant_source
        )
    
    def calculate_consistency_final(
        self,
        voice_result: Optional[VoiceEmotionResult],
        text_result: Optional[TextEmotionResult]
    ) -> str:
        """Вычисление согласованности между двумя источниками эмоций"""
        if not voice_result or not text_result:
            return "single_source"
        
        # Проверяем согласованность между голосом и объединенным текстом
        voice_emotion = voice_result.emotion
        text_primary_emotion = text_result.primary_emotion
        text_secondary_emotion = text_result.secondary_emotion
        
        # Вычисляем согласованность
        if voice_emotion == text_primary_emotion:
            return "high"  # Голос и основная текстовая эмоция совпадают
        elif voice_emotion == text_secondary_emotion:
            return "medium"  # Голос совпадает со вторичной текстовой эмоцией
        elif text_primary_emotion == text_secondary_emotion:
            return "medium"  # Текстовые эмоции совпадают, но голос отличается
        else:
            # Проверяем схожесть эмоций по категориям
            voice_text_similarity = self.calculate_emotion_similarity(voice_emotion, text_primary_emotion)
            if voice_text_similarity > 0.7:
                return "medium"
            else:
                return "low"
    
    def calculate_emotion_similarity(self, emotion1: str, emotion2: str) -> float:
        """Вычисление схожести между эмоциями"""
        # Маппинг эмоций к категориям
        emotion_categories = [
            (["радость", "joy", "счастье"], "positive"),
            (["грусть", "sadness", "печаль"], "negative"),
            (["злость", "anger", "гнев"], "negative"),
            (["страх", "fear", "боязнь"], "negative"),
            (["удивление", "surprise", "изумление"], "positive"),
            (["отвращение", "disgust", "неприязнь"], "negative"),
            (["нейтральная", "neutral", "спокойствие"], "neutral"),
        ]
        
        def get_category(emotion: str) -> str:
            emotion_lower = emotion.lower()
            for emotions, category in emotion_categories:
                if any(e in emotion_lower for e in emotions):
                    return category
            return "unknown"
        
        category1 = get_category(emotion1)
        category2 = get_category(emotion2)
        
        if category1 == category2:
            return 0.5  # Частичная согласованность
        else:
            return 0.0  # Нет согласованности

# Глобальный экземпляр комбинера
emotion_combiner = EmotionCombiner()
