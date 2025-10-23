"""
🎭 Форматтер эмоций для Memory System
Создает адаптивные промпты для ИИ на основе сложности эмоциональных данных
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionComplexity(Enum):
    """Уровни сложности эмоций"""
    SIMPLE = "simple"      # Одна доминирующая эмоция
    MEDIUM = "medium"      # Две эмоции с разной интенсивностью
    COMPLEX = "complex"    # Множественные эмоции или конфликты

@dataclass
class EmotionData:
    """Данные об эмоциях"""
    primary_emotion: str
    primary_confidence: float
    secondary_emotion: Optional[str] = None
    secondary_confidence: Optional[float] = None
    tertiary_emotion: Optional[str] = None
    tertiary_confidence: Optional[float] = None
    consistency: str = "high"
    dominant_source: str = "voice"
    validation_applied: bool = False

@dataclass
class FormattedEmotion:
    """Отформатированная эмоция для ИИ"""
    emotion_prompt: str
    complexity: EmotionComplexity
    context_hints: List[str]
    metadata: Dict[str, any]

class EmotionFormatter:
    """Форматтер эмоций для создания адаптивных промптов"""
    
    def __init__(self):
        # Пороги для определения сложности
        self.complexity_thresholds = {
            "simple": 0.8,    # Одна эмоция > 80%
            "medium": 0.6,    # Две эмоции > 60%
            "complex": 0.4    # Множественные эмоции или конфликты
        }
        
        # Контекстные подсказки для разных ситуаций
        self.context_hints = {
            "mixed_feelings": [
                "Пользователь испытывает смешанные чувства",
                "Одновременно присутствуют разные эмоции",
                "Эмоциональное состояние сложное и многогранное"
            ],
            "voice_text_conflict": [
                "Голос и текст передают разные эмоции",
                "Возможно, пользователь скрывает истинные чувства",
                "Обратите внимание на несоответствие между голосом и словами"
            ],
            "low_confidence": [
                "Эмоциональное состояние неопределенное",
                "Низкая уверенность в определении эмоций",
                "Требуется дополнительное внимание к контексту"
            ],
            "validation_applied": [
                "Эмоции были дополнительно проверены",
                "Результат подтвержден через валидацию",
                "Высокая достоверность эмоционального анализа"
            ],
            "neutral_dominant": [
                "Преобладает нейтральное эмоциональное состояние",
                "Пользователь спокоен и уравновешен",
                "Отсутствуют ярко выраженные эмоции"
            ]
        }
        
        # Шаблоны промптов для разных уровней сложности
        self.prompt_templates = {
            EmotionComplexity.SIMPLE: {
                "template": "Эмоция: {emotion} (уверенность: {confidence:.1%})",
                "description": "Простая эмоция с высокой уверенностью"
            },
            EmotionComplexity.MEDIUM: {
                "template": "Эмоции: {primary_emotion} ({primary_confidence:.1%}) + {secondary_emotion} ({secondary_confidence:.1%})",
                "description": "Две эмоции с разной интенсивностью"
            },
            EmotionComplexity.COMPLEX: {
                "template": "Сложное эмоциональное состояние: {primary_emotion} ({primary_confidence:.1%}), {secondary_emotion} ({secondary_confidence:.1%}), {tertiary_emotion} ({tertiary_confidence:.1%})",
                "description": "Множественные эмоции или конфликты"
            }
        }
    
    def format_emotions_for_ai(self, emotion_data: EmotionData) -> FormattedEmotion:
        """Форматирует эмоции для передачи ИИ"""
        
        # Определяем сложность
        complexity = self._calculate_complexity(emotion_data)
        
        # Создаем промпт
        emotion_prompt = self._create_emotion_prompt(emotion_data, complexity)
        
        # Генерируем контекстные подсказки
        context_hints = self._generate_context_hints(emotion_data, complexity)
        
        # Создаем метаданные
        metadata = self._create_metadata(emotion_data, complexity)
        
        return FormattedEmotion(
            emotion_prompt=emotion_prompt,
            complexity=complexity,
            context_hints=context_hints,
            metadata=metadata
        )
    
    def _calculate_complexity(self, emotion_data: EmotionData) -> EmotionComplexity:
        """Вычисляет сложность эмоционального состояния"""
        
        primary_confidence = emotion_data.primary_confidence
        secondary_confidence = emotion_data.secondary_confidence or 0.0
        tertiary_confidence = emotion_data.tertiary_confidence or 0.0
        
        # Проверяем наличие множественных эмоций
        has_secondary = emotion_data.secondary_emotion is not None
        has_tertiary = emotion_data.tertiary_emotion is not None
        
        # Проверяем конфликты
        has_conflicts = self._has_emotional_conflicts(emotion_data)
        
        # Определяем сложность
        if has_conflicts or (has_secondary and has_tertiary):
            return EmotionComplexity.COMPLEX
        elif has_secondary and secondary_confidence > self.complexity_thresholds["medium"]:
            return EmotionComplexity.MEDIUM
        elif primary_confidence > self.complexity_thresholds["simple"]:
            return EmotionComplexity.SIMPLE
        else:
            return EmotionComplexity.MEDIUM
    
    def _has_emotional_conflicts(self, emotion_data: EmotionData) -> bool:
        """Проверяет наличие эмоциональных конфликтов"""
        
        # Маппинг эмоций к категориям
        emotion_categories = {
            "радость": "positive", "энтузиазм": "positive", "удивление": "positive",
            "грусть": "negative", "злость": "negative", "страх": "negative", "отвращение": "negative",
            "нейтральная": "neutral", "спокойствие": "neutral"
        }
        
        emotions = [emotion_data.primary_emotion]
        if emotion_data.secondary_emotion:
            emotions.append(emotion_data.secondary_emotion)
        if emotion_data.tertiary_emotion:
            emotions.append(emotion_data.tertiary_emotion)
        
        categories = [emotion_categories.get(emotion, "neutral") for emotion in emotions]
        unique_categories = set(categories)
        
        # Конфликт если есть противоположные категории
        return len(unique_categories) > 2 or (
            "positive" in unique_categories and "negative" in unique_categories
        )
    
    def _create_emotion_prompt(self, emotion_data: EmotionData, complexity: EmotionComplexity) -> str:
        """Создает промпт для ИИ"""
        
        template_info = self.prompt_templates[complexity]
        
        if complexity == EmotionComplexity.SIMPLE:
            return template_info["template"].format(
                emotion=emotion_data.primary_emotion,
                confidence=emotion_data.primary_confidence
            )
        
        elif complexity == EmotionComplexity.MEDIUM:
            secondary_emotion = emotion_data.secondary_emotion or "неопределенная"
            secondary_confidence = emotion_data.secondary_confidence or 0.0
            
            return template_info["template"].format(
                primary_emotion=emotion_data.primary_emotion,
                primary_confidence=emotion_data.primary_confidence,
                secondary_emotion=secondary_emotion,
                secondary_confidence=secondary_confidence
            )
        
        else:  # COMPLEX
            secondary_emotion = emotion_data.secondary_emotion or "неопределенная"
            secondary_confidence = emotion_data.secondary_confidence or 0.0
            tertiary_emotion = emotion_data.tertiary_emotion or "неопределенная"
            tertiary_confidence = emotion_data.tertiary_confidence or 0.0
            
            return template_info["template"].format(
                primary_emotion=emotion_data.primary_emotion,
                primary_confidence=emotion_data.primary_confidence,
                secondary_emotion=secondary_emotion,
                secondary_confidence=secondary_confidence,
                tertiary_emotion=tertiary_emotion,
                tertiary_confidence=tertiary_confidence
            )
    
    def _generate_context_hints(self, emotion_data: EmotionData, complexity: EmotionComplexity) -> List[str]:
        """Генерирует контекстные подсказки"""
        
        hints = []
        
        # Проверяем смешанные чувства
        if complexity == EmotionComplexity.COMPLEX:
            hints.extend(self.context_hints["mixed_feelings"])
        
        # Проверяем конфликты между голосом и текстом
        if emotion_data.consistency == "low":
            hints.extend(self.context_hints["voice_text_conflict"])
        
        # Проверяем низкую уверенность
        if emotion_data.primary_confidence < 0.6:
            hints.extend(self.context_hints["low_confidence"])
        
        # Проверяем применение валидации
        if emotion_data.validation_applied:
            hints.extend(self.context_hints["validation_applied"])
        
        # Проверяем нейтральное состояние
        if emotion_data.primary_emotion == "нейтральная":
            hints.extend(self.context_hints["neutral_dominant"])
        
        return hints
    
    def _create_metadata(self, emotion_data: EmotionData, complexity: EmotionComplexity) -> Dict[str, any]:
        """Создает метаданные для эмоций"""
        
        return {
            "complexity": complexity.value,
            "primary_emotion": emotion_data.primary_emotion,
            "primary_confidence": emotion_data.primary_confidence,
            "secondary_emotion": emotion_data.secondary_emotion,
            "secondary_confidence": emotion_data.secondary_confidence,
            "tertiary_emotion": emotion_data.tertiary_emotion,
            "tertiary_confidence": emotion_data.tertiary_confidence,
            "consistency": emotion_data.consistency,
            "dominant_source": emotion_data.dominant_source,
            "validation_applied": emotion_data.validation_applied,
            "has_conflicts": self._has_emotional_conflicts(emotion_data),
            "emotion_count": self._count_emotions(emotion_data)
        }
    
    def _count_emotions(self, emotion_data: EmotionData) -> int:
        """Подсчитывает количество эмоций"""
        count = 1  # primary всегда есть
        
        if emotion_data.secondary_emotion:
            count += 1
        if emotion_data.tertiary_emotion:
            count += 1
        
        return count
    
    def create_adaptive_prompt(self, formatted_emotion: FormattedEmotion, base_prompt: str) -> str:
        """Создает адаптивный промпт на основе сложности эмоций"""
        
        # Базовый промпт
        prompt_parts = [base_prompt]
        
        # Добавляем эмоциональную информацию
        prompt_parts.append(f"\nЭмоциональный контекст: {formatted_emotion.emotion_prompt}")
        
        # Добавляем контекстные подсказки
        if formatted_emotion.context_hints:
            prompt_parts.append(f"\nДополнительные подсказки:")
            for hint in formatted_emotion.context_hints:
                prompt_parts.append(f"- {hint}")
        
        # Добавляем инструкции в зависимости от сложности
        if formatted_emotion.complexity == EmotionComplexity.SIMPLE:
            prompt_parts.append(f"\nИнструкция: Пользователь испытывает четкую эмоцию. Отвечайте соответственно.")
        
        elif formatted_emotion.complexity == EmotionComplexity.MEDIUM:
            prompt_parts.append(f"\nИнструкция: Пользователь испытывает смешанные чувства. Учтите обе эмоции в ответе.")
        
        else:  # COMPLEX
            prompt_parts.append(f"\nИнструкция: Сложное эмоциональное состояние. Будьте особенно внимательны и эмпатичны.")
        
        return "\n".join(prompt_parts)
    
    def format_for_memory_storage(self, formatted_emotion: FormattedEmotion) -> Dict[str, any]:
        """Форматирует эмоции для хранения в памяти"""
        
        return {
            "emotion_summary": formatted_emotion.emotion_prompt,
            "complexity": formatted_emotion.complexity.value,
            "context_hints": formatted_emotion.context_hints,
            "metadata": formatted_emotion.metadata,
            "formatted_at": "2025-09-29T18:00:00Z"  # В реальном коде использовать datetime.now().isoformat()
        }
    
    def get_formatter_stats(self) -> Dict[str, any]:
        """Получение статистики форматтера"""
        return {
            "complexity_thresholds": self.complexity_thresholds,
            "context_hints_categories": list(self.context_hints.keys()),
            "prompt_templates_count": len(self.prompt_templates),
            "supported_complexities": [c.value for c in EmotionComplexity]
        }

# Функция для тестирования
def test_emotion_formatter():
    """Тест форматтера эмоций"""
    print("🧪 Тестируем форматтер эмоций...")
    
    formatter = EmotionFormatter()
    
    # Тестовые случаи
    test_cases = [
        {
            "name": "Простая эмоция",
            "data": EmotionData(
                primary_emotion="радость",
                primary_confidence=0.95,
                consistency="high",
                dominant_source="voice"
            )
        },
        {
            "name": "Смешанные чувства",
            "data": EmotionData(
                primary_emotion="радость",
                primary_confidence=0.70,
                secondary_emotion="грусть",
                secondary_confidence=0.60,
                consistency="medium",
                dominant_source="voice"
            )
        },
        {
            "name": "Сложное состояние",
            "data": EmotionData(
                primary_emotion="радость",
                primary_confidence=0.50,
                secondary_emotion="грусть",
                secondary_confidence=0.45,
                tertiary_emotion="страх",
                tertiary_confidence=0.40,
                consistency="low",
                dominant_source="voice",
                validation_applied=True
            )
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Тест {i}: {case['name']}")
        
        formatted = formatter.format_emotions_for_ai(case['data'])
        
        print(f"   Сложность: {formatted.complexity.value}")
        print(f"   Промпт: {formatted.emotion_prompt}")
        print(f"   Подсказки: {len(formatted.context_hints)}")
        
        if formatted.context_hints:
            for hint in formatted.context_hints:
                print(f"     - {hint}")
        
        # Тест адаптивного промпта
        base_prompt = "Проанализируй эмоциональное состояние пользователя и дай совет."
        adaptive_prompt = formatter.create_adaptive_prompt(formatted, base_prompt)
        
        print(f"   Адаптивный промпт: {len(adaptive_prompt)} символов")
        
        # Тест для хранения в памяти
        memory_format = formatter.format_for_memory_storage(formatted)
        print(f"   Метаданные: {len(memory_format['metadata'])} полей")
    
    # Статистика
    stats = formatter.get_formatter_stats()
    print(f"\n📊 Статистика форматтера:")
    print(f"   Пороги сложности: {stats['complexity_thresholds']}")
    print(f"   Категории подсказок: {stats['context_hints_categories']}")
    print(f"   Поддерживаемые сложности: {stats['supported_complexities']}")

if __name__ == "__main__":
    test_emotion_formatter()
