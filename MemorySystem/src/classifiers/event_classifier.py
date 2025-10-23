"""
EventClassifier: классификация событий (conversation, meeting, task, achievement, problem).
Использует LMStudioProvider (Ollama) с task_type="analysis" для лёгкости.
"""

from typing import Literal
from loguru import logger

try:
    from ..providers.lm_studio_provider import LMStudioProvider
except ImportError:
    try:
        from src.providers.lm_studio_provider import LMStudioProvider
    except ImportError:
        from providers.lm_studio_provider import LMStudioProvider

EventType = Literal["conversation", "meeting", "task", "achievement", "problem", "general"]

CLASSIFY_PROMPT = (
    "Определи тип события из списка: conversation, meeting, task, achievement, problem.\n"
    "Верни ровно одно слово из списка.\n\n"
    "Событие: {content}"
)

class EventClassifier:
    def __init__(self, provider: LMStudioProvider | None = None):
        # Используем локальный относительный путь к конфигу
        self.provider = provider or LMStudioProvider("config/lm_studio_config.yaml")

    async def classify(self, content: str) -> EventType:
        try:
            prompt = CLASSIFY_PROMPT.format(content=content.strip())
            raw = await self.provider.generate_text(prompt, task_type="analysis")
            label = (raw or "").strip().lower()
            if label in {"conversation", "meeting", "task", "achievement", "problem"}:
                return label  # type: ignore
            return "general"
        except Exception as e:
            logger.error(f"Event classification failed: {e}")
            return "general"
