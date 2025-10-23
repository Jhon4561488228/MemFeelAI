"""
FactExtractor: извлечение фактов из текста с помощью локального LLM (Ollama).
Использует провайдер LMStudioProvider (настроен на Ollama API) с task_type="fact_extraction".
"""

from typing import List
from loguru import logger

try:
    from ..providers.lm_studio_provider import LMStudioProvider
except ImportError:
    try:
        from src.providers.lm_studio_provider import LMStudioProvider
    except ImportError:
        from providers.lm_studio_provider import LMStudioProvider

FACT_EXTRACTION_PROMPT = (
    "Выдели только проверяемые факты из текста ниже. "
    "Факт — это утверждение, которое можно проверить. "
    "Верни список фактов, по одному на строку, без нумерации и комментариев.\n\n"
    "Текст:\n{content}\n\n"
    "Факты:"
)

class FactExtractor:
    def __init__(self, provider: LMStudioProvider | None = None):
        # Используем локальный относительный путь к конфигу
        self.provider = provider or LMStudioProvider("config/lm_studio_config.yaml")

    async def extract_facts(self, content: str) -> List[str]:
        try:
            prompt = FACT_EXTRACTION_PROMPT.format(content=content.strip())
            raw = await self.provider.generate_text(prompt, task_type="fact_extraction")
            lines = [line.strip("- ") for line in raw.splitlines()]
            facts = [l for l in (line.strip() for line in lines) if l]
            # простой пост-фильтр: убираем строки, которые слишком короткие
            facts = [f for f in facts if len(f) > 3]
            logger.debug(f"Extracted {len(facts)} facts")
            return facts
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return []
