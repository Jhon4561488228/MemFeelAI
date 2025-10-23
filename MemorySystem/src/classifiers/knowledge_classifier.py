"""
Knowledge classifier for Semantic Memory.
Classifies text into one of predefined categories using the LLM provider.
"""

from __future__ import annotations

from typing import List

try:
    from ..providers.lm_studio_provider import LMStudioProvider
except ImportError:  # pragma: no cover
    from providers.lm_studio_provider import LMStudioProvider


DEFAULT_CATEGORIES: List[str] = [
    "personal",        # личная информация
    "professional",    # профессиональная информация
    "preferences",     # предпочтения
    "relationships",   # отношения
    "skills",          # навыки
    "general",         # общее знание
]


class KnowledgeClassifier:
    def __init__(self, config_path: str = "config/lm_studio_config.yaml") -> None:
        self.provider = LMStudioProvider(config_path)

    async def classify(self, content: str, categories: List[str] | None = None) -> str:
        cats = categories or DEFAULT_CATEGORIES
        prompt = (
            "Определи наиболее подходящую категорию для следующего знания. "
            "Выбери только одно слово из списка: " + ", ".join(cats) + ".\n\n" 
            f"Текст: {content[:800]}\n\n"  # ограничим размер
            "Ответ:"
        )
        try:
            res = await self.provider.generate_text(prompt, task_type="analysis", max_tokens=8, temperature=0.0)
            label = (res or "").strip().lower()
            for c in cats:
                if c.lower() in label:
                    return c
        except Exception:
            pass
        return "general"

