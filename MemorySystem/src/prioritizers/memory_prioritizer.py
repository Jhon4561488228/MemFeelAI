from datetime import datetime
from typing import Optional, Dict, Any


def compute_priority(
    base_importance: float,
    last_accessed: Optional[datetime] = None,
    access_count: int = 0,
    emotional_intensity: Optional[float] = None,
    recency_hours: Optional[float] = None,
) -> float:
    """Расчёт приоритета с учётом доступа/эмоций/давности.

    Простая формула: 60% база, 20% эмоции, 20% недавность, + лёгкий бонус за частый доступ.
    """
    importance = max(0.0, min(1.0, base_importance))
    emo = 0.0 if emotional_intensity is None else max(0.0, min(1.0, emotional_intensity))
    recency_factor = 0.0
    if recency_hours is not None:
        # чем свежее (меньше часов), тем выше фактор
        recency_factor = max(0.0, min(1.0, 1.0 - min(recency_hours, 24.0) / 24.0))
    access_bonus = min(0.2, access_count / 50.0)
    score = importance * 0.6 + emo * 0.2 + recency_factor * 0.2 + access_bonus
    return max(0.0, min(1.0, score))

