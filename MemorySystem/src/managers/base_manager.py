from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMemoryManager(ABC):
    """Базовый интерфейс менеджера памяти."""

    @abstractmethod
    async def add(self, content: str, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:  # noqa: D401
        pass

    @abstractmethod
    async def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def search(self, user_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        pass

    @abstractmethod
    async def cleanup(self, user_id: str) -> int:
        pass

    @abstractmethod
    async def stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        pass

