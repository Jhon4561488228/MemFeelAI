"""
Локальная версия Mem0 для AIRI Memory System
Полностью локальная реализация без зависимостей от OpenAI
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid
from loguru import logger
import yaml
from pathlib import Path

from ..providers.lm_studio_provider import LMStudioProvider
from ..providers.chromadb_provider import ChromaDBProvider
from ..providers.embeddings_provider import EmbeddingsProvider

class LocalMem0:
    """Локальная реализация Mem0 с поддержкой LM Studio и ChromaDB"""
    
    def __init__(self, config_path: str = "config/mem0_config.yaml"):
        """Инициализация локального Mem0"""
        self.config = self._load_config(config_path)
        self.llm_provider = None
        self.embedder = None
        self.vector_store = None
        self._initialize_providers()
        logger.info("Local Mem0 инициализирован")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Конфигурационный файл {config_path} не найден")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию"""
        return {
            "llm": {
                "provider": "lm_studio",
                "config": "config/lm_studio_config.yaml"
            },
            "embedder": {
                "provider": "sentence_transformers",
                "config": "config/embeddings_config.yaml"
            },
            "vector_store": {
                "provider": "chromadb",
                "config": "config/chromadb_config.yaml"
            },
            "memory_settings": {
                "max_memories": 10000,
                "cleanup_threshold": 0.8,
                "similarity_threshold": 0.7,
                "retention_days": 365
            }
        }
    
    def _initialize_providers(self):
        """Инициализация провайдеров"""
        try:
            # Инициализируем LLM провайдер
            llm_config = self.config["llm"]
            if llm_config["provider"] == "lm_studio":
                config_path = llm_config.get("config", "config/lm_studio_config.yaml")
                self.llm_provider = LMStudioProvider(config_path)
            else:
                raise ValueError(f"Неподдерживаемый LLM провайдер: {llm_config['provider']}")
            
            # Инициализируем провайдер эмбеддингов
            embedder_config = self.config["embedder"]
            if embedder_config["provider"] == "sentence_transformers":
                config_path = embedder_config.get("config", "config/embeddings_config.yaml")
                self.embedder = EmbeddingsProvider(config_path)
            else:
                raise ValueError(f"Неподдерживаемый провайдер эмбеддингов: {embedder_config['provider']}")
            
            # Инициализируем векторное хранилище
            vector_config = self.config["vector_store"]
            if vector_config["provider"] == "chromadb":
                config_path = vector_config.get("config", "config/chromadb_config.yaml")
                self.vector_store = ChromaDBProvider(config_path)
            else:
                raise ValueError(f"Неподдерживаемое векторное хранилище: {vector_config['provider']}")
            
            logger.info("Все провайдеры инициализированы")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации провайдеров: {e}")
            raise
    
    async def add_memory(
        self, 
        content: str, 
        user_id: str = "default_user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Добавление воспоминания"""
        try:
            if not content.strip():
                raise ValueError("Содержимое воспоминания не может быть пустым")
            
            # Проверяем размер контента и выбираем подходящий метод анализа
            max_content_length = self.llm_provider.config.get("stability", {}).get("max_content_length", 4000)
            
            if len(content) > max_content_length:
                logger.info(f"Контент большой ({len(content)} символов), используем chunked анализ")
                # Используем новый метод для анализа больших контентов
                analysis = await self.llm_provider.analyze_large_content(content)
            else:
                # Используем обычный метод для небольших контентов
                analysis = await self.llm_provider.analyze_memory_content(content)
            
            # Генерируем эмбеддинг
            embedding = await self.embedder.generate_embedding(content)
            
            # Подготавливаем метаданные
            memory_metadata = {
                "type": analysis.get("type", "conversation"),
                "importance": analysis.get("importance", "medium"),
                "sentiment": analysis.get("sentiment", "neutral"),
                "keywords": analysis.get("keywords", []),
                "entities": analysis.get("entities", []),
                "summary": analysis.get("summary", content[:100])
            }
            
            # Добавляем пользовательские метаданные
            if metadata:
                memory_metadata.update(metadata)
            
            # Сохраняем в векторное хранилище
            memory_id = await self.vector_store.add_memory(
                content=content,
                embedding=embedding,
                user_id=user_id,
                metadata=memory_metadata
            )
            
            # Возвращаем результат
            result = {
                "id": memory_id,
                "content": content,
                "user_id": user_id,
                "metadata": memory_metadata,
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"Воспоминание добавлено: {memory_id}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка добавления воспоминания: {e}")
            raise
    
    async def search_memories(
        self, 
        query: str, 
        user_id: str = "default_user",
        limit: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Поиск воспоминаний"""
        try:
            if not query.strip():
                return {"results": [], "query": query, "total": 0}
            
            # Генерируем эмбеддинг для запроса
            query_embedding = await self.embedder.generate_embedding(query)
            
            # Получаем порог схожести из конфигурации
            threshold = similarity_threshold or self.config["memory_settings"]["similarity_threshold"]
            
            # Ищем в векторном хранилище
            memories = await self.vector_store.search_memories(
                query_embedding=query_embedding,
                user_id=user_id,
                limit=limit,
                similarity_threshold=threshold
            )
            
            # Форматируем результаты
            results = []
            for memory in memories:
                result = {
                    "id": memory["id"],
                    "memory": memory["content"],
                    "metadata": memory["metadata"],
                    "similarity": memory["similarity"],
                    "created_at": memory["metadata"].get("created_at")
                }
                results.append(result)
            
            return {
                "results": results,
                "query": query,
                "total": len(results),
                "similarity_threshold": threshold
            }
            
        except Exception as e:
            logger.error(f"Ошибка поиска воспоминаний: {e}")
            return {"results": [], "query": query, "total": 0, "error": str(e)}
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Получение конкретного воспоминания"""
        try:
            memory = await self.vector_store.get_memory(memory_id)
            if memory:
                return {
                    "id": memory["id"],
                    "content": memory["content"],
                    "metadata": memory["metadata"],
                    "created_at": memory["metadata"].get("created_at")
                }
            return None
        except Exception as e:
            logger.error(f"Ошибка получения воспоминания {memory_id}: {e}")
            return None
    
    async def update_memory(
        self, 
        memory_id: str, 
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Обновление воспоминания"""
        try:
            # Если обновляется контент, нужно перегенерировать эмбеддинг
            if content is not None:
                # Анализируем новый контент
                analysis = await self.llm_provider.analyze_memory_content(content)
                
                # Генерируем новый эмбеддинг
                embedding = await self.embedder.generate_embedding(content)
                
                # Обновляем метаданные
                if metadata is None:
                    metadata = {}
                
                metadata.update({
                    "type": analysis.get("type", "conversation"),
                    "importance": analysis.get("importance", "medium"),
                    "sentiment": analysis.get("sentiment", "neutral"),
                    "keywords": analysis.get("keywords", []),
                    "entities": analysis.get("entities", []),
                    "summary": analysis.get("summary", content[:100])
                })
                
                # Удаляем старое воспоминание и создаем новое
                await self.vector_store.delete_memory(memory_id)
                await self.vector_store.add_memory(
                    content=content,
                    embedding=embedding,
                    user_id=metadata.get("user_id", "default_user"),
                    metadata=metadata
                )
            else:
                # Обновляем только метаданные
                await self.vector_store.update_memory(memory_id, metadata=metadata)
            
            logger.info(f"Воспоминание обновлено: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления воспоминания {memory_id}: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Удаление воспоминания"""
        try:
            success = await self.vector_store.delete_memory(memory_id)
            if success:
                logger.info(f"Воспоминание удалено: {memory_id}")
            return success
        except Exception as e:
            logger.error(f"Ошибка удаления воспоминания {memory_id}: {e}")
            return False
    
    async def get_user_memories(
        self, 
        user_id: str = "default_user",
        limit: int = 100,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Получение всех воспоминаний пользователя"""
        try:
            memories = await self.vector_store.get_user_memories(
                user_id=user_id,
                limit=limit,
                memory_type=memory_type
            )
            
            # Форматируем результаты
            results = []
            for memory in memories:
                result = {
                    "id": memory["id"],
                    "content": memory["content"],
                    "metadata": memory["metadata"],
                    "created_at": memory["metadata"].get("created_at")
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка получения воспоминаний пользователя {user_id}: {e}")
            return []
    
    async def cleanup_old_memories(self, days: Optional[int] = None) -> int:
        """Очистка старых воспоминаний"""
        try:
            retention_days = days or self.config["memory_settings"]["retention_days"]
            deleted_count = await self.vector_store.cleanup_old_memories(retention_days)
            logger.info(f"Очищено {deleted_count} старых воспоминаний")
            return deleted_count
        except Exception as e:
            logger.error(f"Ошибка очистки старых воспоминаний: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики системы"""
        try:
            vector_stats = await self.vector_store.get_stats()
            embedder_stats = self.embedder.get_cache_stats()
            
            return {
                "vector_store": vector_stats,
                "embeddings": embedder_stats,
                "llm_provider": "ollama",
                "embedder_provider": "sentence_transformers",
                "vector_store_provider": "chromadb"
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, bool]:
        """Проверка здоровья всех компонентов"""
        try:
            results = {}
            
            # Проверяем LLM провайдер
            if self.llm_provider:
                results["llm"] = await self.llm_provider.health_check()
            else:
                results["llm"] = False
            
            # Проверяем провайдер эмбеддингов
            if self.embedder:
                results["embeddings"] = await self.embedder.health_check()
            else:
                results["embeddings"] = False
            
            # Проверяем векторное хранилище
            if self.vector_store:
                results["vector_store"] = await self.vector_store.health_check()
            else:
                results["vector_store"] = False
            
            # Общий статус
            results["overall"] = all(results.values())
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка проверки здоровья: {e}")
            return {"overall": False, "error": str(e)}
    
    async def close(self):
        """Закрытие всех соединений"""
        try:
            if self.llm_provider:
                await self.llm_provider.close()
            logger.info("Local Mem0 закрыт")
        except Exception as e:
            logger.error(f"Ошибка закрытия Local Mem0: {e}")
