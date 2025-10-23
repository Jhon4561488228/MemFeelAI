"""
ChromaDB Provider для AIRI Memory System
Обеспечивает работу с векторной базой данных ChromaDB
"""

import uuid
from typing import Dict, List, Optional, Any, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from loguru import logger
import yaml
from pathlib import Path
import json
from datetime import datetime

class ChromaDBProvider:
    """Провайдер для работы с ChromaDB"""
    
    def __init__(self, config_path: str = "config/chromadb_config.yaml"):
        """Инициализация провайдера ChromaDB"""
        self.config = self._load_config(config_path)
        self.client = None
        self.collection = None
        # Используем SentenceTransformerEmbeddingFunction для правильных distances
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer, falling back to DefaultEmbeddingFunction: {e}")
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self._initialize_client()
        logger.info(f"ChromaDB Provider инициализирован: {self.config['path']}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        # Если передан словарь, возвращаем его
        if isinstance(config_path, dict):
            return config_path
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Конфигурационный файл {config_path} не найден, используются настройки по умолчанию")
            return {
                "path": "./data/chroma_db",
                "collection_name": "airi_memories",
                "distance_metric": "cosine"
            }
    
    def _initialize_client(self):
        """Инициализация клиента ChromaDB"""
        try:
            # Создаем директорию если не существует
            Path(self.config["path"]).mkdir(parents=True, exist_ok=True)
            
            # Инициализируем клиент
            self.client = chromadb.PersistentClient(
                path=self.config["path"],
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Получаем или создаем коллекцию
            collection_name = self.config["collection_name"]
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Коллекция '{collection_name}' найдена")
            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "AIRI Memory System"},
                    embedding_function=self.embedding_function
                )
                logger.info(f"Коллекция '{collection_name}' создана")
                
        except Exception as e:
            logger.error(f"Ошибка инициализации ChromaDB: {e}")
            raise
    
    async def add_memory(
        self, 
        content: str, 
        embedding: List[float], 
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Добавление воспоминания в ChromaDB"""
        try:
            memory_id = str(uuid.uuid4())
            
            # Подготавливаем метаданные
            now_iso = datetime.now().isoformat()
            memory_metadata = {
                "user_id": user_id,
                "content": content,
                "created_at": now_iso,
                "created_at_ts": __import__("time").time(),
                "type": metadata.get("type", "conversation") if metadata else "conversation",
                "importance": metadata.get("importance", "medium") if metadata else "medium",
                "sentiment": metadata.get("sentiment", "neutral") if metadata else "neutral"
            }
            
            # Добавляем дополнительные метаданные (сериализуем dict в JSON)
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        # Сериализуем dict в JSON строку для ChromaDB
                        memory_metadata[key] = json.dumps(value, ensure_ascii=False)
                    elif isinstance(value, (list, tuple)):
                        # Сериализуем списки в JSON строку
                        memory_metadata[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        # Примитивные типы оставляем как есть
                        memory_metadata[key] = value
            
            # Добавляем в коллекцию
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[memory_metadata]
            )
            
            logger.info(f"Воспоминание добавлено: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Ошибка добавления воспоминания: {e}")
            raise
    
    async def search_memories(
        self, 
        query_embedding: List[float], 
        user_id: str,
        limit: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Поиск воспоминаний по эмбеддингу с оптимизацией"""
        try:
            # Оптимизированный поиск с предварительной фильтрацией
            search_limit = max(limit * 10, 100)  # Увеличиваем лимит для лучшей фильтрации
            
            # Поиск в коллекции
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_limit,
                where={"user_id": user_id} if user_id else None
            )
            
            memories = []
            if results["ids"] and results["ids"][0]:
                # Сортируем по схожести и фильтруем
                scored_results = []
                for i, memory_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1 - distance  # Преобразуем расстояние в схожесть
                    
                    if similarity >= similarity_threshold:
                        scored_results.append((similarity, i, memory_id))
                
                # Сортируем по убыванию схожести
                scored_results.sort(key=lambda x: x[0], reverse=True)
                
                # Берем только нужное количество
                for similarity, i, memory_id in scored_results[:limit]:
                    
                    # Фильтруем по порогу схожести
                    if similarity >= similarity_threshold:
                        # Десериализуем JSON поля в metadata
                        metadata = results["metadatas"][0][i].copy()
                        for key, value in metadata.items():
                            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                                try:
                                    metadata[key] = json.loads(value)
                                except json.JSONDecodeError:
                                    pass  # Оставляем как строку если не JSON
                        
                        memory = {
                            "id": memory_id,
                            "content": results["documents"][0][i],
                            "metadata": metadata,
                            "similarity": similarity,
                            "distance": distance
                        }
                        memories.append(memory)
            
            # Сортируем по схожести и ограничиваем количество
            memories.sort(key=lambda x: x["similarity"], reverse=True)
            return memories[:limit]
            
        except Exception as e:
            logger.error(f"Ошибка поиска воспоминаний: {e}")
            return []
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Получение конкретного воспоминания"""
        try:
            results = self.collection.get(
                ids=[memory_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "content": results["documents"][0],
                    "metadata": results["metadatas"][0],
                    "embedding": results["embeddings"][0]
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
            # Получаем текущее воспоминание
            current = await self.get_memory(memory_id)
            if not current:
                return False
            
            # Подготавливаем данные для обновления
            update_data = {}
            
            if content is not None:
                update_data["documents"] = [content]
            
            if metadata is not None:
                # Обновляем метаданные
                current_metadata = current["metadata"].copy()
                current_metadata.update(metadata)
                current_metadata["updated_at"] = datetime.now().isoformat()
                update_data["metadatas"] = [current_metadata]
            
            if update_data:
                update_data["ids"] = [memory_id]
                self.collection.update(**update_data)
                logger.info(f"Воспоминание обновлено: {memory_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка обновления воспоминания {memory_id}: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Удаление воспоминания"""
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Воспоминание удалено: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления воспоминания {memory_id}: {e}")
            return False
    
    async def get_user_memories(
        self, 
        user_id: str, 
        limit: int = 100,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Получение всех воспоминаний пользователя"""
        try:
            where_clause = {"user_id": user_id}
            if memory_type:
                where_clause["type"] = memory_type
            
            results = self.collection.get(
                where=where_clause,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            memories = []
            if results["ids"]:
                for i, memory_id in enumerate(results["ids"]):
                    memory = {
                        "id": memory_id,
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i]
                    }
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Ошибка получения воспоминаний пользователя {user_id}: {e}")
            return []
    
    async def cleanup_old_memories(self, days: int = 365) -> int:
        """Очистка старых воспоминаний"""
        try:
            from datetime import datetime, timedelta
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Получаем старые воспоминания
            cutoff_ts = __import__("time").time() - days * 24 * 3600
            old_memories = self.collection.get(
                where={"created_at_ts": {"$lt": cutoff_ts}}
            )
            
            if old_memories["ids"]:
                self.collection.delete(ids=old_memories["ids"])
                deleted_count = len(old_memories["ids"])
                logger.info(f"Удалено {deleted_count} старых воспоминаний")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Ошибка очистки старых воспоминаний: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики коллекции"""
        try:
            count = self.collection.count()
            return {
                "total_memories": count,
                "collection_name": self.config["collection_name"],
                "path": self.config["path"]
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {"total_memories": 0}
    
    async def health_check(self) -> bool:
        """Проверка доступности ChromaDB"""
        try:
            self.collection.count()
            return True
        except Exception as e:
            logger.error(f"ChromaDB недоступен: {e}")
            return False
