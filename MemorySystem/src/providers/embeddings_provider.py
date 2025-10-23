"""
Embeddings Provider для AIRI Memory System
Обеспечивает генерацию эмбеддингов с помощью sentence-transformers
"""

import asyncio
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger
import yaml
from pathlib import Path
import torch
import pickle
import os
from datetime import datetime, timedelta

class EmbeddingsProvider:
    """Провайдер для генерации эмбеддингов"""
    
    def __init__(self, config_path: str = "config/embeddings_config.yaml"):
        """Инициализация провайдера эмбеддингов"""
        self.config = self._load_config(config_path)
        self.model = None
        self.cache = {}
        self.cache_file = None
        self._initialize_model()
        self._load_cache()
        logger.info(f"Embeddings Provider инициализирован: {self.config['model']}")
    
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
                "model": "distiluse-base-multilingual-cased",
                "cache_dir": "./data/embeddings_cache",
                "device": "cpu",
                "batch_size": 32,
                "max_length": 512
            }
    
    def _initialize_model(self):
        """Инициализация модели sentence-transformers"""
        try:
            # Создаем директорию кэша если не существует
            cache_dir = self.config.get("cache_dir", "./data/embeddings_cache")
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            
            # Инициализируем модель
            model_name = self.config["model"]
            device = self.config.get("device", "cpu")
            
            self.model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                device=device
            )
            
            # Настройки модели
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.config.get("max_length", 512)
            
            logger.info(f"Модель {model_name} загружена на устройство {device}")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации модели: {e}")
            raise
    
    def _load_cache(self):
        """Загрузка кэша эмбеддингов"""
        try:
            cache_dir = self.config.get("cache_dir", "./data/embeddings_cache")
            self.cache_file = Path(cache_dir) / "embeddings_cache.pkl"
            
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                # Проверяем TTL кэша
                ttl_hours = self.config.get("cache_settings", {}).get("ttl_hours", 24)
                cutoff_time = datetime.now() - timedelta(hours=ttl_hours)
                
                # Фильтруем устаревшие записи
                self.cache = {
                    key: value for key, value in cache_data.items()
                    if value.get("timestamp", datetime.min) > cutoff_time
                }
                
                logger.info(f"Кэш загружен: {len(self.cache)} записей")
            else:
                self.cache = {}
                
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Сохранение кэша эмбеддингов"""
        try:
            if self.cache_file:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
                logger.debug("Кэш сохранен")
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Генерация ключа кэша для текста"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Генерация эмбеддинга для одного текста"""
        try:
            # Проверяем кэш
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                logger.debug("Эмбеддинг найден в кэше")
                return self.cache[cache_key]["embedding"]
            
            # Генерируем эмбеддинг
            embedding = await self._generate_embedding_async(text)
            
            # Сохраняем в кэш
            max_cache_size = self.config.get("cache_settings", {}).get("max_size", 10000)
            if len(self.cache) < max_cache_size:
                self.cache[cache_key] = {
                    "embedding": embedding,
                    "timestamp": datetime.now()
                }
                self._save_cache()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Ошибка генерации эмбеддинга: {e}")
            raise
    
    async def _generate_embedding_async(self, text: str) -> List[float]:
        """Асинхронная генерация эмбеддинга"""
        # Запускаем в отдельном потоке, так как sentence-transformers синхронный
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            self._generate_embedding_sync, 
            text
        )
        return embedding.tolist()
    
    def _generate_embedding_sync(self, text: str) -> np.ndarray:
        """Синхронная генерация эмбеддинга"""
        try:
            # Нормализуем и очищаем текст
            if not text:
                # Возвращаем нулевой вектор для пустого текста
                return np.zeros(self.model.get_sentence_embedding_dimension())
            
            # Убеждаемся, что текст правильно декодирован как UTF-8
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            
            # Нормализуем текст
            text = text.strip()
            
            # Логируем для отладки
            logger.debug(f"Generating embedding for text: '{text[:100]}...' (length: {len(text)})")
            
            # Генерируем эмбеддинг
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.config.get("model_settings", {}).get("normalize_embeddings", True),
                convert_to_numpy=True,
                show_progress_bar=self.config.get("model_settings", {}).get("show_progress_bar", False)
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Ошибка синхронной генерации эмбеддинга: {e}")
            # Возвращаем нулевой вектор в случае ошибки
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Генерация эмбеддингов для списка текстов"""
        try:
            if not texts:
                return []
            
            # Фильтруем пустые тексты
            valid_texts = [text.strip() for text in texts if text.strip()]
            if not valid_texts:
                return []
            
            # Проверяем кэш для каждого текста
            cached_embeddings = {}
            uncached_texts = []
            
            for text in valid_texts:
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    cached_embeddings[text] = self.cache[cache_key]["embedding"]
                else:
                    uncached_texts.append(text)
            
            # Генерируем эмбеддинги для некешированных текстов
            new_embeddings = []
            if uncached_texts:
                batch_size = self.config.get("batch_size", 32)
                for i in range(0, len(uncached_texts), batch_size):
                    batch = uncached_texts[i:i + batch_size]
                    batch_embeddings = await self._generate_embeddings_batch_async(batch)
                    new_embeddings.extend(batch_embeddings)
                    
                    # Сохраняем в кэш
                    max_cache_size = self.config.get("cache_settings", {}).get("max_size", 10000)
                    for text, embedding in zip(batch, batch_embeddings):
                        if len(self.cache) < max_cache_size:
                            cache_key = self._get_cache_key(text)
                            self.cache[cache_key] = {
                                "embedding": embedding,
                                "timestamp": datetime.now()
                            }
            
            # Объединяем кешированные и новые эмбеддинги
            all_embeddings = []
            for text in valid_texts:
                if text in cached_embeddings:
                    all_embeddings.append(cached_embeddings[text])
                else:
                    # Находим соответствующий новый эмбеддинг
                    text_index = uncached_texts.index(text)
                    all_embeddings.append(new_embeddings[text_index])
            
            # Сохраняем кэш
            if new_embeddings:
                self._save_cache()
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Ошибка генерации эмбеддингов для батча: {e}")
            raise
    
    async def _generate_embeddings_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Асинхронная генерация эмбеддингов для батча"""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self._generate_embeddings_batch_sync,
            texts
        )
        return [emb.tolist() for emb in embeddings]
    
    def _generate_embeddings_batch_sync(self, texts: List[str]) -> List[np.ndarray]:
        """Синхронная генерация эмбеддингов для батча"""
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=self.config.get("model_settings", {}).get("normalize_embeddings", True),
                convert_to_numpy=True,
                show_progress_bar=self.config.get("model_settings", {}).get("show_progress_bar", False),
                batch_size=self.config.get("batch_size", 32)
            )
            
            return embeddings if isinstance(embeddings, list) else [embeddings]
            
        except Exception as e:
            logger.error(f"Ошибка синхронной генерации эмбеддингов для батча: {e}")
            # Возвращаем нулевые векторы в случае ошибки
            dim = self.model.get_sentence_embedding_dimension()
            return [np.zeros(dim) for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        """Получение размерности эмбеддингов"""
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Ошибка получения размерности эмбеддингов: {e}")
            return 512  # Размерность для distiluse-base-multilingual-cased
    
    async def health_check(self) -> bool:
        """Проверка доступности провайдера эмбеддингов"""
        try:
            # Пробуем сгенерировать тестовый эмбеддинг
            test_embedding = await self.generate_embedding("test")
            return len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Провайдер эмбеддингов недоступен: {e}")
            return False
    
    async def clear_cache(self):
        """Очистка кэша эмбеддингов"""
        try:
            self.cache.clear()
            if self.cache_file and self.cache_file.exists():
                self.cache_file.unlink()
            logger.info("Кэш эмбеддингов очищен")
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        return {
            "cache_size": len(self.cache),
            "cache_file": str(self.cache_file) if self.cache_file else None,
            "model_name": self.config["model"],
            "embedding_dimension": self.get_embedding_dimension()
        }
