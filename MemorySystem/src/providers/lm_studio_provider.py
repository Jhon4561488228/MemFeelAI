"""
Ollama Provider для AIRI Memory System
Обеспечивает взаимодействие с локальным LLM через Ollama
"""

import asyncio
import json
import hashlib
import time
import re
from typing import Dict, List, Optional, Any, Tuple
import os
import httpx
from loguru import logger
import yaml
from pathlib import Path
from asyncio import Semaphore
try:
    from json_repair import repair_json  # type: ignore
    JSON_REPAIR_AVAILABLE = True
    logger.info("json_repair library imported successfully")
except ImportError as e:
    JSON_REPAIR_AVAILABLE = False
    repair_json = None  # type: ignore
    logger.warning(f"json_repair library not available: {e}")

class LMStudioProvider:
    """Провайдер для работы с Ollama"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config_path: str = "config/lm_studio_config.yaml"):
        """Singleton pattern для предотвращения множественной инициализации"""
        if cls._instance is None:
            cls._instance = super(LMStudioProvider, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = "config/lm_studio_config.yaml"):
        """Инициализация провайдера Ollama"""
        if self._initialized:
            return  # Уже инициализирован
        self.config = self._load_config(config_path)
        # ENV overrides
        if os.getenv("OLLAMA_BASE_URL"):
            self.config["base_url"] = os.getenv("OLLAMA_BASE_URL")
        if os.getenv("OLLAMA_TIMEOUT"):
            try:
                self.config["timeout"] = int(os.getenv("OLLAMA_TIMEOUT"))
            except Exception:
                pass
        if os.getenv("OLLAMA_RETRIES"):
            try:
                self.config["retry_attempts"] = int(os.getenv("OLLAMA_RETRIES"))
            except Exception:
                pass
        self.config.setdefault("connection", {})
        if os.getenv("OLLAMA_RETRY_DELAY"):
            try:
                self.config["connection"]["retry_delay"] = float(os.getenv("OLLAMA_RETRY_DELAY"))
            except Exception:
                pass
        if os.getenv("OLLAMA_RETRY_MAX_DELAY"):
            try:
                self.config["connection"]["max_retry_delay"] = float(os.getenv("OLLAMA_RETRY_MAX_DELAY"))
            except Exception:
                pass

        self.client = httpx.AsyncClient(
            timeout=self.config.get("timeout", 120),  # Используем таймаут из конфигурации (120 секунд)
            base_url=self.config["base_url"]
        )
        # Ограничение параллельности запросов к Ollama
        max_conc = int(os.getenv("OLLAMA_MAX_CONCURRENCY", "4"))
        self._sem = Semaphore(max(1, max_conc))
        # Кэш для ускорения повторных запросов с TTL
        self._sqlite_cache = None  # Ленивая инициализация SQLiteCache
        self._memory_cache: Dict[str, Tuple[str, float]] = {}  # Fallback кэш
        self._cache_ttl = 3600  # TTL в секундах (1 час)
        self._use_sqlite = True  # Флаг для переключения на fallback
        logger.info(f"Ollama Provider инициализирован: {self.config['base_url']}")
        self._initialized = True
    
    async def _get_cache(self):
        """Get cache instance with fallback to memory"""
        if self._use_sqlite:
            try:
                if self._sqlite_cache is None:
                    from src.cache.sqlite_cache import SQLiteCache
                    # Используем тот же путь к базе данных, что и основной сервис
                    import os
                    data_dir = os.getenv("AIRI_DATA_DIR", "./data")
                    db_path = os.path.join(data_dir, "memory_system.db")
                    self._sqlite_cache = SQLiteCache(db_path)
                    # Инициализируем базу данных
                    await self._sqlite_cache._ensure_initialized()
                return self._sqlite_cache
            except (Exception, asyncio.CancelledError, asyncio.TimeoutError) as e:
                logger.warning(f"SQLiteCache unavailable, falling back to memory: {e}")
                self._use_sqlite = False
                self._sqlite_cache = None
                return None
        return None
    
    def _get_cache_key(self, prompt: str, task_type: str) -> str:
        """Генерация ключа кэша с префиксом типа"""
        content = f"{prompt}:{task_type}"
        hash_key = hashlib.md5(content.encode()).hexdigest()
        return f"llm:{task_type}:{hash_key}"  # Префикс для типизации
    
    async def clear_cache(self, task_type: str = None) -> int:
        """Очистка кэша с fallback"""
        count = 0
        cache = await self._get_cache()
        
        if cache:  # SQLiteCache
            try:
                if task_type:
                    prefix = f"llm:{task_type}:"
                    count = await asyncio.wait_for(cache.delete_prefix(prefix), timeout=3.0)
                else:
                    count = await asyncio.wait_for(cache.delete_prefix("llm:"), timeout=3.0)
                logger.info(f"SQLiteCache cleared: {count} entries")
                return count
            except (Exception, asyncio.CancelledError, asyncio.TimeoutError) as e:
                logger.warning(f"SQLiteCache clear error: {e}")
                self._use_sqlite = False
        
        # Fallback: clear memory cache
        if task_type:
            prefix = f"llm:{task_type}:"
            keys_to_remove = [k for k in self._memory_cache.keys() 
                             if k.startswith(prefix)]
            for key in keys_to_remove:
                del self._memory_cache[key]
            count = len(keys_to_remove)
        else:
            count = len(self._memory_cache)
            self._memory_cache.clear()
        
        logger.info(f"Memory cache cleared: {count} entries")
        return count
    
    async def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Получение из кэша с fallback"""
        cache = await self._get_cache()
        
        if cache:  # SQLiteCache
            try:
                # Добавляем timeout для чтения
                result = await asyncio.wait_for(cache.get(cache_key), timeout=2.0)
                if result:
                    logger.debug(f"SQLiteCache hit: {cache_key[:50]}...")
                    return result.get("response")
            except (Exception, asyncio.CancelledError, asyncio.TimeoutError) as e:
                logger.debug(f"SQLiteCache error: {e}")
                self._use_sqlite = False
        
        # Fallback to memory cache
        if cache_key in self._memory_cache:
            value, timestamp = self._memory_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.debug(f"Memory cache hit: {cache_key[:50]}...")
                return value
            else:
                del self._memory_cache[cache_key]
        
        return None
    
    async def _put_to_cache(self, cache_key: str, value: str) -> None:
        """Сохранение в кэш с fallback"""
        cache = await self._get_cache()
        
        if cache:  # SQLiteCache
            try:
                # Добавляем timeout для записи
                await asyncio.wait_for(
                    cache.set(cache_key, {"response": value}, ttl_sec=self._cache_ttl),
                    timeout=3.0
                )
                logger.debug(f"Saved to SQLiteCache: {cache_key[:50]}...")
                return
            except (Exception, asyncio.CancelledError, asyncio.TimeoutError) as e:
                logger.debug(f"SQLiteCache write error: {e}")
                self._use_sqlite = False
        
        # Fallback to memory cache
        self._memory_cache[cache_key] = (value, time.time())
        logger.debug(f"Saved to memory cache: {cache_key[:50]}...")
        
        # Ограничение размера memory cache
        if len(self._memory_cache) > 100:
            oldest_key = min(self._memory_cache.keys(), 
                            key=lambda k: self._memory_cache[k][1])
            del self._memory_cache[oldest_key]
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        # Если передан словарь, возвращаем его
        if isinstance(config_path, dict):
            return config_path
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # Гарантируем, что возвращаем словарь
                if isinstance(config, dict):
                    return config
                else:
                    logger.warning(f"Конфигурация {config_path} не является словарем, используются настройки по умолчанию")
                    return self._get_default_config()
        except FileNotFoundError:
            # Пробуем альтернативный путь относительно проекта
            try:
                alt = (Path(__file__).resolve().parents[2] / 'config' / 'lm_studio_config.yaml')
                with open(alt, 'r', encoding='utf-8') as f:
                    logger.warning(f"Конфиг {config_path} не найден, используем {alt}")
                    config = yaml.safe_load(f)
                    if isinstance(config, dict):
                        return config
                    else:
                        logger.warning(f"Альтернативная конфигурация {alt} не является словарем, используются настройки по умолчанию")
                        return self._get_default_config()
            except Exception:
                logger.warning(f"Конфигурационный файл {config_path} не найден, используются настройки по умолчанию")
                return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию по умолчанию"""
        return {
            "base_url": "http://localhost:11434/v1",
            "model": "llava:7b",
            "max_tokens": 2048,
            "temperature": 0.7,
            "timeout": 120,
            "retry_attempts": 3
        }
    
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        task_type: str = "text_generation",
        emotional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Генерация текста через Ollama с поддержкой эмоционального контекста и retry механизмом"""
        # Проверяем кэш
        cache_key = self._get_cache_key(prompt, task_type)
        cached_result = await self._get_from_cache(cache_key)
        if cached_result:
            logger.debug(f"Результат найден в кэше для {task_type}")
            return cached_result
        
        max_retries = self.config.get("retry_attempts", 3)
        retry_delay = self.config.get("connection", {}).get("retry_delay", 1.0)
        max_retry_delay = self.config.get("connection", {}).get("max_retry_delay", 10.0)
        
        for attempt in range(max_retries):
            try:
                # Получаем настройки для конкретной задачи
                task_config = self.config.get("tasks", {}).get(task_type, {})
                
                # Добавляем эмоциональный контекст к промпту
                enhanced_prompt = self._enhance_prompt_with_emotions(prompt, emotional_context)
                
                # Ограничиваем размер промпта для предотвращения перегрузки
                if len(enhanced_prompt) > 4000:
                    enhanced_prompt = enhanced_prompt[:4000] + "..."
                    logger.warning("Промпт обрезан до 4000 символов для предотвращения перегрузки")
                
                # Используем модель из конфигурации задачи, если указана
                model = task_config.get("model", self.config["model"])
                
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": enhanced_prompt}],
                    "max_tokens": max_tokens or task_config.get("max_tokens", self.config.get("max_tokens", 512)),
                    "temperature": temperature or task_config.get("temperature", self.config.get("temperature", 0.7)),
                    "stream": False
                }
                
                logger.debug(f"Отправка запроса в Ollama /v1 с моделью {model} для задачи {task_type} (попытка {attempt + 1}/{max_retries})")
                try:
                    async with self._sem:
                        response = await self.client.post("/chat/completions", json=payload)

                    if response.status_code == 200:
                        result = response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            message = result["choices"][0]["message"]
                            content = message.get("content", "")
                            
                            logger.debug(f"Успешный ответ от Ollama /v1 ({model}): {content[:100]}...")
                            await self._put_to_cache(cache_key, content)
                            return content
                        else:
                            raise ValueError("Некорректный формат ответа от Ollama")
                    else:
                        raise httpx.HTTPStatusError(f"HTTP {response.status_code}: {response.text}", request=response.request, response=response)
                except Exception as e_openai:
                    # Fallback на /api/generate если OpenAI совместимый маршрут недоступен
                    try:
                        base_url: str = self.config.get("base_url", "http://localhost:11434")
                        root_url = base_url.replace("/v1", "")
                        gen_payload = {
                            "model": model,
                            "prompt": enhanced_prompt,
                            "stream": False,
                            "options": {
                                "temperature": temperature or task_config.get("temperature", self.config.get("temperature", 0.7)),
                                "num_predict": max_tokens or task_config.get("max_tokens", self.config.get("max_tokens", 512)),
                            },
                        }
                        logger.debug(f"Fallback: /api/generate {root_url} модель {model}")
                        async with self._sem:
                            async with httpx.AsyncClient(base_url=root_url, timeout=self.client.timeout) as c:
                                resp2 = await c.post("/api/generate", json=gen_payload)
                        if resp2.status_code == 200:
                            data = resp2.json()
                            content = data.get("response", "")
                            await self._put_to_cache(cache_key, content)
                            return content
                        else:
                            raise httpx.HTTPStatusError(f"HTTP {resp2.status_code}: {resp2.text}", request=resp2.request, response=resp2)
                    except Exception as e_generate:
                        logger.warning(f"Ollama /v1 и /api fallback не удались: {e_openai} | {e_generate}")
                        raise
                    
            except httpx.TimeoutException as e:
                logger.warning(f"Таймаут при обращении к Ollama (попытка {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(retry_delay * (2 ** attempt), max_retry_delay))
                    continue
                else:
                    raise
            except httpx.ConnectError as e:
                logger.warning(f"Ошибка подключения к Ollama (попытка {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(retry_delay * (2 ** attempt), max_retry_delay))
                    continue
                else:
                    raise
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP ошибка от Ollama (попытка {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(retry_delay * (2 ** attempt), max_retry_delay))
                    continue
                else:
                    raise
            except Exception as e:
                logger.error(f"Неожиданная ошибка генерации текста (попытка {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(retry_delay * (2 ** attempt), max_retry_delay))
                    continue
                else:
                    raise

    async def list_models(self) -> List[str]:
        """Список доступных моделей Ollama/LM Studio (пробуем несколько маршрутов)."""
        models: List[str] = []
        # Попытка 1: OpenAI-совместимый /v1/models
        try:
            resp = await self.client.get("/models")
            if resp.status_code == 200:
                data = resp.json()
                # форматы могут отличаться: {"data": [{"id": "model"}, ...]} или {"models": ["model", ...]}
                if isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list):
                        for item in data["data"]:
                            mid = item.get("id") if isinstance(item, dict) else None
                            if mid:
                                models.append(mid)
                    elif "models" in data and isinstance(data["models"], list):
                        models.extend([str(m) for m in data["models"]])
                if models:
                    return sorted(list(dict.fromkeys(models)))
        except Exception:
            pass
        # Попытка 2: Нативный Ollama /api/tags на корне
        try:
            base_url: str = self.config.get("base_url", "http://localhost:11434")
            root_url = base_url.replace("/v1", "")
            async with httpx.AsyncClient(base_url=root_url, timeout=self.client.timeout) as c:
                resp = await c.get("/api/tags")
            if resp.status_code == 200:
                data = resp.json() or {}
                tags = data.get("models") or data.get("tags") or []
                # Форматы: [{"name": "llava:7b"}, ...] или [{"model": "phi3:mini"}, ...]
                for item in tags:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("model") or item.get("id")
                        if name:
                            models.append(str(name))
                    elif isinstance(item, str):
                        models.append(item)
        except Exception:
            pass
        return sorted(list(dict.fromkeys(models)))
    
    def _enhance_prompt_with_emotions(self, prompt: str, emotional_context: Optional[Dict[str, Any]]) -> str:
        """Улучшение промпта с эмоциональным контекстом"""
        if not emotional_context:
            return prompt
        
        # Обрабатываем случай, когда emotional_context является строкой
        if isinstance(emotional_context, str):
            emotion = emotional_context
            confidence = 0.5
            consistency = 'medium'
        elif isinstance(emotional_context, dict):
            emotion = emotional_context.get('emotion', 'нейтральная')
            confidence = emotional_context.get('confidence', 0.5)
            consistency = emotional_context.get('consistency', 'medium')
        else:
            # Если тип неизвестен, используем значения по умолчанию
            emotion = 'нейтральная'
            confidence = 0.5
            consistency = 'medium'
        
        # Создаем эмоциональную инструкцию
        emotional_instruction = self._create_emotional_instruction(emotion, confidence, consistency)
        
        # Объединяем с оригинальным промптом
        enhanced_prompt = f"{emotional_instruction}\n\n{prompt}"
        
        return enhanced_prompt
    
    def _create_emotional_instruction(self, emotion: str, confidence: float, consistency: str) -> str:
        """Создание эмоциональной инструкции для LLM"""
        base_instruction = f"Пользователь испытывает эмоцию: {emotion} (уверенность: {confidence:.2f})"
        
        if emotion == 'радость':
            tone_instruction = "Отвечай позитивно и энергично, поддерживай хорошее настроение."
        elif emotion == 'грусть':
            tone_instruction = "Прояви сочувствие и поддержку, будь деликатным и понимающим."
        elif emotion == 'злость':
            tone_instruction = "Отвечай спокойно и понимающе, не усугубляй ситуацию."
        elif emotion == 'страх':
            tone_instruction = "Успокой и поддержи, будь уверенным и обнадеживающим."
        elif emotion == 'удивление':
            tone_instruction = "Отвечай с энтузиазмом и интересом, поддерживай удивление."
        else:
            tone_instruction = "Отвечай нейтрально и дружелюбно."
        
        consistency_note = f"Консистентность эмоций: {consistency}"
        
        return f"{base_instruction}\n{tone_instruction}\n{consistency_note}"
    
    async def generate_multimodal(
        self,
        prompt: str,
        images_b64: List[str],
        task_type: str = "image_analysis",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Генерация текста по изображению(ям) с использованием мультимодели (например, llava:13b)."""
        task_config = self.config.get("tasks", {}).get(task_type, {})
        model = task_config.get("model", self.config["model"])
        # /api/generate доступен на корне (без /v1)
        base_url: str = self.config.get("base_url", "http://localhost:11434")
        root_url = base_url.replace("/v1", "")
        # Таймаут для vision (ENV VISION_TIMEOUT_SEC, по умолчанию 240 сек)
        timeout_sec = int(os.getenv("VISION_TIMEOUT_SEC", str(self.config.get("timeout", 60))))  # ОПТИМИЗАЦИЯ: Уменьшено с 240 до 60 секунд
        # Безопасность: ограничим количество и размер картинок
        safe_images: List[str] = []
        max_image_len = 5 * 1024 * 1024 * 4  # ~5MB binary => ~6.6MB base64; берем запас
        for img in images_b64[:3]:  # максимум 3 изображения за раз
            if len(img) > max_image_len:
                logger.warning("Изображение слишком большое, обрезаем base64 до ~5MB")
                safe_images.append(img[:max_image_len])
            else:
                safe_images.append(img)
        # Настройки по умолчанию (с ENV override)
        # Кол-во слоёв на GPU: -1 = авто (максимум на GPU). 1 может увести большую часть на CPU → не ставим по умолчанию
        num_gpu = int(os.getenv("VISION_NUM_GPU", "-1"))  # -1 = авто (максимум на GPU)
        num_ctx = int(os.getenv("VISION_NUM_CTX", "768"))  # Стандартный контекст
        num_predict_opt = int(os.getenv("VISION_NUM_PREDICT", str(max_tokens or task_config.get("max_tokens", 256))))
        num_thread = int(os.getenv("VISION_NUM_THREAD", str(os.cpu_count() or 8)))  # Автоматическое определение
        num_batch = int(os.getenv("VISION_NUM_BATCH", "24"))
        keep_alive = int(os.getenv("VISION_KEEP_ALIVE", "60"))

        payload = {
            "model": model,
            "prompt": prompt,
            "images": safe_images,
            "stream": False,
            "options": {
                "temperature": temperature or task_config.get("temperature", 0.2),
                "num_predict": num_predict_opt,
                "num_ctx": num_ctx,
                "num_gpu": num_gpu,
                "use_mmap": True,
                "num_thread": num_thread,
                "num_batch": num_batch,
            },
            "keep_alive": keep_alive,
        }
        logger.debug(f"Vision request → {root_url}/api/generate model={model}, timeout={timeout_sec}s")
        async with self._sem:
            async with httpx.AsyncClient(base_url=root_url, timeout=httpx.Timeout(timeout_sec)) as c:
                resp = await c.post("/api/generate", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("response", "")
        raise httpx.HTTPStatusError(f"HTTP {resp.status_code}: {resp.text}", request=resp.request, response=resp)
    
    async def analyze_memory_content(self, content: str) -> Dict[str, Any]:
        """Анализ контента для создания воспоминания с retry механизмом и обработкой больших данных"""
        max_retries = 3
        
        # Ограничиваем размер контента для предотвращения перегрузки
        original_content = content
        if len(content) > 2000:
            content = content[:2000] + "..."
            logger.warning(f"Контент обрезан с {len(original_content)} до {len(content)} символов для предотвращения перегрузки")
        
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    # Первая попытка с полным промптом
                    prompt = self.create_adaptive_prompt(content)
                else:
                    # Повторные попытки с упрощенным промптом
                    prompt = self._create_simple_prompt(content)
                
                logger.info(f"Memory analysis attempt {attempt + 1}/{max_retries}")
                result = await self.generate_text(prompt, task_type="analysis")
                # Безопасное логирование с правильной кодировкой
                try:
                    safe_result = result[:200].encode('utf-8', errors='replace').decode('utf-8')
                    logger.debug(f"Ollama raw response: {safe_result}...")
                except Exception:
                    logger.debug(f"Ollama raw response: [encoding error] {len(result)} chars")
                
                # Улучшенное извлечение JSON из markdown блоков
                result = self._extract_json_from_markdown(result)
                
                # Дополнительная очистка (сохраняем структуру JSON)
                # Убираем только лишние пробелы, но сохраняем переносы строк для читаемости
                result = result.replace('\r', ' ').replace('\t', ' ')
                
                logger.debug(f"Cleaned JSON: {result}")
                
                parsed_result = json.loads(result)
                
                # Валидация структуры JSON от Ollama
                if not self._validate_json_structure(parsed_result):
                    raise ValueError("Invalid JSON structure")
                
                # Преобразуем списки в строки для совместимости с ChromaDB
                if "keywords" in parsed_result and isinstance(parsed_result["keywords"], list):
                    parsed_result["keywords"] = ", ".join(parsed_result["keywords"])
                
                # Улучшенная обработка entities для поддержки сложных объектов
                if "entities" in parsed_result and isinstance(parsed_result["entities"], list):
                    if parsed_result["entities"] and isinstance(parsed_result["entities"][0], dict):
                        # Если entities - список словарей, извлекаем только entity
                        parsed_result["entities"] = ", ".join([
                            item.get("entity", str(item)) for item in parsed_result["entities"]
                        ])
                    else:
                        # Если entities - список строк, объединяем как обычно
                        parsed_result["entities"] = ", ".join(parsed_result["entities"])
                
                # Добавляем информацию о том, что контент был обрезан
                if len(original_content) > 2000:
                    parsed_result["content_truncated"] = True
                    parsed_result["original_length"] = len(original_content)
                
                logger.info(f"Memory analysis successful on attempt {attempt + 1}")
                logger.debug(f"Parsed result: {parsed_result}")
                return parsed_result
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON parsing attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying with simplified prompt...")
                    continue
                else:
                    logger.error(f"All JSON parsing attempts failed, using fallback")
                    return self._create_fallback_analysis(content)
            except Exception as e:
                logger.error(f"Unexpected error in attempt {attempt + 1}: {e}")
                
                if attempt < max_retries - 1:
                    continue
                else:
                    logger.error(f"All attempts failed, using fallback")
                    return self._create_fallback_analysis(content)
    
    def _create_fallback_analysis(self, content: str) -> Dict[str, Any]:
        """Улучшенный fallback анализ для сохранения максимальной информации"""
        keywords = self._extract_simple_keywords(content)
        return {
            "summary": content[:200] + "..." if len(content) > 200 else content,
            "keywords": keywords,
            "type": "conversation",
            "importance": "medium",
            "entities": keywords,  # Используем keywords как entities в fallback
            "sentiment": "neutral"
        }
    
    def _extract_json_from_markdown(self, text: str) -> str:
        """Извлечение JSON из markdown блоков"""
        import re
        
        # Удаляем markdown блоки
        json_pattern = r'```(?:json)?\s*(.*?)\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Ищем JSON в тексте (более агрессивный поиск)
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if matches:
            # Берем самый длинный JSON (скорее всего полный)
            return max(matches, key=len).strip()
        
        # Если нет JSON, ищем простые пары ключ-значение
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            return text[json_start:json_end]
        
        return text

    def _validate_json_structure(self, data: dict) -> bool:
        """Валидация структуры JSON от Ollama"""
        required_fields = ["summary", "keywords", "type", "importance"]
        return all(field in data for field in required_fields)
    
    def _create_simple_prompt(self, content: str) -> str:
        """Создание упрощенного промпта для retry попыток"""
        return f"""Только JSON:

{content[:500]}

{{"summary":"краткое описание","keywords":"ключевые слова","type":"conversation","importance":"medium","entities":"сущности","sentiment":"neutral"}}"""

    def detect_language(self, text: str) -> str:
        """Определение языка текста"""
        import re
        
        russian_chars = len(re.findall(r'[а-яё]', text.lower()))
        english_chars = len(re.findall(r'[a-z]', text.lower()))
        
        if russian_chars > english_chars:
            return 'ru'
        else:
            return 'en'

    def create_adaptive_prompt(self, content: str) -> str:
        """Создание адаптивного промпта на основе языка контента"""
        
        detected_lang = self.detect_language(content)
        
        if detected_lang == 'ru':
            prompt_template = f"""Только JSON, без объяснений:

{content[:800]}

{{"summary":"краткое резюме","keywords":["ключевое","слово1"],"type":"conversation","importance":"medium","entities":["сущности"],"sentiment":"neutral"}}"""
        else:
            prompt_template = f"""Only JSON, no explanations:

{content[:800]}

{{"summary":"brief summary","keywords":["key","word1"],"type":"conversation","importance":"medium","entities":["entities"],"sentiment":"neutral"}}"""
        
        return prompt_template

    def _extract_simple_keywords(self, content: str) -> str:
        """Простое извлечение ключевых слов без Ollama"""
        # Простая логика извлечения ключевых слов
        words = content.lower().split()
        # Убираем стоп-слова и возвращаем первые 5 слов
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'к', 'у', 'о', 'об', 'что', 'как', 'где', 'когда', 'почему', 'это', 'то', 'а', 'но', 'или', 'же', 'ли', 'бы', 'не', 'ни', 'уже', 'еще', 'только', 'даже', 'все', 'всего', 'всех', 'всем', 'всеми', 'всему', 'всё', 'всей', 'всю', 'всё', 'всё', 'всё'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3][:5]
        return ", ".join(keywords)
    
    async def extract_key_information(self, content: str) -> List[str]:
        """Извлечение ключевой информации из контента"""
        prompt = f"""
Извлеки ключевую информацию из следующего текста. Верни только список важных фактов, по одному на строку:

{content}
"""
        
        try:
            result = await self.generate_text(prompt, task_type="extraction")
            # Разбиваем на строки и фильтруем пустые
            facts = [line.strip() for line in result.split('\n') if line.strip()]
            return facts
        except Exception as e:
            logger.error(f"Ошибка извлечения информации: {e}")
            return [content[:200] + "..." if len(content) > 200 else content]
    
    async def summarize_content(self, content: str) -> str:
        """Суммаризация контента"""
        prompt = f"""
Создай краткое резюме следующего текста, сохранив ключевую информацию:

{content}
"""
        
        try:
            return await self.generate_text(prompt, task_type="summarization")
        except Exception as e:
            logger.error(f"Ошибка суммаризации: {e}")
            return content[:200] + "..." if len(content) > 200 else content
    
    async def classify_memory_type(self, content: str) -> str:
        """Классификация типа воспоминания"""
        prompt = f"""
Определи тип следующего воспоминания. Верни только одно слово из списка:
- conversation (разговор)
- fact (факт)
- preference (предпочтение)
- event (событие)
- task (задача)
- knowledge (знание)

Текст: {content}
"""
        
        try:
            result = await self.generate_text(prompt, task_type="extraction")
            result = result.strip().lower()
            
            valid_types = ["conversation", "fact", "preference", "event", "task", "knowledge"]
            if result in valid_types:
                return result
            else:
                return "conversation"  # По умолчанию
        except Exception as e:
            logger.error(f"Ошибка классификации: {e}")
            return "conversation"
    
    async def analyze_large_content(self, content: str) -> Dict[str, Any]:
        """Анализ большого контента с разбиением на части"""
        max_chunk_size = self.config.get("stability", {}).get("max_content_length", 4000)
        
        if len(content) <= max_chunk_size:
            return await self.analyze_memory_content(content)
        
        logger.info(f"Контент слишком большой ({len(content)} символов), разбиваем на части")
        
        # Разбиваем контент на части
        chunks = self._split_content_into_chunks(content, max_chunk_size)
        logger.info(f"Контент разбит на {len(chunks)} частей")
        
        # Анализируем каждую часть
        chunk_analyses = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Анализ части {i + 1}/{len(chunks)}")
            try:
                analysis = await self.analyze_memory_content(chunk)
                chunk_analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Ошибка анализа части {i + 1}: {e}")
                # Используем fallback для этой части
                fallback = self._create_fallback_analysis(chunk)
                chunk_analyses.append(fallback)
        
        # Объединяем результаты
        return self._merge_chunk_analyses(chunk_analyses, content)
    
    def _split_content_into_chunks(self, content: str, max_size: int) -> List[str]:
        """Разбиение контента на части"""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + max_size
            
            # Если это не последняя часть, ищем хорошее место для разрыва
            if end < len(content):
                # Ищем последний пробел, точку или перенос строки
                for i in range(end, start + max_size // 2, -1):
                    if content[i] in [' ', '.', '\n', '\r']:
                        end = i + 1
                        break
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end
        
        return chunks
    
    def _merge_chunk_analyses(self, analyses: List[Dict[str, Any]], original_content: str) -> Dict[str, Any]:
        """Объединение анализов частей в один результат"""
        if not analyses:
            return self._create_fallback_analysis(original_content)
        
        # Объединяем ключевые слова
        all_keywords = []
        for analysis in analyses:
            if "keywords" in analysis:
                keywords = analysis["keywords"]
                if isinstance(keywords, str):
                    all_keywords.extend([k.strip() for k in keywords.split(",")])
                elif isinstance(keywords, list):
                    all_keywords.extend(keywords)
        
        # Убираем дубликаты и ограничиваем количество
        unique_keywords = list(dict.fromkeys(all_keywords))[:10]
        
        # Объединяем сущности
        all_entities = []
        for analysis in analyses:
            if "entities" in analysis:
                entities = analysis["entities"]
                if isinstance(entities, str):
                    all_entities.extend([e.strip() for e in entities.split(",")])
                elif isinstance(entities, list):
                    all_entities.extend(entities)
        
        unique_entities = list(dict.fromkeys(all_entities))[:10]
        
        # Определяем общий тип и важность
        types = [analysis.get("type", "conversation") for analysis in analyses]
        importances = [analysis.get("importance", "medium") for analysis in analyses]
        
        # Выбираем наиболее частый тип
        from collections import Counter
        most_common_type = Counter(types).most_common(1)[0][0] if types else "conversation"
        
        # Выбираем максимальную важность
        importance_priority = {"high": 3, "medium": 2, "low": 1}
        max_importance = max(importances, key=lambda x: importance_priority.get(x, 2))
        
        # Создаем общее резюме
        summaries = [analysis.get("summary", "") for analysis in analyses if analysis.get("summary")]
        combined_summary = " ".join(summaries)[:500] + "..." if len(" ".join(summaries)) > 500 else " ".join(summaries)
        
        return {
            "summary": combined_summary or original_content[:200] + "...",
            "keywords": ", ".join(unique_keywords),
            "type": most_common_type,
            "importance": max_importance,
            "entities": ", ".join(unique_entities),
            "sentiment": "neutral",  # Для больших контентов сложно определить общий sentiment
            "content_chunked": True,
            "chunks_analyzed": len(analyses),
            "original_length": len(original_content)
        }

    async def health_check(self) -> bool:
        """Проверка доступности Ollama"""
        try:
            response = await self.client.get("/models")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama недоступен: {e}")
            return False
    
    def _is_valid_fact_triple(self, subject: str, action: str, object_: str) -> bool:
        """
        Проверяет, является ли тройка [субъект, действие, объект] валидным фактом
        
        Args:
            subject: Субъект действия
            action: Действие (должно быть глаголом)
            object_: Объект действия
            
        Returns:
            bool: True если факт валиден
        """
        # Базовые проверки
        if not subject or not action or not object_:
            return False
        
        # Проверяем, что субъект и объект не одинаковые
        if subject.lower() == object_.lower():
            return False
        
        # Проверяем, что действие не является существительным (простая эвристика)
        # Русские глаголы часто заканчиваются на -ет, -ит, -ает, -ует, -ет, -ит
        russian_verb_endings = ['ет', 'ит', 'ает', 'ует', 'ет', 'ит', 'ает', 'ует', 'ет', 'ит', 'ает', 'ует']
        
        # Проверяем, что действие содержит глагольные окончания или является глаголом
        action_lower = action.lower()
        is_verb = any(action_lower.endswith(ending) for ending in russian_verb_endings)
        
        # Дополнительные проверки на глаголы
        common_verbs = ['работает', 'живет', 'создала', 'любит', 'является', 'есть', 'находится', 'расположен', 'содержит', 'включает']
        is_common_verb = action_lower in common_verbs
        
        return is_verb or is_common_verb

    async def process_memory_unified(self, content: str, tasks: List[str] = None) -> Dict[str, Any]:
        """
        Единый оптимизированный вызов для обработки памяти
        Объединяет entities, facts, summary в один запрос для ускорения
        
        Args:
            content: Текст для обработки
            tasks: Список задач ["entities", "facts", "summary"]
            
        Returns:
            Словарь с результатами всех задач
        """
        if tasks is None:
            tasks = ["entities", "facts", "summary"]
        
        # Упрощенный промпт для стабильного JSON
        prompt = f"""Анализируй текст и верни JSON:

ТЕКСТ: "{content}"

Верни ТОЛЬКО валидный JSON в формате:
{{
  "entities": [{{"text": "имя", "type": "person"}}],
  "facts": [["субъект", "действие", "объект"]],
  "summary": "краткое описание"
}}

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. entities: ТОЛЬКО объекты с полями "text" и "type" - НЕ добавляй их в facts!
2. facts: ТОЛЬКО массивы из 3 элементов [субъект, глагол, объект] - НЕ добавляй entities в facts!
3. summary: ТОЛЬКО строка с описанием - НЕ добавляй в facts!
4. КРИТИЧЕСКИ ВАЖНО: после каждого массива ] должна быть запятая , перед следующим ключом
5. НЕ смешивай entities и facts - они РАЗНЫЕ типы данных!
6. АНАЛИЗИРУЙ ТОЛЬКО ПРЕДОСТАВЛЕННЫЙ ТЕКСТ - НЕ выдумывай информацию!
7. НЕ копируй примеры из промпта - создавай результат на основе РЕАЛЬНОГО текста!

ВАЖНО: Анализируй ТОЛЬКО предоставленный текст, НЕ используй примеры!

НЕПРАВИЛЬНЫЙ ПРИМЕР (НЕ ДЕЛАЙ ТАК!):
{{
  "entities": [{{"text": "Иван", "type": "person"}}],
  "facts": [["Иван", "работает", "в Google"], {{"text": "Google", "type": "organization"}}],
  "summary": "Иван работает в Google."
}}

ЕЩЕ НЕПРАВИЛЬНЫЙ ПРИМЕР (НЕ ДЕЛАЙ ТАК!):
{{
  "entities": [{{"text": "Иван", "type": "person"}}],
  "facts": [["Иван", "работает", "в Google"], {{"text": "Google", "typedependent_on_entity": "organization"}}],
  "summary": "Иван работает в Google."
}}

Верни ТОЛЬКО валидный JSON без дополнительного текста!"""
        
        try:
            logger.info(f"Unified LLM processing: tasks={tasks}, content_length={len(content)}")
            
            # Единый вызов к LLM с агрессивной оптимизацией
            response = await self.generate_text(
                prompt, 
                max_tokens=300,  # Агрессивно ограничиваем для скорости
                temperature=0.3,  # Низкая температура для стабильности
                task_type="unified_processing"
            )
            
            # Парсим JSON ответ - СУПЕР АГРЕССИВНАЯ очистка
            import json
            import re
            
            # Убеждаемся, что response - это строка с правильной кодировкой
            if isinstance(response, bytes):
                response = response.decode('utf-8', errors='ignore')
            elif not isinstance(response, str):
                response = str(response)
            
            # Безопасное логирование с правильной кодировкой
            try:
                safe_response = response[:200].encode('utf-8', errors='replace').decode('utf-8')
                logger.debug(f"Raw LLM response: {safe_response}...")
            except Exception:
                logger.debug(f"Raw LLM response: [encoding error] {len(response)} chars")
            
            # СУПЕР АГРЕССИВНАЯ очистка ответа
            cleaned_response = response.strip()
            
            # Убираем markdown блоки ```json и ```
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # убираем ```json
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]  # убираем ```
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # убираем ```
            
            # Убираем лишние символы и комментарии
            cleaned_response = cleaned_response.strip()
            
            # Ищем JSON объект в тексте (между { и })
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                cleaned_response = cleaned_response[json_start:json_end + 1]
            
            # ДОПОЛНИТЕЛЬНАЯ очистка: убираем все после последней закрывающей скобки
            # Ищем последний валидный JSON объект
            brace_count = 0
            last_valid_end = -1
            for i, char in enumerate(cleaned_response):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_valid_end = i
                        break
            
            if last_valid_end != -1:
                cleaned_response = cleaned_response[:last_valid_end + 1]
            
            # Минимальная очистка: убираем только комментарии
            # json-repair справится с остальными проблемами автоматически
            cleaned_response = re.sub(r'//.*?(?=\n|$)', '', cleaned_response)  # убираем // комментарии
            cleaned_response = re.sub(r'/\*.*?\*/', '', cleaned_response, flags=re.DOTALL)  # убираем /* */ комментарии
            
            # Безопасное логирование с правильной кодировкой
            try:
                safe_cleaned = cleaned_response[:200].encode('utf-8', errors='replace').decode('utf-8')
                logger.debug(f"Cleaned response: {safe_cleaned}...")
            except Exception:
                logger.debug(f"Cleaned response: [encoding error] {len(cleaned_response)} chars")
            
            try:
                result = json.loads(cleaned_response)
                
                # Валидируем структуру
                if not isinstance(result, dict):
                    raise ValueError("Response is not a dictionary")
                
                # Обеспечиваем наличие всех полей
                unified_result = {
                    "entities": result.get("entities", []),
                    "facts": result.get("facts", []),
                    "summary": result.get("summary", content[:200] + "..." if len(content) > 200 else content)
                }
                
                logger.info(f"Unified processing completed: entities={len(unified_result['entities'])}, facts={len(unified_result['facts'])}")
                return unified_result
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse unified LLM response as JSON: {e}")
                # Безопасное логирование с правильной кодировкой
                try:
                    safe_problematic = cleaned_response[:500].encode('utf-8', errors='replace').decode('utf-8')
                    logger.warning(f"Problematic response: {safe_problematic}...")
                except Exception:
                    logger.warning(f"Problematic response: [encoding error] {len(cleaned_response)} chars")
                
                # Улучшенная попытка исправления JSON
                try:
                    logger.info("Attempting to repair JSON with improved logic...")
                    
                    # Сначала пробуем использовать json_repair библиотеку
                    if JSON_REPAIR_AVAILABLE and repair_json:
                        try:
                            logger.debug("Trying json_repair library...")
                            repaired_json = repair_json(cleaned_response)
                            result = json.loads(repaired_json)
                            logger.info("JSON successfully repaired with json_repair library!")
                            
                            # Валидируем структуру
                            if not isinstance(result, dict):
                                raise ValueError("Repaired response is not a dictionary")
                            
                            # Обеспечиваем наличие всех полей
                            unified_result = {
                                "entities": result.get("entities", []),
                                "facts": result.get("facts", []),
                                "summary": result.get("summary", content[:200] + "..." if len(content) > 200 else content)
                            }
                            
                            # ВАЛИДАЦИЯ И ИСПРАВЛЕНИЕ ФАКТОВ
                            if "facts" in unified_result and isinstance(unified_result["facts"], list):
                                validated_facts = []
                                moved_entities = []
                                
                                for fact in unified_result["facts"]:
                                    if isinstance(fact, list) and len(fact) == 3:
                                        # Проверяем, что второй элемент - это глагол (действие)
                                        subject, action, object_ = fact
                                        if self._is_valid_fact_triple(subject, action, object_):
                                            validated_facts.append(fact)
                                        else:
                                            logger.warning(f"Invalid fact triple: {fact} - skipping")
                                    elif isinstance(fact, dict) and "text" in fact and "type" in fact:
                                        # Это entity, который попал в facts - перемещаем в entities
                                        moved_entities.append(fact)
                                        logger.warning(f"Entity in facts array: {fact} - moving to entities")
                                    elif isinstance(fact, dict) and ("text" in fact or "typedependent_on_entity" in fact):
                                        # Это entity с другими ключами - также перемещаем
                                        logger.warning(f"Entity-like object found in facts array: {fact} - moving to entities")
                                        # Нормализуем entity
                                        normalized_entity = {
                                            "text": fact.get("text", fact.get("name", "unknown")),
                                            "type": fact.get("type", fact.get("entity_type", fact.get("typedependent_on_entity", "unknown")))
                                        }
                                        moved_entities.append(normalized_entity)
                                    else:
                                        logger.warning(f"Invalid fact format: {fact} - skipping")
                                
                                # Добавляем перемещенные entities
                                if moved_entities:
                                    if "entities" not in unified_result:
                                        unified_result["entities"] = []
                                    unified_result["entities"].extend(moved_entities)
                                    logger.info(f"Moved {len(moved_entities)} entities from facts to entities")
                                
                                unified_result["facts"] = validated_facts
                                logger.info(f"Validated facts: {len(validated_facts)} valid facts from {len(unified_result.get('facts', []))} total")
                            
                            logger.info(f"JSON repair library success: entities={len(unified_result['entities'])}, facts={len(unified_result['facts'])}")
                            return unified_result
                            
                        except Exception as json_repair_error:
                            logger.warning(f"json_repair library failed: {json_repair_error}")
                            # Продолжаем с ручным исправлением
                    
                    # Агрессивная очистка проблемных паттернов
                    fixed_response = cleaned_response
                    
                    # Исправляем типичные ошибки LLM
                    fixed_response = fixed_response.replace('"typedependent:', '"type": "')
                    fixed_response = fixed_response.replace('"textdependent:', '"text": "')
                    fixed_response = fixed_response.replace('"type": "organization"}', '"type": "organization"}')
                    
                    # Убираем лишние запятые перед закрывающими скобками
                    fixed_response = re.sub(r',\s*}', '}', fixed_response)
                    fixed_response = re.sub(r',\s*]', ']', fixed_response)
                    
                    # Убираем дублирующие элементы в массивах
                    fixed_response = re.sub(r'},\s*{', '}, {', fixed_response)
                    
                    # Исправляем отсутствующие запятые между элементами массива
                    fixed_response = re.sub(r'"\s*\n\s*"', '",\n"', fixed_response)
                    fixed_response = re.sub(r'}\s*\n\s*{', '},\n{', fixed_response)
                    fixed_response = re.sub(r']\s*\n\s*"', '],\n"', fixed_response)
                    
                    # Исправляем отсутствующие запятые между ] и следующими ключами
                    fixed_response = re.sub(r']\s*\n\s*"([^"]+)":', r'],\n"\1":', fixed_response)
                    fixed_response = re.sub(r']\s*"([^"]+)":', r'],\n"\1":', fixed_response)
                    
                    # ИСПРАВЛЕНИЕ JSON: добавляем отсутствующие запятые после закрывающих скобок массивов
                    # Исправляем случай: "текст"], "summary": -> "текст"], "summary":
                    fixed_response = re.sub(r'([^"]+"])\s*\n\s*"([^"]+)":', r'\1,\n"\2":', fixed_response)
                    fixed_response = re.sub(r'([^"]+"])\s*"([^"]+)":', r'\1, "\2":', fixed_response)
                    
                    # Исправляем случай: ], "summary": -> ], "summary":
                    fixed_response = re.sub(r']\s*\n\s*"([^"]+)":', r'],\n"\1":', fixed_response)
                    fixed_response = re.sub(r']\s*"([^"]+)":', r'], "\1":', fixed_response)
                    
                    # Пытаемся парсить исправленный JSON
                    result = json.loads(fixed_response)
                    
                    # Валидируем структуру
                    if not isinstance(result, dict):
                        raise ValueError("Repaired response is not a dictionary")
                    
                    # Обеспечиваем наличие всех полей
                    unified_result = {
                        "entities": result.get("entities", []),
                        "facts": result.get("facts", []),
                        "summary": result.get("summary", content[:200] + "..." if len(content) > 200 else content)
                    }
                    
                    # ВАЛИДАЦИЯ И ИСПРАВЛЕНИЕ ФАКТОВ
                    if "facts" in unified_result and isinstance(unified_result["facts"], list):
                        validated_facts = []
                        moved_entities = []
                        
                        for fact in unified_result["facts"]:
                            if isinstance(fact, list) and len(fact) == 3:
                                # Проверяем, что второй элемент - это глагол (действие)
                                subject, action, object_ = fact
                                if self._is_valid_fact_triple(subject, action, object_):
                                    validated_facts.append(fact)
                                else:
                                    logger.warning(f"Invalid fact triple: {fact} - skipping")
                            elif isinstance(fact, dict) and "text" in fact and "type" in fact:
                                # Это entity, который попал в facts - перемещаем в entities
                                moved_entities.append(fact)
                                logger.warning(f"Entity in facts array: {fact} - moving to entities")
                            else:
                                logger.warning(f"Invalid fact format: {fact} - skipping")
                        
                        # Добавляем перемещенные entities
                        if moved_entities:
                            if "entities" not in unified_result:
                                unified_result["entities"] = []
                            unified_result["entities"].extend(moved_entities)
                            logger.info(f"Moved {len(moved_entities)} entities from facts to entities")
                        
                        unified_result["facts"] = validated_facts
                        logger.info(f"Validated facts: {len(validated_facts)} valid facts from {len(unified_result.get('facts', []))} total")
                    
                    logger.info(f"JSON successfully repaired! Unified processing completed: entities={len(unified_result['entities'])}, facts={len(unified_result['facts'])}")
                    return unified_result
                    
                except Exception as repair_error:
                    logger.warning(f"JSON repair failed: {repair_error}")
                
                # Очищаем кэш для unified_processing, так как он содержит некорректные данные
                await self.clear_cache("unified_processing")
                
                # Fallback: создаем базовую структуру
                return {
                    "entities": [],
                    "facts": [],
                    "summary": content[:200] + "..." if len(content) > 200 else content
                }
                
        except Exception as e:
            logger.error(f"Error in unified memory processing: {e}")
            # Fallback: создаем базовую структуру
            return {
                "entities": [],
                "facts": [],
                "summary": content[:200] + "..." if len(content) > 200 else content
            }
    
    async def close(self):
        """Закрытие соединения"""
        await self.client.aclose()
        logger.info("Ollama Provider закрыт")
