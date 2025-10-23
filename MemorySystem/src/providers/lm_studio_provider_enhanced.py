"""
LM Studio Provider Enhanced для AIRI Memory System
Улучшенная версия с кэшированием, оптимизацией промптов и мониторингом
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
import httpx
from loguru import logger
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import os
import sys

# Добавляем путь к модулям MemorySystem
sys.path.append(str(Path(__file__).parent.parent))
from emotion_formatter import EmotionFormatter, EmotionData, FormattedEmotion

class LMStudioProviderEnhanced:
    """Улучшенный провайдер для работы с LM Studio с кэшированием"""
    
    def __init__(self, config_path: str = "config/lm_studio_config.yaml"):
        """Инициализация провайдера LM Studio"""
        self.config = self._load_config(config_path)
        self.client = httpx.AsyncClient(
            timeout=self.config.get("timeout", 120),
            base_url=self.config["base_url"]
        )
        
        # Кэш для результатов
        self.cache = {}
        self.cache_file = "lm_studio_cache.pkl"
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 час по умолчанию
        
        # Статистика
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time": 0.0,
            "average_time": 0.0
        }
        
        # Загружаем кэш из файла
        self._load_cache()
        
        # Инициализируем форматтер эмоций
        self.emotion_formatter = EmotionFormatter()
        
        logger.info(f"LM Studio Provider Enhanced инициализирован: {self.config['base_url']}")
        logger.info(f"Кэш TTL: {self.cache_ttl} секунд")
        logger.info(f"Форматтер эмоций: готов")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        if isinstance(config_path, dict):
            return config_path
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # Добавляем настройки кэширования
                config.setdefault("cache_ttl", 3600)  # 1 час
                config.setdefault("cache_enabled", True)
                config.setdefault("prompt_optimization", True)
                return config
        except FileNotFoundError:
            logger.warning(f"Конфигурационный файл {config_path} не найден, используются настройки по умолчанию")
            return {
                "base_url": "http://localhost:11434/v1",
                "model": "gemma2:2b",
                "max_tokens": 2048,
                "temperature": 0.7,
                "timeout": 120,
                "retry_attempts": 3,
                "cache_ttl": 3600,
                "cache_enabled": True,
                "prompt_optimization": True
            }
    
    def _load_cache(self):
        """Загрузка кэша из файла"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Кэш загружен: {len(self.cache)} записей")
            else:
                self.cache = {}
        except Exception as e:
            logger.warning(f"Ошибка загрузки кэша: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Сохранение кэша в файл"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Ошибка сохранения кэша: {e}")
    
    def _get_cache_key(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Генерация ключа кэша"""
        content = f"{prompt}|{max_tokens}|{temperature}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Проверка валидности записи кэша"""
        if not self.config.get("cache_enabled", True):
            return False
        
        timestamp = cache_entry.get("timestamp", 0)
        return time.time() - timestamp < self.cache_ttl
    
    def _optimize_prompt(self, prompt: str) -> str:
        """Оптимизация промпта для ускорения генерации"""
        if not self.config.get("prompt_optimization", True):
            return prompt
        
        # Убираем избыточные пробелы
        prompt = " ".join(prompt.split())
        
        # Сокращаем повторяющиеся фразы
        lines = prompt.split('\n')
        optimized_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                optimized_lines.append(line)
                seen_lines.add(line)
        
        optimized_prompt = '\n'.join(optimized_lines)
        
        # Логируем оптимизацию
        if len(optimized_prompt) < len(prompt):
            reduction = len(prompt) - len(optimized_prompt)
            logger.info(f"Промпт оптимизирован: -{reduction} символов ({reduction/len(prompt)*100:.1f}%)")
        
        return optimized_prompt
    
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True
    ) -> str:
        """Генерация текста с кэшированием"""
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        # Параметры по умолчанию
        max_tokens = max_tokens or self.config.get("max_tokens", 512)
        temperature = temperature or self.config.get("temperature", 0.7)
        
        # Оптимизируем промпт
        optimized_prompt = self._optimize_prompt(prompt)
        
        # Проверяем кэш
        if use_cache and self.config.get("cache_enabled", True):
            cache_key = self._get_cache_key(optimized_prompt, max_tokens, temperature)
            
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if self._is_cache_valid(cache_entry):
                    self.stats["cache_hits"] += 1
                    processing_time = time.time() - start_time
                    self.stats["total_time"] += processing_time
                    self.stats["average_time"] = self.stats["total_time"] / self.stats["total_requests"]
                    
                    logger.info(f"Кэш HIT: {processing_time:.3f}с (ключ: {cache_key[:8]}...)")
                    return cache_entry["response"]
                else:
                    # Удаляем устаревшую запись
                    del self.cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        # Генерируем текст
        try:
            response = await self._generate_with_retry(optimized_prompt, max_tokens, temperature)
            
            # Сохраняем в кэш
            if use_cache and self.config.get("cache_enabled", True):
                cache_key = self._get_cache_key(optimized_prompt, max_tokens, temperature)
                self.cache[cache_key] = {
                    "response": response,
                    "timestamp": time.time(),
                    "prompt_length": len(optimized_prompt),
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                self._save_cache()
            
            processing_time = time.time() - start_time
            self.stats["total_time"] += processing_time
            self.stats["average_time"] = self.stats["total_time"] / self.stats["total_requests"]
            
            logger.info(f"Генерация завершена: {processing_time:.3f}с")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Ошибка генерации: {e} ({processing_time:.3f}с)")
            raise
    
    async def _generate_with_retry(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> str:
        """Генерация с повторными попытками"""
        retry_attempts = self.config.get("retry_attempts", 3)
        
        for attempt in range(retry_attempts):
            try:
                payload = {
                    "model": self.config["model"],
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
                
                response = await self.client.post("/chat/completions", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    logger.warning(f"HTTP {response.status_code}: {response.text}")
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(2 ** attempt)  # Экспоненциальная задержка
                        continue
                    else:
                        raise Exception(f"HTTP {response.status_code}: {response.text}")
                        
            except httpx.TimeoutException:
                logger.warning(f"Таймаут при обращении к Ollama (попытка {attempt + 1}/{retry_attempts})")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise Exception("Таймаут при обращении к LM Studio")
            except Exception as e:
                logger.error(f"Ошибка при обращении к Ollama (попытка {attempt + 1}/{retry_attempts}): {e}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise
    
    async def analyze_memory_content(self, content: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Анализ содержимого памяти с кэшированием"""
        # Создаем промпт для анализа
        prompt = self._create_analysis_prompt(content, context)
        
        # Используем настройки для анализа
        analysis_config = self.config.get("tasks", {}).get("analysis", {})
        max_tokens = analysis_config.get("max_tokens", 256)
        temperature = analysis_config.get("temperature", 0.3)
        
        try:
            response_text = await self.generate_text(prompt, max_tokens, temperature)
            return self._parse_analysis_response(response_text)
        except Exception as e:
            logger.error(f"Ошибка анализа содержимого: {e}")
            return self._get_fallback_analysis(content)
    
    def _create_analysis_prompt(self, content: str, context: Optional[str] = None) -> str:
        """Создание оптимизированного промпта для анализа"""
        base_prompt = f"""Проанализируй следующий текст и верни результат в формате JSON:

Текст: "{content[:500]}{'...' if len(content) > 500 else ''}"

Верни JSON с полями:
- summary: краткое описание (до 50 слов)
- keywords: ключевые слова (массив строк)
- type: тип события (conversation, question, statement, etc.)
- importance: важность (low, medium, high)
- entities: именованные сущности (массив строк)
- sentiment: эмоциональная окраска (positive, negative, neutral)"""
        
        if context:
            base_prompt += f"\n\nКонтекст: {context[:200]}{'...' if len(context) > 200 else ''}"
        
        return base_prompt
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Парсинг ответа анализа"""
        try:
            # Ищем JSON в ответе
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("JSON не найден в ответе")
        except Exception as e:
            logger.warning(f"Ошибка парсинга ответа: {e}")
            return self._get_fallback_analysis(response_text)
    
    def _get_fallback_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback анализ при ошибках"""
        return {
            "summary": content[:100] + "..." if len(content) > 100 else content,
            "keywords": content.split()[:5],
            "type": "unknown",
            "importance": "medium",
            "entities": [],
            "sentiment": "neutral"
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        valid_entries = sum(1 for entry in self.cache.values() if self._is_cache_valid(entry))
        total_entries = len(self.cache)
        
        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": total_entries - valid_entries,
            "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["total_requests"], 1),
            "average_response_time": self.stats["average_time"],
            "total_requests": self.stats["total_requests"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"]
        }
    
    def create_emotion_enhanced_prompt(
        self, 
        base_prompt: str, 
        emotion_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Создает промпт с учетом эмоционального контекста"""
        
        if not emotion_data:
            return base_prompt
        
        try:
            # Создаем EmotionData из переданных данных
            emotion_obj = EmotionData(
                primary_emotion=emotion_data.get('primary_emotion', 'нейтральная'),
                primary_confidence=emotion_data.get('primary_confidence', 0.5),
                secondary_emotion=emotion_data.get('secondary_emotion'),
                secondary_confidence=emotion_data.get('secondary_confidence'),
                tertiary_emotion=emotion_data.get('tertiary_emotion'),
                tertiary_confidence=emotion_data.get('tertiary_confidence'),
                consistency=emotion_data.get('consistency', 'high'),
                dominant_source=emotion_data.get('dominant_source', 'voice'),
                validation_applied=emotion_data.get('validation_applied', False)
            )
            
            # Форматируем эмоции
            formatted_emotion = self.emotion_formatter.format_emotions_for_ai(emotion_obj)
            
            # Создаем адаптивный промпт
            enhanced_prompt = self.emotion_formatter.create_adaptive_prompt(formatted_emotion, base_prompt)
            
            logger.info(f"Создан эмоционально-усиленный промпт: {len(enhanced_prompt)} символов")
            logger.info(f"Сложность эмоций: {formatted_emotion.complexity.value}")
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Ошибка создания эмоционального промпта: {e}")
            return base_prompt
    
    async def generate_text_with_emotions(
        self, 
        prompt: str, 
        emotion_data: Optional[Dict[str, Any]] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Генерирует текст с учетом эмоционального контекста"""
        
        # Создаем эмоционально-усиленный промпт
        enhanced_prompt = self.create_emotion_enhanced_prompt(prompt, emotion_data)
        
        # Генерируем текст
        result = await self.generate_text(
            prompt=enhanced_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Добавляем информацию об эмоциях в результат
        if emotion_data:
            result["emotion_context"] = {
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "emotion_data": emotion_data,
                "prompt_enhancement": len(enhanced_prompt) - len(prompt)
            }
        
        return result
    
    def clear_cache(self):
        """Очистка кэша"""
        self.cache.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logger.info("Кэш очищен")
    
    def cleanup_expired_cache(self):
        """Очистка устаревших записей кэша"""
        expired_keys = []
        for key, entry in self.cache.items():
            if not self._is_cache_valid(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self._save_cache()
            logger.info(f"Удалено {len(expired_keys)} устаревших записей кэша")
    
    async def close(self):
        """Закрытие провайдера"""
        await self.client.aclose()
        self._save_cache()
        logger.info("LM Studio Provider Enhanced закрыт")

# Функция для тестирования
async def test_enhanced_provider():
    """Тест улучшенного провайдера"""
    print("🧪 Тестируем LM Studio Provider Enhanced...")
    
    provider = LMStudioProviderEnhanced()
    
    # Тестовый промпт
    test_prompt = "Привет, как дела? Расскажи что-нибудь интересное."
    
    try:
        # Первый запрос (кэш miss)
        print("🔄 Первый запрос (кэш miss)...")
        start_time = time.time()
        response1 = await provider.generate_text(test_prompt, max_tokens=100, temperature=0.7)
        time1 = time.time() - start_time
        print(f"   Время: {time1:.2f}с")
        print(f"   Ответ: {response1[:100]}...")
        
        # Второй запрос (кэш hit)
        print("\n🔄 Второй запрос (кэш hit)...")
        start_time = time.time()
        response2 = await provider.generate_text(test_prompt, max_tokens=100, temperature=0.7)
        time2 = time.time() - start_time
        print(f"   Время: {time2:.2f}с")
        print(f"   Ответ: {response2[:100]}...")
        
        # Проверяем, что ответы одинаковые
        if response1 == response2:
            print("✅ Ответы идентичны")
        else:
            print("❌ Ответы различаются")
        
        # Статистика
        stats = provider.get_cache_stats()
        print(f"\n📊 Статистика:")
        print(f"   Всего запросов: {stats['total_requests']}")
        print(f"   Кэш попадания: {stats['cache_hits']}")
        print(f"   Кэш промахи: {stats['cache_misses']}")
        print(f"   Hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"   Среднее время: {stats['average_response_time']:.2f}с")
        print(f"   Ускорение: {time1/time2:.1f}x")
        
        await provider.close()
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        await provider.close()

if __name__ == "__main__":
    asyncio.run(test_enhanced_provider())
