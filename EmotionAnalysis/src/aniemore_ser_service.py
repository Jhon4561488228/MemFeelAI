#!/usr/bin/env python3
"""
🎭 Aniemore SER Service
HTTP сервис для анализа эмоций в голосе с помощью Aniemore/wavlm-emotion-russian-resd
"""

import asyncio
import logging
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
import base64
import aiohttp

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, AutoTokenizer, AutoModelForSequenceClassification
from aniemore.recognizers.voice import VoiceRecognizer
from aniemore.recognizers.text import TextRecognizer
from aniemore.models import HuggingFaceModel

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("[INFO] Запуск Aniemore SER Service...")
    
    # Попытки инициализации
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            logger.info(f"[INFO] Попытка инициализации {attempt + 1}/{max_retries}")
            await ser_service.initialize()
            logger.info("[INFO] Aniemore SER Service готов к работе")
            break
        except Exception as e:
            logger.error(f"[ERROR] Ошибка инициализации (попытка {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"[INFO] Повторная попытка через {retry_delay} секунд...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Экспоненциальная задержка
            else:
                logger.warning("[WARNING] Не удалось инициализировать сервис при запуске")
                logger.info("[INFO] Aniemore SER Service запущен (инициализация отложена до первого запроса)")
    
    yield
    # Shutdown
    logger.info("[INFO] Aniemore SER Service завершает работу...")
    if hasattr(app.state, 'voice_recognizer') and app.state.voice_recognizer:
        logger.info("[INFO] Очистка ресурсов Aniemore...")
        # Очистка ресурсов Aniemore (если нужно)
    logger.info("[INFO] Aniemore SER Service завершен")

app = FastAPI(
    title="Aniemore SER Service",
    description="""
    Сервис анализа эмоций в голосе и тексте с помощью Aniemore.
    
    **Важно**: WavLM - это аудио-только модель для анализа эмоций в голосе.
    Текст анализируется отдельно через Aniemore Text и Dostoevsky модели.
    Финальное решение принимается на основе взвешенного объединения результатов.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    try:
        logger.info("[ROOT] Запрос к корневому endpoint")
        return {
            "service": "Aniemore SER Service",
            "version": "1.0.0",
            "status": "running",
            "endpoints": [
                "/health",
                "/model-info",
                "/analyze-bytes",
                "/analyze-bytes-json",
                "/analyze-text"
            ]
        }
    except Exception as e:
        logger.error(f"[ROOT] Ошибка: {e}")
        import traceback
        logger.error(f"[ROOT] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

class AniemoreSERService:
    def __init__(self):
        self.voice_recognizer = None
        self.text_recognizer = None
        # Принудительно используем CPU из-за CUDA ошибок
        self.device = "cpu"
        self.is_initialized = False
        self.text_initialized = False
        
        # Пути к локальным моделям
        self.models_dir = Path(__file__).parent.parent / "models"
        self.voice_model_path = self.models_dir / "aniemore" / "voice"
        self.text_model_path = self.models_dir / "aniemore" / "text"
        
        # Проверяем существование локальных моделей
        self.use_local_models = self.voice_model_path.exists() and self.text_model_path.exists()
        
        if self.use_local_models:
            logger.info(f"[INFO] Используем локальные модели Aniemore:")
            logger.info(f"[INFO] Voice: {self.voice_model_path}")
            logger.info(f"[INFO] Text: {self.text_model_path}")
            self.voice_model = str(self.voice_model_path)
            self.text_model = str(self.text_model_path)
            self.model_name = "Aniemore/wavlm-emotion-russian-resd (local)"
            self.text_model_name = "Aniemore/rubert-tiny2-russian-emotion-detection (local)"
        else:
            logger.info(f"[INFO] Локальные модели не найдены, используем HuggingFace:")
            logger.info(f"[INFO] Voice: {self.voice_model_path} - {self.voice_model_path.exists()}")
            logger.info(f"[INFO] Text: {self.text_model_path} - {self.text_model_path.exists()}")
            self.voice_model = HuggingFaceModel.Voice.WavLM
            self.text_model = HuggingFaceModel.Text.Bert_Tiny2
            self.model_name = "Aniemore/wavlm-emotion-russian-resd"
            self.text_model_name = "Aniemore/rubert-tiny2-russian-emotion-detection"
        
        # Добавляем недостающие атрибуты
        self.model = None  # Будет установлен при инициализации
        
        # Маппинг эмоций на русский язык
        self.emotion_mapping = {
            "angry": "злость",
            "anger": "злость",
            "calm": "спокойствие", 
            "disgust": "отвращение",
            "fearful": "страх",
            "fear": "страх",
            "happy": "радость",
            "happiness": "радость",
            "enthusiasm": "энтузиазм",
            "neutral": "нейтральная",
            "sad": "грусть",
            "sadness": "грусть",
            "surprised": "удивление"
        }
        
        # URL других сервисов
        self.whispercpp_url = "http://localhost:8002/transcribe"
        self.dostoevsky_url = "http://localhost:8007/analyze"
        
        # Таймауты для HTTP запросов
        self.timeouts = {
            "whispercpp": 30,
            "dostoevsky": 30
        }
        
        # Веса убраны - объединение эмоций теперь происходит в Rust
    
    async def initialize(self):
        """Инициализация модели Aniemore SER"""
        if self.is_initialized:
            return
        
        logger.info(f"[INFO] Инициализация Aniemore Voice Recognizer: {self.voice_model}")
        
        try:
            if self.use_local_models:
                # Используем локальную модель напрямую через transformers
                logger.info(f"[INFO] Загрузка локальной голосовой модели из: {self.voice_model}")
                from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
                
                # Находим snapshot директорию
                snapshots_dir = Path(self.voice_model) / "snapshots"
                if snapshots_dir.exists():
                    snapshot_dirs = list(snapshots_dir.iterdir())
                    if snapshot_dirs:
                        model_path = snapshot_dirs[0]  # Берем первый snapshot
                        logger.info(f"[INFO] Используем snapshot: {model_path}")
                        
                        # Загружаем модель и feature extractor
                        # ИСПРАВЛЕНО: Всегда используем float32 для совместимости с CUDA
                        self.model = AutoModelForAudioClassification.from_pretrained(
                            str(model_path), 
                            torch_dtype=torch.float32  # Всегда float32 для совместимости
                        ).to(self.device)
                        
                        self.feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_path))
                        
                        # Создаем реальный VoiceRecognizer для локальных моделей
                        class LocalVoiceRecognizer:
                            def __init__(self, model, feature_extractor, device):
                                self.model = model
                                self.feature_extractor = feature_extractor
                                self.device = device
                            
                            def recognize(self, audio_path, return_single_label=True):
                                """Анализ эмоций в аудио файле"""
                                import librosa
                                import torch
                                
                                # Загружаем аудио
                                audio, sr = librosa.load(audio_path, sr=16000)
                                
                                # Извлекаем features
                                inputs = self.feature_extractor(
                                    audio, 
                                    sampling_rate=16000, 
                                    return_tensors="pt"
                                )
                                
                                # Переносим на устройство и приводим к типу модели
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                                
                                # ИСПРАВЛЕНО: Всегда используем float32 для совместимости
                                logger.info(f"Тип входных данных: {inputs['input_values'].dtype}")
                                logger.info(f"Модель загружена с типом: {next(self.model.parameters()).dtype}")
                                
                                # Убеждаемся что входные данные в float32
                                if inputs['input_values'].dtype != torch.float32:
                                    inputs = {k: v.float() for k, v in inputs.items()}
                                    logger.info(f"Приведены входные данные к float32: {inputs['input_values'].dtype}")
                                
                                # Предсказание
                                with torch.no_grad():
                                    outputs = self.model(**inputs)
                                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                                    
                                # Получаем топ-3 эмоции
                                top3_indices = torch.topk(predictions[0], k=3).indices
                                top3_confidences = torch.topk(predictions[0], k=3).values
                                
                                # Основная эмоция (топ-1)
                                predicted_class_id = top3_indices[0].item()
                                confidence = top3_confidences[0].item()
                                
                                # Получаем название основной эмоции
                                if hasattr(self.model.config, 'id2label'):
                                    emotion = self.model.config.id2label[predicted_class_id]
                                else:
                                    emotion = f"emotion_{predicted_class_id}"
                                
                                # Получаем топ-3 эмоции для детального анализа
                                top3_emotions = []
                                for i in range(3):
                                    class_id = top3_indices[i].item()
                                    conf = top3_confidences[i].item()
                                    if hasattr(self.model.config, 'id2label'):
                                        emotion_name = self.model.config.id2label[class_id]
                                    else:
                                        emotion_name = f"emotion_{class_id}"
                                    top3_emotions.append({
                                        'emotion': emotion_name,
                                        'confidence': conf,
                                        'class_id': class_id
                                    })
                                
                                logger.info(f"Топ-3 эмоции (голос): {top3_emotions}")
                                
                                if return_single_label:
                                    return emotion
                                else:
                                    return {
                                        'label': emotion,
                                        'score': confidence,
                                        'top3_emotions': top3_emotions
                                    }
                        
                        self.voice_recognizer = LocalVoiceRecognizer(self.model, self.feature_extractor, self.device)
                    else:
                        raise Exception("Не найдены snapshot директории в локальной модели")
                else:
                    raise Exception("Не найдена директория snapshots в локальной модели")
            else:
                # Используем HuggingFace модель через Aniemore API
                logger.info("[INFO] Загрузка голосовой модели из HuggingFace через Aniemore...")
                self.voice_recognizer = VoiceRecognizer(model=self.voice_model, device=self.device)
                self.model = self.voice_recognizer.model
            
            logger.info(f"[OK] Голосовая модель загружена на {self.device}")
            self.is_initialized = True
            
            # Инициализируем текстовую модель
            logger.info("[INFO] Инициализация текстовой модели Aniemore...")
            await self.initialize_text_model()
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка инициализации голосовой модели: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка инициализации голосовой модели: {e}")
    
    async def initialize_text_model(self):
        """Инициализация текстовой модели Aniemore"""
        if self.text_initialized:
            return
        
        logger.info(f"[INFO] Инициализация текстовой модели: {self.text_model}")
        
        try:
            if self.use_local_models:
                # Используем локальную модель напрямую через transformers
                logger.info(f"[INFO] Загрузка локальной текстовой модели из: {self.text_model}")
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                
                # Находим snapshot директорию
                snapshots_dir = Path(self.text_model) / "snapshots"
                if snapshots_dir.exists():
                    snapshot_dirs = list(snapshots_dir.iterdir())
                    if snapshot_dirs:
                        model_path = snapshot_dirs[0]  # Берем первый snapshot
                        logger.info(f"[INFO] Используем snapshot: {model_path}")
                        
                        # Загружаем модель и токенизатор
                        self.text_model_obj = AutoModelForSequenceClassification.from_pretrained(
                            str(model_path), 
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                        ).to(self.device)
                        
                        self.text_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                        
                        # Создаем реальный TextRecognizer для локальных моделей
                        class LocalTextRecognizer:
                            def __init__(self, model, tokenizer, device):
                                self.model = model
                                self.tokenizer = tokenizer
                                self.device = device
                            
                            def recognize(self, text, return_single_label=True):
                                """Анализ эмоций в тексте"""
                                import torch
                                
                                # Токенизируем текст
                                inputs = self.tokenizer(
                                    text, 
                                    return_tensors="pt", 
                                    truncation=True, 
                                    padding=True, 
                                    max_length=512
                                )
                                
                                # Переносим на устройство
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                                
                                # Предсказание
                                with torch.no_grad():
                                    outputs = self.model(**inputs)
                                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                                    
                                # Получаем топ-3 эмоции
                                top3_indices = torch.topk(predictions[0], k=3).indices
                                top3_confidences = torch.topk(predictions[0], k=3).values
                                
                                # Основная эмоция (топ-1)
                                predicted_class_id = top3_indices[0].item()
                                confidence = top3_confidences[0].item()
                                
                                # Получаем название основной эмоции
                                if hasattr(self.model.config, 'id2label'):
                                    emotion = self.model.config.id2label[predicted_class_id]
                                else:
                                    emotion = f"emotion_{predicted_class_id}"
                                
                                # Получаем топ-3 эмоции для детального анализа
                                top3_emotions = []
                                for i in range(3):
                                    class_id = top3_indices[i].item()
                                    conf = top3_confidences[i].item()
                                    if hasattr(self.model.config, 'id2label'):
                                        emotion_name = self.model.config.id2label[class_id]
                                    else:
                                        emotion_name = f"emotion_{class_id}"
                                    top3_emotions.append({
                                        'emotion': emotion_name,
                                        'confidence': conf,
                                        'class_id': class_id
                                    })
                                
                                logger.info(f"Топ-3 эмоции (текст): {top3_emotions}")
                                
                                if return_single_label:
                                    return emotion
                                else:
                                    return {
                                        'label': emotion,
                                        'score': confidence,
                                        'top3_emotions': top3_emotions
                                    }
                        
                        self.text_recognizer = LocalTextRecognizer(self.text_model_obj, self.text_tokenizer, self.device)
                    else:
                        raise Exception("Не найдены snapshot директории в локальной текстовой модели")
                else:
                    raise Exception("Не найдена директория snapshots в локальной текстовой модели")
            else:
                # Используем HuggingFace модель через Aniemore API
                logger.info("[INFO] Загрузка текстовой модели из HuggingFace через Aniemore...")
                self.text_recognizer = TextRecognizer(model=self.text_model, device=self.device)
            
            logger.info(f"[OK] Текстовая модель загружена на {self.device}")
            self.text_initialized = True
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка инициализации текстовой модели: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка инициализации текстовой модели: {e}")
    
    async def analyze_emotions(self, audio_data: bytes, sample_rate: int = 16000) -> Dict:
        """Анализ эмоций в аудио данных"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            logger.info(f"🎵 Входящие аудио данные: {len(audio_data)} bytes, sample_rate: {sample_rate} Hz")
            
            # Валидация длины аудио
            if len(audio_data) < 1600:  # < 0.1 сек при 16kHz
                raise HTTPException(status_code=400, detail="Аудио слишком короткое (минимум 0.1 секунды)")
            
            # Создаем временный файл для аудио
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file.flush()  # Убеждаемся, что данные записаны
                    temp_path = temp_file.name
                
                logger.info(f"📁 Создан временный файл: {temp_path}")
                
                # Валидация аудио файла
                try:
                    import librosa
                    audio, sr = librosa.load(temp_path, sr=None)
                    if len(audio) < 1600:  # < 0.1 сек
                        raise HTTPException(status_code=400, detail="Аудио слишком короткое после загрузки")
                    logger.info(f"🎵 Аудио загружено: {len(audio)} samples, {sr} Hz")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Невалидный аудио файл: {str(e)}")
                
                # Используем Aniemore VoiceRecognizer для анализа эмоций
                logger.info("🎭 Анализ эмоций с помощью Aniemore...")
                result = self.voice_recognizer.recognize(temp_path, return_single_label=True)
                
                logger.info(f"✅ Результат Aniemore: {result}")
                
                # Обрабатываем результат
                if isinstance(result, dict):
                    emotion = result.get('label', 'neutral')
                    confidence = result.get('score', 0.0)
                else:
                    # Если результат не словарь, значит это строка с эмоцией
                    # Нужно получить confidence из модели напрямую
                    emotion = str(result)
                    
                    # Получаем confidence из модели напрямую
                    try:
                        # Используем voice_recognizer для получения полного результата
                        full_result = self.voice_recognizer.recognize(temp_path, return_single_label=False)
                        if isinstance(full_result, dict):
                            confidence = full_result.get('score', 0.5)
                            logger.debug(f"Получен confidence для голоса из full_result: {confidence}")
                        else:
                            # Если full_result не словарь, попробуем получить confidence из модели напрямую
                            logger.warning(f"full_result для голоса не является словарем: {type(full_result)}, значение: {full_result}")
                            # Попробуем альтернативный способ получения confidence
                            try:
                                # Повторный вызов с детальным логированием
                                detailed_result = self.voice_recognizer.recognize(temp_path, return_single_label=False)
                                logger.debug(f"Детальный результат для голоса: {detailed_result}")
                                if isinstance(detailed_result, dict) and 'score' in detailed_result:
                                    confidence = detailed_result['score']
                                    logger.info(f"Успешно получен confidence для голоса из детального результата: {confidence}")
                                else:
                                    # Последняя попытка - используем минимальное значение вместо заглушки
                                    confidence = 0.1  # Минимальное значение вместо заглушки 0.5
                                    logger.warning(f"Не удалось получить confidence для голоса, используем минимальное значение: {confidence}")
                            except Exception as inner_e:
                                logger.error(f"Ошибка при попытке получить детальный результат для голоса: {inner_e}")
                                confidence = 0.1  # Минимальное значение вместо заглушки 0.5
                                logger.warning(f"Используем минимальное значение confidence для голоса: {confidence}")
                    except Exception as e:
                        logger.error(f"Критическая ошибка при получении confidence для голоса: {e}")
                        # Вместо заглушки 0.5 используем минимальное значение
                        confidence = 0.1
                        logger.warning(f"Используем минимальное значение confidence для голоса из-за ошибки: {confidence}")
                
                # Переводим эмоцию на русский
                russian_emotion = self.emotion_mapping.get(emotion.lower(), emotion)
                
                processing_time = time.time() - start_time
                
                # Получаем топ-3 эмоции из результата recognize
                top3_emotions = []
                try:
                    # Получаем полный результат с топ-3 эмоциями
                    full_result_with_top3 = self.voice_recognizer.recognize(temp_path, return_single_label=False)
                    if isinstance(full_result_with_top3, dict) and 'top3_emotions' in full_result_with_top3:
                        top3_emotions = full_result_with_top3['top3_emotions']
                        # Переводим эмоции на русский
                        for emotion_data in top3_emotions:
                            emotion_data['emotion'] = self.emotion_mapping.get(emotion_data['emotion'].lower(), emotion_data['emotion'])
                        logger.info(f"Получены топ-3 эмоции: {top3_emotions}")
                    else:
                        logger.warning(f"Не удалось получить топ-3 эмоции из full_result_with_top3: {full_result_with_top3}")
                except Exception as e:
                    logger.error(f"Ошибка при получении топ-3 эмоций: {e}")
                    top3_emotions = []
                
                result = {
                    "emotion": russian_emotion,
                    "confidence": float(confidence),
                    "processing_time": processing_time,
                    "model": "Aniemore/WavLM",
                    "device": self.device,
                    "top3_emotions": top3_emotions
                }
                
                logger.info(f"[RAW-VOICE] Сырой результат голосового анализа: {result}")
                return result
                
            finally:
                # Удаляем временный файл
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                        logger.info(f"🗑️ Временный файл удален: {temp_path}")
                    except Exception as e:
                        logger.warning(f"Не удалось удалить временный файл {temp_path}: {e}")
        
        except Exception as e:
            logger.error(f"❌ Ошибка анализа эмоций: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка анализа эмоций: {e}")
    
    async def analyze_text_emotions(self, text: str) -> Dict:
        """Анализ эмоций в тексте"""
        if not self.text_initialized:
            await self.initialize_text_model()
        
        start_time = time.time()
        
        try:
            logger.info(f"📝 Входящий текст для анализа: '{text}'")
            logger.info(f"📏 Длина текста: {len(text)} символов")
            
            # Используем Aniemore TextRecognizer для анализа эмоций
            logger.info("📝 Анализ эмоций с помощью Aniemore TextRecognizer...")
            result = self.text_recognizer.recognize(text, return_single_label=True)
            
            logger.info(f"✅ Результат Aniemore Text: {result}")
            
            # Обрабатываем результат
            if isinstance(result, dict):
                emotion = result.get('label', 'neutral')
                confidence = result.get('score', 0.0)
            else:
                # Если результат не словарь, значит это строка с эмоцией
                # Нужно получить confidence из модели напрямую
                emotion = str(result)
                
                # Получаем confidence из модели напрямую
                try:
                    # Используем text_recognizer для получения полного результата
                    full_result = self.text_recognizer.recognize(text, return_single_label=False)
                    if isinstance(full_result, dict):
                        confidence = full_result.get('score', 0.5)
                        logger.debug(f"Получен confidence из full_result: {confidence}")
                    else:
                        # Если full_result не словарь, попробуем получить confidence из модели напрямую
                        logger.warning(f"full_result не является словарем: {type(full_result)}, значение: {full_result}")
                        # Попробуем альтернативный способ получения confidence
                        try:
                            # Повторный вызов с детальным логированием
                            detailed_result = self.text_recognizer.recognize(text, return_single_label=False)
                            logger.debug(f"Детальный результат: {detailed_result}")
                            if isinstance(detailed_result, dict) and 'score' in detailed_result:
                                confidence = detailed_result['score']
                                logger.info(f"Успешно получен confidence из детального результата: {confidence}")
                            else:
                                # Последняя попытка - используем минимальное значение вместо заглушки
                                confidence = 0.1  # Минимальное значение вместо заглушки 0.5
                                logger.warning(f"Не удалось получить confidence, используем минимальное значение: {confidence}")
                        except Exception as inner_e:
                            logger.error(f"Ошибка при попытке получить детальный результат: {inner_e}")
                            confidence = 0.1  # Минимальное значение вместо заглушки 0.5
                            logger.warning(f"Используем минимальное значение confidence: {confidence}")
                except Exception as e:
                    logger.error(f"Критическая ошибка при получении confidence: {e}")
                    # Вместо заглушки 0.5 используем минимальное значение
                    confidence = 0.1
                    logger.warning(f"Используем минимальное значение confidence из-за ошибки: {confidence}")
            
            # Переводим эмоцию на русский
            russian_emotion = self.emotion_mapping.get(emotion.lower(), emotion)
            
            processing_time = time.time() - start_time
            
            # Получаем топ-3 эмоции из результата recognize
            top3_emotions = []
            try:
                # Получаем полный результат с топ-3 эмоциями
                full_result_with_top3 = self.text_recognizer.recognize(text, return_single_label=False)
                if isinstance(full_result_with_top3, dict) and 'top3_emotions' in full_result_with_top3:
                    top3_emotions = full_result_with_top3['top3_emotions']
                    # Переводим эмоции на русский
                    for emotion_data in top3_emotions:
                        emotion_data['emotion'] = self.emotion_mapping.get(emotion_data['emotion'].lower(), emotion_data['emotion'])
                    logger.info(f"Получены топ-3 эмоции (текст): {top3_emotions}")
                else:
                    logger.warning(f"Не удалось получить топ-3 эмоции из full_result_with_top3 (текст): {full_result_with_top3}")
            except Exception as e:
                logger.error(f"Ошибка при получении топ-3 эмоций (текст): {e}")
                top3_emotions = []
            
            result = {
                "emotion": russian_emotion,
                "confidence": float(confidence),
                "processing_time": processing_time,
                "model": "Aniemore/Bert_Tiny2",
                "device": self.device,
                "top3_emotions": top3_emotions
            }
            
            logger.info(f"[RAW-TEXT-ANIEMORE] Сырой результат текстового анализа Aniemore: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа текста: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка анализа текста: {e}")
    
    async def transcribe_audio(self, audio_data: bytes) -> str:
        """Транскрипция аудио через WhisperCPP"""
        try:
            # WhisperCPP поддерживает base64 формат для передачи аудио данных
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            transcription_request = {
                'audio_data': audio_base64,
                'language': 'ru',
                'model': 'base',
                'translate': False,
                'temperature': 0.0
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeouts["whispercpp"])) as session:
                async with session.post(self.whispercpp_url, json=transcription_request) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('text', '').strip()
                    else:
                        error_text = await response.text()
                        logger.error(f"WhisperCPP error {response.status}: {error_text}")
                        return ""
        
        except asyncio.TimeoutError:
            logger.error(f"WhisperCPP timeout after {self.timeouts['whispercpp']}s")
            return ""
        except Exception as e:
            logger.error(f"WhisperCPP error: {e}")
            return ""
    
    async def analyze_text_dostoevsky(self, text: str) -> Dict:
        """Анализ эмоций в тексте через Dostoevsky"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeouts["dostoevsky"])) as session:
                data = {'text': text}
                
                async with session.post(self.dostoevsky_url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"[RAW-TEXT-DOSTOEVSKY] Сырой результат текстового анализа Dostoevsky: {result}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Dostoevsky error {response.status}: {error_text}")
                        return {"emotion": "нейтральная", "confidence": 0.0, "error": f"HTTP {response.status}"}
        
        except asyncio.TimeoutError:
            logger.error(f"Dostoevsky timeout after {self.timeouts['dostoevsky']}s")
            return {"emotion": "нейтральная", "confidence": 0.0, "error": "timeout"}
        except Exception as e:
            logger.error(f"Dostoevsky error: {e}")
            return {"emotion": "нейтральная", "confidence": 0.0, "error": str(e)}
    
    # Метод combine_emotion_results убран - объединение эмоций теперь происходит в Rust


@app.get("/emotions")
async def get_available_emotions():
    """Получить список доступных эмоций"""
    return {
        "emotions": list(ser_service.emotion_mapping.values()),
        "mapping": ser_service.emotion_mapping
    }

@app.post("/analyze")
async def analyze_voice_emotions(
    audio_file: UploadFile = File(...),
    sample_rate: int = Form(16000)
):
    """Анализ эмоций в голосе"""
    # Проверяем content_type с защитой от None
    if audio_file.content_type is not None and not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Файл должен быть аудио")
    
    try:
        # Читаем аудио данные
        audio_data = await audio_file.read()
        
        # Анализируем эмоции
        result = await ser_service.analyze_emotions(audio_data, sample_rate)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Ошибка обработки файла: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ошибка анализа эмоций: {str(e)}")

@app.post("/analyze-bytes")
async def analyze_voice_emotions_bytes(
    audio_data: bytes = File(...),
    sample_rate: int = Form(16000)
):
    """Анализ эмоций в голосе (байты)"""
    try:
        # Анализируем эмоции
        result = await ser_service.analyze_emotions(audio_data, sample_rate)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Ошибка обработки байтов: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-bytes-json")
async def analyze_voice_emotions_bytes_json(
    request: dict
):
    """Анализ эмоций в голосе (JSON с base64 audio_data)"""
    try:
        # Извлекаем данные из JSON запроса
        audio_b64 = request.get("audio_data", "")
        sample_rate = request.get("sample_rate", 16000)
        
        if not audio_b64:
            raise HTTPException(status_code=400, detail="audio_data не может быть пустым")
        
        # Декодируем base64 в байты
        try:
            audio_data = base64.b64decode(audio_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Ошибка декодирования base64: {str(e)}")
        
        # Анализируем эмоции
        result = await ser_service.analyze_emotions(audio_data, sample_rate)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обработки JSON байтов: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-text")
async def analyze_text_emotions(
    request: dict
):
    """Анализ эмоций в тексте"""
    text = request.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    try:
        # Анализируем эмоции в тексте
        result = await ser_service.analyze_text_emotions(text)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Ошибка обработки текста: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ошибка анализа текста: {str(e)}")

# Endpoint /analyze-complete убран - полный анализ эмоций теперь происходит в Rust

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        logger.info("[HEALTH] Проверка состояния сервиса...")
        
        # Определяем статус сервиса
        if ser_service.is_initialized and ser_service.text_initialized:
            service_status = "healthy"
        elif ser_service.is_initialized or ser_service.text_initialized:
            service_status = "degraded"
        else:
            service_status = "unhealthy"
        
        status = {
            "status": service_status,
            "service": "Aniemore SER Service",
            "version": "1.0.0",
            "initialized": ser_service.is_initialized,
            "text_initialized": ser_service.text_initialized,
            "device": ser_service.device,
            "voice_model": ser_service.model_name,
            "text_model": ser_service.text_model_name,
            "use_local_models": ser_service.use_local_models
        }
        
        logger.info(f"[HEALTH] Статус: {status}")
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"[HEALTH] Ошибка health check: {e}")
        import traceback
        logger.error(f"[HEALTH] Traceback: {traceback.format_exc()}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "service": "Aniemore SER Service"
            },
            status_code=500
        )

@app.get("/model-info")
async def get_model_info():
    """Информация о модели"""
    try:
        logger.info("[MODEL-INFO] Запрос информации о модели...")
        
        if not ser_service.is_initialized:
            return {
                "status": "not_initialized",
                "message": "Модель не инициализирована",
                "device": ser_service.device,
                "voice_model": ser_service.model_name,
                "text_model": ser_service.text_model_name
            }
        
        return {
            "status": "initialized",
            "model_name": ser_service.model_name,
            "device": ser_service.device,
            "emotion_mapping": ser_service.emotion_mapping,
            "voice_model": ser_service.model_name,
            "text_model": ser_service.text_model_name,
            "model_labels": ser_service.model.config.id2label if ser_service.model and hasattr(ser_service.model, 'config') else None
        }
        
    except Exception as e:
        logger.error(f"[MODEL-INFO] Ошибка: {e}")
        import traceback
        logger.error(f"[MODEL-INFO] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# Глобальный экземпляр сервиса
ser_service = AniemoreSERService()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aniemore SER Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8006, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    logger.info(f"🎭 Запуск Aniemore SER Service на {args.host}:{args.port}")
    
    uvicorn.run(
        "aniemore_ser_service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False
    )
