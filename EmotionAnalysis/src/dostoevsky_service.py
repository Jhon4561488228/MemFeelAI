#!/usr/bin/env python3
"""
📝 Dostoevsky Text Emotion Service
HTTP сервис для анализа эмоций в тексте с помощью Dostoevsky
"""

import asyncio
import logging
import sys
import time
from typing import Dict, List, Optional
import json

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("[INFO] Запуск Dostoevsky Text Emotion Service...")
    await dostoevsky_service.initialize()
    logger.info("[OK] Dostoevsky Text Emotion Service готов к работе")
    yield
    # Shutdown
    logger.info("[INFO] Dostoevsky Text Emotion Service завершает работу...")
    if hasattr(app.state, 'sentiment_analyzer') and app.state.sentiment_analyzer:
        logger.info("[INFO] Очистка ресурсов Dostoevsky...")
        # Очистка ресурсов Dostoevsky (если нужно)
    logger.info("[INFO] Dostoevsky Text Emotion Service завершен")

app = FastAPI(
    title="Dostoevsky Text Emotion Service",
    description="Сервис анализа эмоций в тексте с помощью Dostoevsky",
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
            "service": "Dostoevsky Text Emotion Service",
            "version": "1.0.0",
            "status": "running",
            "endpoints": [
                "/health",
                "/model-info",
                "/emotions",
                "/analyze",
                "/analyze-json"
            ]
        }
    except Exception as e:
        logger.error(f"[ROOT] Ошибка: {e}")
        import traceback
        logger.error(f"[ROOT] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

class DostoevskyService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
        # Маппинг эмоций на русский язык
        self.emotion_mapping = {
            'joy': 'радость',
            'sadness': 'грусть',
            'anger': 'злость',
            'fear': 'страх',
            'surprise': 'удивление',
            'neutral': 'нейтральная',
            'positive': 'радость',
            'negative': 'грусть',
            'speech': 'нейтральная',
            'skip': 'нейтральная'
        }
    
    async def initialize(self):
        """Инициализация модели Dostoevsky"""
        if self.is_initialized:
            return
        
        logger.info("📝 Инициализация Dostoevsky модели...")
        
        try:
            # Импортируем Dostoevsky
            from dostoevsky.tokenization import RegexTokenizer
            from dostoevsky.models import FastTextSocialNetworkModel
            import os
            
            # Инициализируем tokenizer
            self.tokenizer = RegexTokenizer()
            
            # Используем локальную модель Dostoevsky
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'dostoevsky', 'fasttext-social-network-model.bin')
            logger.info(f"[INFO] Проверяем путь к модели: {model_path}")
            logger.info(f"[INFO] Файл существует: {os.path.exists(model_path)}")
            
            if os.path.exists(model_path):
                logger.info(f"[INFO] Используем локальную модель Dostoevsky: {model_path}")
                # Устанавливаем переменную окружения для Dostoevsky
                os.environ['DOSTOEVSKY_DATA_PATH'] = os.path.dirname(model_path)
                logger.info(f"[INFO] DOSTOEVSKY_DATA_PATH установлен: {os.environ.get('DOSTOEVSKY_DATA_PATH')}")
                self.model = FastTextSocialNetworkModel(tokenizer=self.tokenizer)
            else:
                logger.info("[INFO] Используем стандартную модель Dostoevsky")
                self.model = FastTextSocialNetworkModel(tokenizer=self.tokenizer)
            
            logger.info("[OK] Dostoevsky модель загружена")
            self.is_initialized = True
            
        except ImportError as e:
            logger.error(f"[ERROR] Ошибка импорта Dostoevsky: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Dostoevsky не установлен. Установите: pip install dostoevsky"
            )
        except Exception as e:
            logger.error(f"[ERROR] Ошибка инициализации модели: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка инициализации модели: {e}")
    
    async def analyze_emotions(self, text: str) -> Dict:
        """Анализ эмоций в тексте"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            logger.info(f"[INFO] Входящий текст для анализа: '{text}'")
            logger.info(f"[INFO] Длина текста: {len(text)} символов")
            
            # Проверяем, что модель инициализирована
            if self.model is None:
                raise Exception("Модель не инициализирована")
            
            logger.info(f"🔧 Модель готова к анализу: {type(self.model)}")
            
            # Анализируем текст
            results = self.model.predict([text], k=5)  # Получаем топ-5 эмоций
            logger.info(f"🔍 Результаты модели: {results}")
            
            # Получаем эмоции
            emotion_scores = results[0]
            logger.info(f"📊 Сырые оценки эмоций: {emotion_scores}")
            
            # Находим доминирующую эмоцию
            if emotion_scores:
                dominant_emotion_key = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
                dominant_emotion = self.emotion_mapping.get(dominant_emotion_key, 'нейтральная')
                confidence = float(emotion_scores[dominant_emotion_key])
                logger.info(f"🎯 Доминирующая эмоция: {dominant_emotion} (ключ: {dominant_emotion_key}, уверенность: {confidence:.3f})")
            else:
                 # Если нет результатов, пытаемся получить минимальную уверенность из модели
                try:
                    # Пытаемся получить минимальную уверенность из модели
                    if hasattr(self.model, 'predict_proba') and text.strip():
                        # Получаем вероятности для всех классов
                        probas = self.model.predict_proba([text])
                        if len(probas) > 0 and len(probas[0]) > 0:
                            # Берем максимальную вероятность как confidence
                            max_proba = float(max(probas[0]))
                            confidence = max_proba if max_proba > 0.1 else 0.1
                            dominant_emotion = 'нейтральная'
                            logger.info(f"🎯 Получена минимальная уверенность из модели: {confidence:.3f}")
                        else:
                            confidence = 0.1
                            dominant_emotion = 'нейтральная'
                            logger.warning("⚠️ Модель не вернула вероятности, используем минимальное значение")
                    else:
                        confidence = 0.1
                        dominant_emotion = 'нейтральная'
                        logger.warning("⚠️ Модель недоступна, используем минимальное значение")
                except Exception as e:
                    confidence = 0.1
                    dominant_emotion = 'нейтральная'
                    logger.error(f"❌ Ошибка получения confidence из модели: {e}, используем минимальное значение")
            
            # Создаем словарь всех эмоций с русскими названиями
            all_emotions = {}
            for key, score in emotion_scores.items():
                emotion = self.emotion_mapping.get(key, 'нейтральная')
                # Если эмоция уже существует, берем максимальный score
                if emotion in all_emotions:
                    all_emotions[emotion] = max(all_emotions[emotion], float(score))
                else:
                    all_emotions[emotion] = float(score)
            
            # Убеждаемся, что у нас есть нейтральная эмоция
            if 'нейтральная' not in all_emotions:
                all_emotions['нейтральная'] = 0.1
            
            # Нормализуем вероятности
            total = sum(all_emotions.values())
            if total > 0:
                all_emotions = {k: v/total for k, v in all_emotions.items()}
            
            processing_time = time.time() - start_time
            
            result = {
                "emotion": dominant_emotion,
                "confidence": confidence,
                "all_emotions": all_emotions,
                "processing_time": processing_time,
                "text_length": len(text),
                "model_info": {
                    "name": "FastTextSocialNetworkModel",
                    "tokenizer": "RegexTokenizer"
                }
            }
            
            logger.info(f"✅ Анализ завершен за {processing_time:.3f}с")
            logger.info(f"📤 Исходящий результат: {result}")
            logger.info(f"🎯 Финальная эмоция: {dominant_emotion} (уверенность: {confidence:.3f})")
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"❌ Ошибка анализа эмоций: {e}")
            logger.error(f"📋 Детали ошибки: {error_details}")
            raise HTTPException(status_code=500, detail=f"Ошибка анализа эмоций: {str(e)}")

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "service": "dostoevsky-text-emotion",
        "initialized": dostoevsky_service.is_initialized,
        "model": "FastTextSocialNetworkModel"
    }

@app.get("/emotions")
async def get_available_emotions():
    """Получить список доступных эмоций"""
    return {
        "emotions": list(dostoevsky_service.emotion_mapping.values()),
        "mapping": dostoevsky_service.emotion_mapping
    }

@app.post("/analyze")
async def analyze_text_emotions(
    text: str = Form(...)
):
    """Анализ эмоций в тексте"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    try:
        # Анализируем эмоции
        result = await dostoevsky_service.analyze_emotions(text)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Ошибка обработки текста: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-json")
async def analyze_text_emotions_json(data: dict):
    """Анализ эмоций в тексте (JSON)"""
    text = data.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    try:
        # Проверяем инициализацию
        if not dostoevsky_service.is_initialized:
            await dostoevsky_service.initialize()
        
        # Анализируем эмоции
        result = await dostoevsky_service.analyze_emotions(text)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Ошибка обработки JSON: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Информация о модели"""
    if not dostoevsky_service.is_initialized:
        raise HTTPException(status_code=503, detail="Модель не инициализирована")
    
    return {
        "model_name": "FastTextSocialNetworkModel",
        "tokenizer": "RegexTokenizer",
        "emotion_mapping": dostoevsky_service.emotion_mapping,
        "supported_languages": ["ru", "en"]
    }

# Глобальный экземпляр сервиса
dostoevsky_service = DostoevskyService()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dostoevsky Text Emotion Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8007, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    logger.info(f"📝 Запуск Dostoevsky Text Emotion Service на {args.host}:{args.port}")
    
    uvicorn.run(
        "dostoevsky_service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False
    )
