"""
🎭 Сервис анализа эмоций для Memory System
Полноценная интеграция с Aniemore и Dostoevsky
Улучшенная логика комбинирования эмоций из Rust emotion-engine
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
import os

# Импортируем улучшенный комбинер эмоций
from .emotion_combiner import emotion_combiner, VoiceEmotionResult, TextEmotionResult, CombinedEmotionResult

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmotionAnalysisResult:
    """Результат анализа эмоций"""
    primary_emotion: str
    primary_confidence: float
    secondary_emotion: Optional[str] = None
    secondary_confidence: Optional[float] = None
    tertiary_emotion: Optional[str] = None
    tertiary_confidence: Optional[float] = None
    consistency: str = "high"
    dominant_source: str = "text"
    validation_applied: bool = False
    analysis_time: float = 0.0
    raw_results: Dict[str, Any] = None
    # Дополнительно: итоговая тональность текста (Dostoevsky)
    sentiment: Optional[str] = None

class EmotionAnalysisService:
    """Полноценный сервис анализа эмоций"""
    
    def __init__(self):
        # URLs сервисов
        # Базовый URL Aniemore SER (см. Swagger на 8006)
        self.base_url = os.getenv("ANIEMORE_BASE_URL", "http://localhost:8006")
        self.aniemore_text = f"{self.base_url}/analyze-text"
        self.aniemore_voice = f"{self.base_url}/analyze"          # multipart: file
        self.aniemore_voice_bytes = f"{self.base_url}/analyze-bytes"  # multipart: file
        self.aniemore_voice_bytes_json = f"{self.base_url}/analyze-bytes-json"  # JSON: {audio_b64}
        self.dostoevsky_url = os.getenv("DOSTOEVSKY_BASE_URL", "http://localhost:8007") + "/analyze"
        self.timeout = int(os.getenv("EMOTION_TIMEOUT_SEC", "30"))
        
        # Веса для источников
        self.source_weights = {
            "aniemore": 0.6,  # Aniemore более точный для текста
            "dostoevsky": 0.4  # Dostoevsky как валидация
        }
        
        # Маппинг эмоций к категориям
        self.emotion_categories = {
            "радость": "positive", "энтузиазм": "positive", "удивление": "positive",
            "грусть": "negative", "злость": "negative", "страх": "negative", "отвращение": "negative",
            "нейтральная": "neutral", "спокойствие": "neutral"
        }
        
        logger.info("EmotionAnalysisService инициализирован")
    
    async def analyze_text_emotions(self, text: str, include_validation: bool = True) -> EmotionAnalysisResult:
        """Полноценный анализ эмоций в тексте"""
        
        start_time = time.time()
        # Безопасное логирование с правильной кодировкой
        try:
            safe_text = text[:50].encode('utf-8', errors='replace').decode('utf-8')
            logger.info(f"Начинаем анализ эмоций для текста: '{safe_text}{'...' if len(text) > 50 else ''}'")
        except Exception:
            logger.info(f"Начинаем анализ эмоций для текста: [encoding error] {len(text)} chars")
        
        try:
            # Параллельный анализ через Aniemore и Dostoevsky
            aniemore_task = self._analyze_text_with_aniemore(text)
            dostoevsky_task = self._analyze_with_dostoevsky(text) if include_validation else None
            
            # Ждем результаты
            if dostoevsky_task:
                aniemore_result, dostoevsky_result = await asyncio.gather(
                    aniemore_task, dostoevsky_task, return_exceptions=True
                )
            else:
                aniemore_result = await aniemore_task
                dostoevsky_result = None
            
            # Обрабатываем исключения
            if isinstance(aniemore_result, Exception):
                logger.error(f"Ошибка анализа Aniemore: {aniemore_result}")
                aniemore_result = {"emotion": "нейтральная", "confidence": 0.0, "error": str(aniemore_result)}
            
            if dostoevsky_result and isinstance(dostoevsky_result, Exception):
                logger.error(f"Ошибка анализа Dostoevsky: {dostoevsky_result}")
                dostoevsky_result = {"emotion": "нейтральная", "confidence": 0.0, "error": str(dostoevsky_result)}
            
            # Объединяем результаты
            final_result = self._combine_emotion_results(aniemore_result, None, dostoevsky_result, include_validation)
            
            analysis_time = time.time() - start_time
            final_result.analysis_time = analysis_time
            
            logger.info(f"Анализ эмоций завершен за {analysis_time:.3f}с")
            logger.info(f"Результат: {final_result.primary_emotion} ({final_result.primary_confidence:.3f})")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Критическая ошибка анализа эмоций: {e}")
            # Возвращаем безопасный результат
            return EmotionAnalysisResult(
                primary_emotion="нейтральная",
                primary_confidence=0.0,
                consistency="unknown",
                dominant_source="error",
                validation_applied=False,
                analysis_time=time.time() - start_time,
                raw_results={"error": str(e)}
            )
    
    async def _analyze_text_with_aniemore(self, text: str) -> Dict[str, Any]:
        """Анализ эмоций в тексте через Aniemore.
        Сначала пытаемся JSON {"text": ...}, при неуспехе пробуем form-data."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                # Попытка 1: JSON тело
                payload = {"text": text}
                async with session.post(self.aniemore_text, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Aniemore text(JSON): {result}")
                        return result
                    # Фоллбек на form-data при 415/422/400
                    if response.status in (400, 415, 422):
                        error_text = await response.text()
                        # Безопасное логирование с правильной кодировкой
                        try:
                            safe_error = error_text.encode('utf-8', errors='replace').decode('utf-8')
                            logger.warning(f"Aniemore text JSON {response.status}, fallback to form-data: {safe_error}")
                        except Exception:
                            logger.warning(f"Aniemore text JSON {response.status}, fallback to form-data: [encoding error]")
                        form = {"text": text}
                        async with session.post(self.aniemore_text, data=form) as r2:
                            if r2.status == 200:
                                result = await r2.json()
                                logger.info(f"Aniemore text(form): {result}")
                                return result
                            err2 = await r2.text()
                            # Безопасное логирование с правильной кодировкой
                            try:
                                safe_err2 = err2.encode('utf-8', errors='replace').decode('utf-8')
                                logger.error(f"Aniemore text form error {r2.status}: {safe_err2}")
                            except Exception:
                                logger.error(f"Aniemore text form error {r2.status}: [encoding error]")
                            return {"emotion": "нейтральная", "confidence": 0.0, "error": f"HTTP {r2.status}: {err2}"}
                    # Иные коды
                    error_text = await response.text()
                    # Безопасное логирование с правильной кодировкой
                    try:
                        safe_error = error_text.encode('utf-8', errors='replace').decode('utf-8')
                        logger.error(f"Aniemore text error {response.status}: {safe_error}")
                    except Exception:
                        logger.error(f"Aniemore text error {response.status}: [encoding error]")
                    return {"emotion": "нейтральная", "confidence": 0.0, "error": f"HTTP {response.status}: {error_text}"}
        except asyncio.TimeoutError:
            logger.error(f"Aniemore text timeout after {self.timeout}s")
            return {"emotion": "нейтральная", "confidence": 0.0, "error": "timeout"}
        except Exception as e:
            logger.error(f"Aniemore text error: {e}")
            return {"emotion": "нейтральная", "confidence": 0.0, "error": str(e)}

    async def _analyze_voice_with_aniemore_bytes(self, audio_b64: str) -> Dict[str, Any]:
        """Анализ эмоций в голосе через Aniemore (JSON: audio_b64)."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                # По OpenAPI у /analyze-bytes-json требуется поле 'audio_data' и (опц.) 'sample_rate'
                sample_rate = self._detect_wav_sample_rate(audio_b64) or 44100
                payload = {"audio_data": audio_b64, "sample_rate": int(sample_rate)}
                headers = {"Content-Type": "application/json"}
                async with session.post(self.aniemore_voice_bytes_json, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Aniemore voice-bytes-json: {result}")
                        return result
                    # Фоллбеки: старый multipart endpoint и form-urlencoded
                    error_text = await response.text()
                    # Безопасное логирование с правильной кодировкой
                    try:
                        safe_error = error_text.encode('utf-8', errors='replace').decode('utf-8')
                        logger.warning(f"Aniemore voice-bytes-json {response.status}: {safe_error}. Trying fallbacks...")
                    except Exception:
                        logger.warning(f"Aniemore voice-bytes-json {response.status}: [encoding error]. Trying fallbacks...")
                    # 1) form-urlencoded: audio_data=...
                    form_data = {"audio_data": audio_b64, "sample_rate": str(sample_rate)}
                    async with session.post(self.aniemore_voice_bytes, data=form_data) as r2:
                        if r2.status == 200:
                            result = await r2.json()
                            logger.info(f"Aniemore voice-bytes(form): {result}")
                            return result
                    # 2) multipart file upload to /analyze
                    try:
                        import base64 as _b64
                        audio_bytes = _b64.b64decode(audio_b64)
                        form = aiohttp.FormData()
                        # Aniemore expects 'audio_file' field name
                        form.add_field('audio_file', audio_bytes, filename='audio.wav', content_type='audio/wav')
                        async with session.post(self.aniemore_voice, data=form) as r3:
                            if r3.status == 200:
                                result = await r3.json()
                                logger.info(f"Aniemore voice(form-file): {result}")
                                return result
                            err3 = await r3.text()
                            logger.error(f"Aniemore voice(form-file) error {r3.status}: {err3}")
                    except Exception as _e:
                        logger.error(f"Aniemore voice fallback(build multipart) error: {_e}")
                    return {"emotion": "нейтральная", "confidence": 0.0, "error": f"HTTP {response.status}: {error_text}"}
        except Exception as e:
            logger.error(f"Aniemore voice-bytes error: {e}")
            return {"emotion": "нейтральная", "confidence": 0.0, "error": str(e)}

    def _detect_wav_sample_rate(self, audio_b64: str) -> Optional[int]:
        """Пытается определить sample rate из WAV-заголовка, если это WAV. Возвращает None при неудаче."""
        try:
            import io, wave, base64 as _b64
            raw = _b64.b64decode(audio_b64)
            with wave.open(io.BytesIO(raw), 'rb') as wf:
                return int(wf.getframerate())
        except Exception:
            return None

    async def _analyze_voice_with_aniemore_file(self, wav_path: str) -> Dict[str, Any]:
        """Анализ эмоций в голосе через Aniemore (multipart file)."""
        try:
            import aiofiles
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                form = aiohttp.FormData()
                async with aiofiles.open(wav_path, mode='rb') as f:
                    content = await f.read()
                # Aniemore expects 'audio_file' field name
                form.add_field('audio_file', content, filename='audio.wav', content_type='audio/wav')
                async with session.post(self.aniemore_voice, data=form) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Aniemore voice: {result}")
                        return result
                    else:
                        error_text = await response.text()
                        # Безопасное логирование с правильной кодировкой
                        try:
                            safe_error = error_text.encode('utf-8', errors='replace').decode('utf-8')
                            logger.error(f"Aniemore voice error {response.status}: {safe_error}")
                        except Exception:
                            logger.error(f"Aniemore voice error {response.status}: [encoding error]")
                        return {"emotion": "нейтральная", "confidence": 0.0, "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Aniemore voice error: {e}")
            return {"emotion": "нейтральная", "confidence": 0.0, "error": str(e)}

    async def analyze_voice_emotions(self, *, audio_b64: Optional[str] = None, wav_path: Optional[str] = None) -> Dict[str, Any]:
        """Публичный метод для голосового анализа. Возвращает ответ Aniemore."""
        if audio_b64:
            return await self._analyze_voice_with_aniemore_bytes(audio_b64)
        if wav_path:
            return await self._analyze_voice_with_aniemore_file(wav_path)
        raise ValueError("audio_b64 or wav_path required")
    
    async def _analyze_with_dostoevsky(self, text: str) -> Dict[str, Any]:
        """Анализ эмоций через Dostoevsky"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                data = {'text': text}
                
                async with session.post(self.dostoevsky_url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Dostoevsky результат: {result}")
                        return result
                    else:
                        error_text = await response.text()
                        # Безопасное логирование с правильной кодировкой
                        try:
                            safe_error = error_text.encode('utf-8', errors='replace').decode('utf-8')
                            logger.error(f"Dostoevsky error {response.status}: {safe_error}")
                        except Exception:
                            logger.error(f"Dostoevsky error {response.status}: [encoding error]")
                        return {"emotion": "нейтральная", "confidence": 0.0, "error": f"HTTP {response.status}"}
        
        except asyncio.TimeoutError:
            logger.error(f"Dostoevsky timeout after {self.timeout}s")
            return {"emotion": "нейтральная", "confidence": 0.0, "error": "timeout"}
        except Exception as e:
            logger.error(f"Dostoevsky error: {e}")
            return {"emotion": "нейтральная", "confidence": 0.0, "error": str(e)}
    
    def _normalize_top3(self, res: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Нормализует ответ Aniemore в топ-3 (label, confidence)."""
        if not res:
            return [("нейтральная", 0.0)]
        if "top3" in res and isinstance(res["top3"], list):
            pairs = []
            for item in res["top3"][:3]:
                if isinstance(item, dict) and "label" in item and "confidence" in item:
                    pairs.append((item["label"], float(item["confidence"])) )
            if pairs:
                return pairs
        # support Aniemore "top3_emotions" format as in logs
        if "top3_emotions" in res and isinstance(res["top3_emotions"], list):
            pairs = []
            for item in res["top3_emotions"][:3]:
                label_key = "label" if "label" in item else ("emotion" if "emotion" in item else None)
                if isinstance(item, dict) and label_key and "confidence" in item:
                    pairs.append((item[label_key], float(item["confidence"])) )
            if pairs:
                return pairs
        # fallback: single
        return [(res.get("emotion", "нейтральная"), float(res.get("confidence", 0.0)))]

    def _label_category(self, label: str) -> str:
        return self.emotion_categories.get(label, "neutral")

    def _build_conflict_and_guidance(self, voice_top: List[Tuple[str, float]], text_top: List[Tuple[str, float]]) -> Tuple[Dict[str, Any], str]:
        v1 = (voice_top[0][0], float(voice_top[0][1])) if voice_top else ("нейтральная", 0.0)
        t1 = (text_top[0][0], float(text_top[0][1])) if text_top else ("нейтральная", 0.0)
        v_cat = self._label_category(v1[0])
        t_cat = self._label_category(t1[0])
        if v_cat != t_cat and (v1[1] >= 0.5 or t1[1] >= 0.5):
            level = "strong"
            reason = f"voice: {v1[0]} vs text: {t1[0]}"
            guidance = "Отвечать мягко: признать возможное скрытое переживание, поддержать, уточнить причину."
        elif v_cat != t_cat:
            level = "mild"
            reason = f"voice: {v1[0]} vs text: {t1[0]}"
            guidance = "Нейтрально‑поддерживающий тон, уточнить детали."
        else:
            level = "none"
            reason = "aligned"
            guidance = "Стандартный дружелюбный тон."
        return ({"level": level, "reason": reason}, guidance)

    def _combine_emotion_results(
        self,
        text_result: Optional[Dict[str, Any]],
        voice_result: Optional[Dict[str, Any]],
        dostoevsky_result: Optional[Dict[str, Any]],
        validation_applied: bool,
    ) -> EmotionAnalysisResult:
        """Сведение голос+текст в топ‑3 с учётом веса и валидации."""
        voice_top = self._normalize_top3(voice_result or {})
        text_top = self._normalize_top3(text_result or {})

        # динамические веса источников
        if voice_result:
            voice_conf_max = max((c for _, c in voice_top), default=0.0)
            text_conf_max = max((c for _, c in text_top), default=0.0)
            # базовые веса
            base_voice, base_text = 0.6, 0.4
            # если голос нейтрален и текст выраженный — усиливаем текст
            voice_primary_label = (voice_top[0][0] if voice_top else None) or "нейтральная"
            text_primary_label = (text_top[0][0] if text_top else None) or "нейтральная"
            if voice_primary_label == "нейтральная" and text_primary_label != "нейтральная" and text_conf_max >= 0.5:
                base_voice, base_text = 0.3, 0.7
            # масштабируем по уверенности
            w_voice = base_voice * (0.5 + 0.5 * voice_conf_max)
            w_text = base_text * (0.5 + 0.5 * text_conf_max)
            # нормализуем
            s = (w_voice + w_text) or 1.0
            w_voice, w_text = w_voice / s, w_text / s
        else:
            w_voice, w_text = 0.0, 1.0

        # агрегируем веса по лейблам
        score: Dict[str, float] = {}
        for label, conf in voice_top:
            score[label] = score.get(label, 0.0) + conf * w_voice
        for label, conf in text_top:
            score[label] = score.get(label, 0.0) + conf * w_text

        # нормируем и выбираем топ‑3
        items = sorted(score.items(), key=lambda x: x[1], reverse=True)[:3]
        if not items:
            items = [("нейтральная", 0.0)]

        # sentiment от Dostoevsky
        sentiment = None
        if dostoevsky_result and isinstance(dostoevsky_result, dict):
            # поддержка англ и русских ключей
            if any(k in dostoevsky_result for k in ("positive", "neutral", "negative")):
                cand = {k: float(v) for k, v in dostoevsky_result.items() if k in ("positive", "neutral", "negative")}
                if cand:
                    sentiment = max(cand.items(), key=lambda x: x[1])[0]
            elif "all_emotions" in dostoevsky_result and isinstance(dostoevsky_result["all_emotions"], dict):
                ae = dostoevsky_result["all_emotions"]
                # маппинг русских эмоций в обобщённый тег
                ru_to_sent = {
                    "радость": "positive",
                    "счастье": "positive",
                    "удовольствие": "positive",
                    "нейтральная": "neutral",
                    "спокойствие": "neutral",
                    "грусть": "negative",
                    "злость": "negative",
                    "страх": "negative",
                    "отвращение": "negative",
                }
                best = None
                best_v = -1.0
                for k, v in ae.items():
                    try:
                        vv = float(v)
                    except Exception:
                        continue
                    if vv > best_v:
                        best, best_v = k, vv
                if best:
                    sentiment = ru_to_sent.get(best, None)

        # consistency по категориям
        def cat(e: str) -> str:
            return self.emotion_categories.get(e, "neutral")
        categories = {cat(lbl) for lbl, _ in items}
        if "positive" in categories and "negative" in categories:
            consistency = "low"
        elif len(categories) > 1:
            consistency = "medium"
        else:
            consistency = "high"

        # dominant_source
        if voice_result and text_result:
            dominant_source = "mixed"
        elif voice_result:
            dominant_source = "voice"
        else:
            dominant_source = "text"

        # формируем результат (primary/secondary/tertiary)
        p = items + [(None, None)] * (3 - len(items))
        primary_emotion, primary_confidence = p[0]
        secondary_emotion, secondary_confidence = p[1]
        tertiary_emotion, tertiary_confidence = p[2]

        # sentiment принудительно равен категории основной эмоции
        sentiment = self.emotion_categories.get(primary_emotion or "нейтральная", "neutral")

        return EmotionAnalysisResult(
            primary_emotion=primary_emotion or "нейтральная",
            primary_confidence=float(primary_confidence or 0.0),
            secondary_emotion=secondary_emotion,
            secondary_confidence=float(secondary_confidence or 0.0) if secondary_confidence is not None else None,
            tertiary_emotion=tertiary_emotion,
            tertiary_confidence=float(tertiary_confidence or 0.0) if tertiary_confidence is not None else None,
            consistency=consistency,
            dominant_source=dominant_source,
            validation_applied=validation_applied and dostoevsky_result is not None,
            raw_results={"voice": voice_result, "text": text_result, "dostoevsky": dostoevsky_result},
            sentiment=sentiment,
        )
        
        # Извлекаем данные Aniemore
        aniemore_emotion = aniemore_result.get('emotion', 'нейтральная')
        aniemore_confidence = aniemore_result.get('confidence', 0.0)
        
        # Если нет Dostoevsky результата, используем только Aniemore
        if not dostoevsky_result or 'error' in dostoevsky_result:
            return EmotionAnalysisResult(
                primary_emotion=aniemore_emotion,
                primary_confidence=aniemore_confidence,
                consistency="high",
                dominant_source="aniemore",
                validation_applied=False,
                raw_results={
                    "aniemore": aniemore_result,
                    "dostoevsky": dostoevsky_result
                }
            )
        
        # Извлекаем данные Dostoevsky
        dostoevsky_emotion = dostoevsky_result.get('emotion', 'нейтральная')
        dostoevsky_confidence = dostoevsky_result.get('confidence', 0.0)
        
        # Определяем категории эмоций
        aniemore_category = self.emotion_categories.get(aniemore_emotion, "neutral")
        dostoevsky_category = self.emotion_categories.get(dostoevsky_emotion, "neutral")
        
        # Вычисляем согласованность
        if aniemore_category == dostoevsky_category:
            consistency = "high"
        elif (aniemore_category == "positive" and dostoevsky_category == "negative") or \
             (aniemore_category == "negative" and dostoevsky_category == "positive"):
            consistency = "low"
        else:
            consistency = "medium"
        
        # Принимаем решение о финальной эмоции
        if consistency == "high":
            # Высокая согласованность - выбираем более уверенный результат
            if aniemore_confidence >= dostoevsky_confidence:
                final_emotion = aniemore_emotion
                final_confidence = aniemore_confidence
                dominant_source = "aniemore"
            else:
                final_emotion = dostoevsky_emotion
                final_confidence = dostoevsky_confidence
                dominant_source = "dostoevsky"
        else:
            # Низкая согласованность - приоритет Aniemore, но снижаем уверенность
            final_emotion = aniemore_emotion
            final_confidence = aniemore_confidence * 0.8  # Снижаем уверенность
            dominant_source = "aniemore"
        
        # Определяем вторичную эмоцию
        secondary_emotion = None
        secondary_confidence = None
        
        if consistency != "high" and dostoevsky_confidence > 0.3:
            secondary_emotion = dostoevsky_emotion
            secondary_confidence = dostoevsky_confidence
        
        return EmotionAnalysisResult(
            primary_emotion=final_emotion,
            primary_confidence=final_confidence,
            secondary_emotion=secondary_emotion,
            secondary_confidence=secondary_confidence,
            consistency=consistency,
            dominant_source=dominant_source,
            validation_applied=validation_applied and dostoevsky_result is not None,
            raw_results={
                "aniemore": aniemore_result,
                "dostoevsky": dostoevsky_result
            }
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья сервисов"""
        health_status = {
            "aniemore": False,
            "dostoevsky": False,
            "overall": False
        }
        
        try:
            # Проверяем Aniemore
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get("http://localhost:8006/health") as response:
                    health_status["aniemore"] = response.status == 200
        except:
            health_status["aniemore"] = False
        
        try:
            # Проверяем Dostoevsky
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get("http://localhost:8007/health") as response:
                    health_status["dostoevsky"] = response.status == 200
        except:
            health_status["dostoevsky"] = False
        
        health_status["overall"] = health_status["aniemore"]  # Aniemore критичен
        
        return health_status
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Получение статистики сервиса"""
        return {
            "service_name": "EmotionAnalysisService",
            "aniemore_endpoints": {
                "text": self.aniemore_text,
                "voice": self.aniemore_voice,
                "voice_bytes": self.aniemore_voice_bytes,
                "voice_bytes_json": self.aniemore_voice_bytes_json,
            },
            "dostoevsky_url": self.dostoevsky_url,
            "timeout": self.timeout,
            "source_weights": self.source_weights,
            "emotion_categories": self.emotion_categories
        }
    
    async def analyze_emotions_enhanced(
        self,
        text: Optional[str] = None,
        audio_b64: Optional[str] = None,
        include_validation: bool = True
    ) -> CombinedEmotionResult:
        """
        Улучшенный анализ эмоций с логикой из Rust emotion-engine
        
        Args:
            text: Текст для анализа
            audio_b64: Аудио данные в base64
            include_validation: Включать ли валидацию тональности
            
        Returns:
            CombinedEmotionResult с улучшенной логикой комбинирования
        """
        logger.info("Starting enhanced emotion analysis with Rust emotion-engine logic")
        
        # Анализ голосовых эмоций
        voice_result = None
        if audio_b64:
            try:
                voice_data = await self.analyze_voice_emotions(audio_b64=audio_b64)
                if voice_data and voice_data.get("emotion"):
                    voice_result = VoiceEmotionResult(
                        emotion=voice_data["emotion"],
                        confidence=voice_data.get("confidence", 0.5),
                        source="voice"
                    )
            except Exception as e:
                logger.warning(f"Voice emotion analysis failed: {e}")
        
        # Анализ текстовых эмоций
        text_result = None
        if text:
            try:
                # Получаем результаты от Aniemore и Dostoevsky
                aniemore_result = await self._analyze_text_with_aniemore(text)
                dostoevsky_result = await self._analyze_with_dostoevsky(text)
                
                if aniemore_result and dostoevsky_result:
                    # Создаем TextEmotionResult с данными от обеих моделей
                    text_result = TextEmotionResult(
                        primary_emotion=aniemore_result.get("emotion", "нейтральная"),
                        primary_confidence=aniemore_result.get("confidence", 0.5),
                        secondary_emotion=dostoevsky_result.get("sentiment", "neutral"),
                        secondary_confidence=dostoevsky_result.get("confidence", 0.5),
                        consistency="medium",  # Будет пересчитано в комбинере
                        source="text_combined"
                    )
            except Exception as e:
                logger.warning(f"Text emotion analysis failed: {e}")
        
        # Используем улучшенный комбинер эмоций
        combined_result = emotion_combiner.combine_emotions_final(
            voice_result=voice_result,
            text_result=text_result,
            transcription=text or ""
        )
        
        logger.info(f"Enhanced emotion analysis completed: {combined_result.emotion} (confidence: {combined_result.confidence:.3f})")
        return combined_result

# Глобальный экземпляр сервиса
emotion_service: Optional[EmotionAnalysisService] = None

def get_emotion_service() -> EmotionAnalysisService:
    """Получение экземпляра сервиса анализа эмоций"""
    global emotion_service
    if emotion_service is None:
        emotion_service = EmotionAnalysisService()
    return emotion_service
