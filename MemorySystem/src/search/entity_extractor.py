"""
Entity Extractor для AIRI Memory System
Автоматическое извлечение сущностей и связей из текста
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from loguru import logger

from ..providers.lm_studio_provider import LMStudioProvider

@dataclass
class ExtractedEntity:
    """Извлеченная сущность"""
    name: str
    entity_type: str  # person, organization, technology, concept, etc.
    confidence: float
    context: str
    start_pos: int
    end_pos: int

@dataclass
class ExtractedRelationship:
    """Извлеченная связь"""
    source_entity: str
    target_entity: str
    relationship_type: str  # works_for, uses, contains, etc.
    confidence: float
    context: str

@dataclass
class EntityExtractionResult:
    """Результат извлечения сущностей"""
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    confidence: float

class EntityExtractor:
    """Извлекатель сущностей и связей из текста"""
    
    def __init__(self, llm_provider: Optional[LMStudioProvider] = None):
        self.llm_provider = llm_provider or LMStudioProvider()
        
        # Предопределенные паттерны для извлечения
        self.entity_patterns = {
            'person': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Имя Фамилия
                r'\b(?:Mr|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b'  # Титулы
            ],
            'organization': [
                r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Systems|Technologies)\b',
                r'\b[A-Z]{2,}\b'  # Аббревиатуры
            ],
            'technology': [
                r'\b(?:Python|Java|JavaScript|React|Vue|Angular|Node\.js|Django|Flask|FastAPI|SQLite|PostgreSQL|MongoDB|Redis|Docker|Kubernetes|AWS|Azure|GCP)\b',
                r'\b[A-Z][a-z]+(?:\.js|\.py|\.java|\.ts|\.go|\.rs)\b'  # Технологии с расширениями
            ],
            'concept': [
                r'\b(?:AI|ML|DL|NLP|CV|API|REST|GraphQL|Microservices|DevOps|CI/CD|TDD|BDD)\b',
                r'\b(?:машинное обучение|искусственный интеллект|нейронные сети|граф знаний)\b'
            ]
        }
        
        self.relationship_patterns = {
            'works_for': [r'\b(?:работает в|работает на|сотрудник|employee of)\b'],
            'uses': [r'\b(?:использует|применяет|работает с|uses|utilizes)\b'],
            'contains': [r'\b(?:содержит|включает|состоит из|contains|includes)\b'],
            'implements': [r'\b(?:реализует|внедряет|implements|develops)\b'],
            'related_to': [r'\b(?:связан с|относится к|related to|associated with)\b']
        }
    
    async def extract_entities(self, text: str, user_id: str) -> EntityExtractionResult:
        """
        Извлечение сущностей и связей из текста
        
        Args:
            text: Текст для анализа
            user_id: ID пользователя
            
        Returns:
            Результат извлечения
        """
        try:
            # ОПТИМИЗАЦИЯ: Проверяем кэш сущностей (только для повторных запросов)
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_key = f"entities:{text_hash}"
            
            try:
                from ..cache.sqlite_cache import get_sqlite_cache
                cache = await get_sqlite_cache()
                # Используем timeout для быстрой проверки кэша
                cached_result = await asyncio.wait_for(cache.get(cache_key), timeout=1.0)
                if cached_result:
                    logger.debug(f"Cache hit for entities: {len(cached_result.get('entities', []))} entities")
                    # Восстанавливаем объекты из кэша
                    entities = [ExtractedEntity(**e) for e in cached_result.get('entities', [])]
                    relationships = [ExtractedRelationship(**r) for r in cached_result.get('relationships', [])]
                    return EntityExtractionResult(
                        entities=entities,
                        relationships=relationships,
                        confidence=cached_result.get('confidence', 0.8)
                    )
            except (Exception, asyncio.TimeoutError) as e:
                logger.debug(f"Cache miss for entities: {e}")
            
            logger.info(f"Extracting entities from text (length: {len(text)})")
            
            # 1. Извлечение сущностей с помощью паттернов (быстро и надежно)
            pattern_entities = self._extract_with_patterns(text)
            logger.debug(f"Pattern-based extraction found {len(pattern_entities)} entities")
            
            # 2. Пробуем LLM извлечение
            llm_entities = []
            try:
                llm_entities = await self._extract_with_llm(text)
                logger.debug(f"LLM extraction found {len(llm_entities)} entities")
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
            
            # 3. Если LLM не сработал, используем pattern-based fallback
            if not llm_entities:
                logger.info("LLM failed, using pattern-based fallback")
                pattern_entities = self._extract_entities_pattern_based(text)
                logger.debug(f"Pattern fallback found {len(pattern_entities)} entities")
            
            # 4. Если и паттерны не сработали, используем keywords fallback
            if not pattern_entities and not llm_entities:
                logger.info("Pattern extraction failed, using keywords fallback")
                pattern_entities = self._extract_keywords_fallback(text)
                logger.debug(f"Keywords fallback found {len(pattern_entities)} entities")
            
            # 5. Объединение и дедупликация сущностей
            all_entities = self._merge_entities(pattern_entities, llm_entities)
            
            # 6. Извлечение связей
            relationships = await self._extract_relationships(text, all_entities)
            
            # 7. Вычисление общей уверенности
            confidence = self._calculate_confidence(all_entities, relationships)
            
            result = EntityExtractionResult(
                entities=all_entities,
                relationships=relationships,
                confidence=confidence
            )
            
            extraction_method = "hybrid" if llm_entities else "pattern_fallback"
            logger.info(f"Extracted {len(all_entities)} entities and {len(relationships)} relationships using {extraction_method}")
            
            # ОПТИМИЗАЦИЯ: Асинхронное сохранение в кэш (не блокирует основной поток)
            asyncio.create_task(self._cache_entities_async(cache_key, all_entities, relationships, result.confidence))
            
            return result
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return EntityExtractionResult(entities=[], relationships=[], confidence=0.0)
    
    async def _cache_entities_async(self, cache_key: str, entities: List[ExtractedEntity], relationships: List[ExtractedRelationship], confidence: float):
        """Асинхронное сохранение сущностей в кэш"""
        try:
            from ..cache.sqlite_cache import get_sqlite_cache
            cache = await get_sqlite_cache()
            cache_data = {
                'entities': [entity.__dict__ for entity in entities],
                'relationships': [rel.__dict__ for rel in relationships],
                'confidence': confidence
            }
            await cache.set(cache_key, cache_data, ttl_sec=3600)  # Кэш на 1 час
            logger.debug(f"Cached entities: {len(entities)} entities, {len(relationships)} relationships")
        except Exception as e:
            logger.debug(f"Failed to cache entities: {e}")
    
    def _extract_with_patterns(self, text: str) -> List[ExtractedEntity]:
        """Извлечение сущностей с помощью регулярных выражений"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = ExtractedEntity(
                        name=match.group().strip(),
                        entity_type=entity_type,
                        confidence=0.7,  # Средняя уверенность для паттернов
                        context=self._get_context(text, match.start(), match.end()),
                        start_pos=match.start(),
                        end_pos=match.end()
                    )
                    entities.append(entity)
        
        return entities
    
    async def _extract_with_llm(self, text: str) -> List[ExtractedEntity]:
        """Извлечение сущностей с помощью LLM"""
        try:
            if not self.llm_provider:
                logger.debug("LLM provider not available, skipping LLM extraction")
                return []
            
            # Ограничиваем длину текста для LLM
            max_text_length = 2000
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            
            # ЗАКОММЕНТИРОВАНО ДЛЯ ТЕСТИРОВАНИЯ БЕЗ ВАЛИДАЦИИ
            # prompt = f"""
            # ТЫ ДОЛЖЕН ВЕРНУТЬ ТОЛЬКО ВАЛИДНЫЙ JSON БЕЗ ДОПОЛНИТЕЛЬНОГО ТЕКСТА!
            #
            # Задача: Извлечь сущности из текста и вернуть в строгом JSON формате.
            #
            # Текст: {text}
            #
            # ТРЕБОВАНИЯ:
            # 1. Верни ТОЛЬКО JSON объект
            # 2. НЕ добавляй объяснения, комментарии или дополнительный текст
            # 3. JSON должен быть валидным и корректным
            # 4. Максимум 5 сущностей
            # 5. confidence ДОЛЖНО быть ЧИСЛОМ от 0.0 до 1.0 (например: 0.8, 0.95, 1.0)
            # 6. НЕ пиши текст после числа в confidence
            #
            # ФОРМАТ (точно такой же):
            # {{
            #     "entities": [
            #         {{
            #             "name": "AIRI",
            #             "type": "organization",
            #             "confidence": 0.95,
            #             "context": "система искусственного интеллекта"
            #         }}
            #     ]
            # }}
            #
            # ТИПЫ СУЩНОСТЕЙ (только эти):
            # - person (человек)
            # - organization (организация) 
            # - technology (технология)
            # - concept (концепция)
            # - project (проект)
            # - service (сервис)
            #
            # ПРИМЕРЫ ПРАВИЛЬНОГО confidence:
            # - 0.8
            # - 0.95
            # - 1.0
            # - 0.5
            #
            # ВЕРНИ ТОЛЬКО JSON:
            # """
            
            prompt = f"""
            ТЫ ДОЛЖЕН ВЕРНУТЬ ТОЛЬКО ВАЛИДНЫЙ JSON БЕЗ ДОПОЛНИТЕЛЬНОГО ТЕКСТА!

            Задача: Извлечь сущности из текста и вернуть в строгом JSON формате.

            Текст: {text}

            ТРЕБОВАНИЯ:
            1. Верни ТОЛЬКО JSON объект
            2. НЕ добавляй объяснения, комментарии или дополнительный текст
            3. JSON должен быть валидным и корректным
            4. Максимум 5 сущностей
            5. confidence ДОЛЖНО быть ЧИСЛОМ от 0.0 до 1.0 (например: 0.8, 0.95, 1.0)
            6. НЕ пиши текст после числа в confidence

            ФОРМАТ (точно такой же):
            {{
                "entities": [
                    {{
                        "name": "AIRI",
                        "type": "organization",
                        "confidence": 0.95,
                        "context": "система искусственного интеллекта"
                    }}
                ]
            }}

            ТИПЫ СУЩНОСТЕЙ (только эти):
            - person (человек)
            - organization (организация) 
            - technology (технология)
            - concept (концепция)
            - project (проект)
            - service (сервис)

            ПРИМЕРЫ ПРАВИЛЬНОГО confidence:
            - 0.8
            - 0.95
            - 1.0
            - 0.5

            ПРИМЕРЫ НЕПРАВИЛЬНОГО confidence:
            - 0.8 (высокая)
            - 0.95 (очень высокая)
            - 1.0 (максимальная)
            - 0.5 (средняя)

            ВЕРНИ ТОЛЬКО JSON:
            """
            
            # Добавляем timeout для LLM запроса
            response = await asyncio.wait_for(
                self.llm_provider.generate_text(prompt, max_tokens=1000, task_type="fact_extraction"),
                timeout=30.0
            )
            
            # Парсим JSON ответ
            entities = self._parse_llm_response(response)
            logger.debug(f"LLM extracted {len(entities)} entities")
            return entities
            
        except asyncio.TimeoutError:
            logger.warning("LLM entity extraction timed out")
            return []
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")
            return []
    
    def _parse_llm_response(self, response: str) -> List[ExtractedEntity]:
        """Надежный парсинг с многоуровневой валидацией"""
        try:
            import json
            
            # Безопасное логирование с правильной кодировкой
            try:
                safe_response = response[:500].encode('utf-8', errors='replace').decode('utf-8')
                logger.debug(f"Raw LLM response: {safe_response}...")
            except Exception:
                logger.debug(f"Raw LLM response: [encoding error] {len(response)} chars")
            
            # Уровень 1: Предварительная очистка
            cleaned_response = self._clean_response(response)
            
            # Уровень 2: Проверка структуры
            if not self._validate_json_structure(cleaned_response):
                logger.warning("Invalid JSON structure")
                return []
            
            # Уровень 3: Парсинг JSON
            try:
                data = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                return []
            
            # Уровень 4: Валидация сущностей
            entities = self._validate_entities(data.get('entities', []))
            
            # Уровень 5: Преобразование в объекты
            result_entities = []
            for entity_data in entities:
                try:
                    entity = ExtractedEntity(
                        name=entity_data['name'],
                        entity_type=entity_data['type'],
                        confidence=float(entity_data['confidence']),
                        context=entity_data['context'],
                        start_pos=0,
                        end_pos=0
                    )
                    result_entities.append(entity)
                except Exception as item_error:
                    logger.warning(f"Failed to create entity: {item_error}, data: {entity_data}")
                    continue
            
            logger.info(f"Successfully parsed {len(result_entities)} entities from LLM response")
            return result_entities
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Безопасное логирование с правильной кодировкой
            try:
                safe_response = response.encode('utf-8', errors='replace').decode('utf-8')
                logger.debug(f"Response that failed to parse: {safe_response}")
            except Exception:
                logger.debug(f"Response that failed to parse: [encoding error] {len(response)} chars")
            return []
    
    def _clean_response(self, response: str) -> str:
        """Очистка ответа от лишних символов"""
        # Убираем markdown code blocks
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        
        response = response.strip()
        
        # Ищем JSON в ответе
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group()
        
        return response
    
    def _validate_json_structure(self, json_str: str) -> bool:
        """Проверка структуры JSON перед парсингом"""
        try:
            # Проверяем базовую структуру
            if not json_str.strip().startswith('{'):
                return False
            if not json_str.strip().endswith('}'):
                return False
                
            # Проверяем наличие обязательных полей
            required_fields = ['entities']
            for field in required_fields:
                if f'"{field}"' not in json_str:
                    return False
                    
            return True
        except:
            return False
    
    def _validate_entities(self, entities: List[dict]) -> List[dict]:
        """Валидация и очистка сущностей"""
        valid_entities = []
        
        for entity in entities:
            # Проверяем обязательные поля
            if not all(key in entity for key in ['name', 'type', 'confidence', 'context']):
                continue
                
            # Валидируем типы
            if not isinstance(entity['name'], str) or not entity['name'].strip():
                continue
            
            # Поддерживаем русские и английские типы
            valid_types = ['person', 'organization', 'technology', 'concept', 'project', 'service',
                          'человек', 'организация', 'технология', 'концепция', 'проект', 'сервис']
            if entity['type'] not in valid_types:
                continue
            
            # Исправляем confidence если это не число
            confidence = entity['confidence']
            if not isinstance(confidence, (int, float)):
                # Если confidence содержит текст, извлекаем число или используем 0.5
                if isinstance(confidence, str):
                    import re
                    # Ищем числа в начале строки (0, 0.5, 1.0, etc.)
                    # Убираем лишние пробелы и ищем число в начале
                    clean_confidence = str(confidence).strip()
                    numbers = re.findall(r'^(0\.\d+|\d+\.\d+|\d+)', clean_confidence)
                    if numbers:
                        confidence = float(numbers[0])
                        # Ограничиваем диапазон 0-1
                        confidence = max(0.0, min(1.0, confidence))
                        logger.debug(f"Extracted confidence {confidence} from '{clean_confidence}'")
                    else:
                        confidence = 0.5
                        logger.debug(f"Could not extract confidence from '{clean_confidence}', using 0.5")
                else:
                    confidence = 0.5
            
            if not (0 <= confidence <= 1):
                confidence = 0.5  # Fallback значение
            
            entity['confidence'] = confidence
            
            # Нормализуем типы (русские → английские)
            type_mapping = {
                'человек': 'person',
                'организация': 'organization', 
                'технология': 'technology',
                'концепция': 'concept',
                'проект': 'project',
                'сервис': 'service'
            }
            entity['type'] = type_mapping.get(entity['type'], entity['type'])
                
            # Очищаем данные
            entity['name'] = entity['name'].strip()
            entity['context'] = entity['context'].strip()
            
            valid_entities.append(entity)
        
        return valid_entities
    
    def _fix_json_errors(self, json_str: str) -> str:
        """Исправление распространенных ошибок в JSON от LLM"""
        try:
            # Убираем лишние символы в начале и конце
            json_str = json_str.strip()
            
            # Убираем markdown code blocks если есть
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.startswith('```'):
                json_str = json_str[3:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            
            json_str = json_str.strip()
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: LLM генерирует "endentities:" вместо "entities:"
            json_str = re.sub(r'endentities\s*:', '"entities":', json_str, flags=re.IGNORECASE)
            
            # Дополнительное исправление для случая без кавычек
            json_str = re.sub(r'endentities\s*\[', '"entities": [', json_str, flags=re.IGNORECASE)
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: LLM генерирует "end of document]" вместо "]"
            json_str = re.sub(r'end of document\]', ']', json_str, flags=re.IGNORECASE)
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: LLM генерирует "end of document text provided]" вместо "]"
            json_str = re.sub(r'end of document text provided\]', ']', json_str, flags=re.IGNORECASE)
            
            # Исправляем другие распространенные ошибки LLM
            json_str = re.sub(r'(\w+)\s*:', r'"\1":', json_str)  # Добавляем кавычки к ключам без кавычек
            
            # Исправляем обрезанные строки - добавляем закрывающие скобки если нужно
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            
            # Добавляем недостающие закрывающие скобки
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            if open_brackets > close_brackets:
                json_str += ']' * (open_brackets - close_brackets)
            
            # Исправляем обрезанные строки в значениях
            json_str = re.sub(r'"([^"]*)\s*$', r'"\1"', json_str)  # Добавляем кавычки в конце строк
            
            # Убираем лишние запятые перед закрывающими скобками
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            logger.debug(f"Fixed JSON: {json_str[:200]}...")
            return json_str
            
        except Exception as e:
            logger.warning(f"Failed to fix JSON errors: {e}")
            return json_str
    
    def _extract_entities_pattern_based(self, text: str) -> List[ExtractedEntity]:
        """Извлечение сущностей на основе паттернов (fallback)"""
        entities = []
        
        # Паттерны для разных типов сущностей
        patterns = {
            'organization': [
                r'\b[A-ZА-Я]{2,}\b',  # Заглавные буквы (AIRI, NASA)
                r'\b\w+\s+(система|системы|компания|корпорация)\b',  # Система, компания
            ],
            'technology': [
                r'\b(Python|Java|JavaScript|SQLite|MySQL|PostgreSQL|Node\.js|React|Vue)\b',  # Технологии
                r'\b(машинное обучение|нейронные сети|искусственный интеллект|deep learning)\b',  # AI термины
            ],
            'concept': [
                r'\b(алгоритм|метод|подход|технология|процесс|анализ|обработка)\b',  # Концепции
            ]
        }
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = ExtractedEntity(
                        name=match.group().strip(),
                        entity_type=entity_type,
                        confidence=0.7,  # Средняя уверенность для паттернов
                        context=text[max(0, match.start()-20):match.end()+20],
                        start_pos=match.start(),
                        end_pos=match.end()
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_keywords_fallback(self, text: str) -> List[ExtractedEntity]:
        """Последний fallback - простые ключевые слова"""
        entities = []
        
        # Простые ключевые слова
        keywords = {
            'technology': ['Python', 'SQLite', 'JavaScript', 'Java', 'C++', 'Go', 'Rust'],
            'concept': ['алгоритм', 'метод', 'подход', 'технология', 'процесс'],
            'organization': ['AIRI', 'Google', 'Microsoft', 'Apple', 'Amazon']
        }
        
        for entity_type, word_list in keywords.items():
            for word in word_list:
                if word.lower() in text.lower():
                    entity = ExtractedEntity(
                        name=word,
                        entity_type=entity_type,
                        confidence=0.5,  # Низкая уверенность для ключевых слов
                        context=text,
                        start_pos=0,
                        end_pos=0
                    )
                    entities.append(entity)
        
        return entities
    
    def _merge_entities(self, pattern_entities: List[ExtractedEntity], 
                       llm_entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Объединение и дедупликация сущностей"""
        merged = []
        seen = set()
        
        # Добавляем сущности из паттернов
        for entity in pattern_entities:
            key = entity.name.lower()
            if key not in seen:
                seen.add(key)
                merged.append(entity)
        
        # Добавляем сущности из LLM (если не дублируются)
        for entity in llm_entities:
            key = entity.name.lower()
            if key not in seen:
                seen.add(key)
                merged.append(entity)
            else:
                # Обновляем уверенность для существующих сущностей
                for existing in merged:
                    if existing.name.lower() == key:
                        existing.confidence = max(existing.confidence, entity.confidence)
                        break
        
        return merged
    
    async def _extract_relationships(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Извлечение связей между сущностями"""
        relationships = []
        
        # Извлечение с помощью паттернов
        pattern_relationships = self._extract_relationships_with_patterns(text, entities)
        relationships.extend(pattern_relationships)
        
        # Извлечение с помощью LLM
        llm_relationships = await self._extract_relationships_with_llm(text, entities)
        relationships.extend(llm_relationships)
        
        return relationships
    
    def _extract_relationships_with_patterns(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Извлечение связей с помощью паттернов"""
        relationships = []
        
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Ищем ближайшие сущности
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end]
                    
                    # Ищем сущности в контексте
                    nearby_entities = [e for e in entities if context_start <= e.start_pos <= context_end]
                    
                    if len(nearby_entities) >= 2:
                        # Создаем связи между ближайшими сущностями
                        for i in range(len(nearby_entities) - 1):
                            rel = ExtractedRelationship(
                                source_entity=nearby_entities[i].name,
                                target_entity=nearby_entities[i + 1].name,
                                relationship_type=rel_type,
                                confidence=0.6,
                                context=context
                            )
                            relationships.append(rel)
        
        return relationships
    
    async def _extract_relationships_with_llm(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Извлечение связей с помощью LLM"""
        try:
            if not self.llm_provider or not entities:
                return []
            
            entity_names = [e.name for e in entities]
            
            # ЗАКОММЕНТИРОВАНО ДЛЯ ТЕСТИРОВАНИЯ БЕЗ ВАЛИДАЦИИ
            # prompt = f"""
            # Найди связи между сущностями в тексте:
            #
            # Текст: {text[:1000]}...
            # Сущности: {', '.join(entity_names)}
            #
            # Верни JSON с массивом связей:
            # {{
            #     "relationships": [
            #         {{
            #             "source": "источник",
            #             "target": "цель", 
            #             "type": "works_for|uses|contains|implements|related_to",
            #             "confidence": 0.0-1.0,
            #             "context": "контекст из текста"
            #         }}
            #     ]
            # }}
            #
            # Найди только реальные связи между указанными сущностями.
            # """
            
            prompt = f"""
            Найди связи между сущностями в тексте:

            Сущности: {', '.join(entity_names)}
            Текст: {text[:500]}

            ТРЕБОВАНИЯ:
            1. Верни ТОЛЬКО JSON объект
            2. НЕ добавляй объяснения или комментарии
            3. JSON должен быть валидным
            4. Максимум 3 связи

            ФОРМАТ:
            {{
                "relationships": [
                    {{
                        "source": "AIRI",
                        "target": "Python",
                        "type": "uses",
                        "confidence": 0.9
                    }}
                ]
            }}

            ТИПЫ СВЯЗЕЙ:
            - uses (использует)
            - works_with (работает с)
            - related_to (связан с)
            - part_of (часть)
            - manages (управляет)

            Найди только реальные связи между указанными сущностями.
            """
            
            response = await self.llm_provider.generate_text(prompt, max_tokens=1000, task_type="fact_extraction")
            
            # Парсим JSON ответ
            relationships = self._parse_llm_relationships(response)
            return relationships
            
        except Exception as e:
            logger.warning(f"LLM relationship extraction failed: {e}")
            return []
    
    def _parse_llm_relationships(self, response: str) -> List[ExtractedRelationship]:
        """Парсинг связей из ответа LLM"""
        try:
            import json
            
            # Улучшенный парсинг JSON - ищем только первый валидный JSON объект
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if not json_match:
                # Fallback: попробуем найти JSON с более простым паттерном
                json_match = re.search(r'\{.*?"relationships".*?\}', response, re.DOTALL)
                if not json_match:
                    return []
            
            json_str = json_match.group()
            
            # Попытка исправить распространенные ошибки JSON
            json_str = self._fix_json_errors(json_str)
            
            data = json.loads(json_str)
            relationships = []
            
            for item in data.get('relationships', []):
                rel = ExtractedRelationship(
                    source_entity=item.get('source', ''),
                    target_entity=item.get('target', ''),
                    relationship_type=item.get('type', 'related_to'),
                    confidence=float(item.get('confidence', 0.5)),
                    context=item.get('context', '')
                )
                relationships.append(rel)
            
            return relationships
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM relationships: {e}")
            return []
    
    def _fix_json_errors(self, json_str: str) -> str:
        """Исправление распространенных ошибок JSON"""
        try:
            # Удаляем лишние символы в конце
            json_str = json_str.strip()
            
            # Удаляем все символы после последней закрывающей скобки
            last_brace = json_str.rfind('}')
            if last_brace != -1:
                json_str = json_str[:last_brace + 1]
            
            # Исправляем отсутствующие запятые между объектами
            json_str = re.sub(r'}\s*{', '},{', json_str)
            
            # Исправляем отсутствующие запятые в массивах
            json_str = re.sub(r'}\s*\[', '},[', json_str)
            json_str = re.sub(r'\]\s*{', '],{', json_str)
            
            # Исправляем отсутствующие запятые после строк
            json_str = re.sub(r'"\s*"', '","', json_str)
            
            # Удаляем лишние запятые перед закрывающими скобками
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            
            # Удаляем лишние символы в начале (если есть)
            first_brace = json_str.find('{')
            if first_brace > 0:
                json_str = json_str[first_brace:]
            
            return json_str
            
        except Exception as e:
            logger.warning(f"Failed to fix JSON errors: {e}")
            return json_str
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Получение контекста вокруг позиции"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _calculate_confidence(self, entities: List[ExtractedEntity], 
                            relationships: List[ExtractedRelationship]) -> float:
        """Вычисление общей уверенности"""
        if not entities and not relationships:
            return 0.0
        
        entity_conf = sum(e.confidence for e in entities) / len(entities) if entities else 0.0
        rel_conf = sum(r.confidence for r in relationships) / len(relationships) if relationships else 0.0
        
        # Взвешенная сумма (сущности важнее связей)
        return entity_conf * 0.7 + rel_conf * 0.3
