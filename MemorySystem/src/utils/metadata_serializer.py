"""
Централизованная система сериализации метаданных для AIRI Memory System.
Исправляет проблемы с неэффективным использованием __import__("json") и обеспечивает
консистентную сериализацию/десериализацию метаданных.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class MetadataSerializer:
    """Централизованный сериализатор метаданных"""
    
    @staticmethod
    def serialize_metadata(data: Any) -> str:
        """
        Сериализует данные в JSON строку с правильной обработкой ошибок
        
        Args:
            data: Данные для сериализации
            
        Returns:
            JSON строка
            
        Raises:
            ValueError: Если данные не могут быть сериализованы
        """
        try:
            return json.dumps(data, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as e:
            logger.error(f"Ошибка сериализации метаданных: {e}, данные: {data}")
            raise ValueError(f"Не удалось сериализовать метаданные: {e}")
    
    @staticmethod
    def deserialize_metadata(json_str: str, default: Any = None) -> Any:
        """
        Десериализует JSON строку в данные с обработкой ошибок
        
        Args:
            json_str: JSON строка для десериализации
            default: Значение по умолчанию при ошибке
            
        Returns:
            Десериализованные данные или default
        """
        if not json_str:
            return default
            
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Ошибка десериализации метаданных: {e}, строка: {json_str}")
            return default
    
    @staticmethod
    def safe_serialize_list(data: List[Any]) -> str:
        """
        Безопасная сериализация списка
        
        Args:
            data: Список для сериализации
            
        Returns:
            JSON строка или "[]" при ошибке
        """
        if not data:
            return "[]"
        return MetadataSerializer.serialize_metadata(data)
    
    @staticmethod
    def safe_deserialize_list(json_str: str) -> List[Any]:
        """
        Безопасная десериализация списка
        
        Args:
            json_str: JSON строка
            
        Returns:
            Список или пустой список при ошибке
        """
        result = MetadataSerializer.deserialize_metadata(json_str, [])
        if not isinstance(result, list):
            logger.warning(f"Ожидался список, получен {type(result)}: {result}")
            return []
        return result
    
    @staticmethod
    def safe_serialize_dict(data: Dict[str, Any]) -> str:
        """
        Безопасная сериализация словаря
        
        Args:
            data: Словарь для сериализации
            
        Returns:
            JSON строка или "{}" при ошибке
        """
        if not data:
            return "{}"
        return MetadataSerializer.serialize_metadata(data)
    
    @staticmethod
    def safe_deserialize_dict(json_str: str) -> Dict[str, Any]:
        """
        Безопасная десериализация словаря
        
        Args:
            json_str: JSON строка
            
        Returns:
            Словарь или пустой словарь при ошибке
        """
        result = MetadataSerializer.deserialize_metadata(json_str, {})
        if not isinstance(result, dict):
            logger.warning(f"Ожидался словарь, получен {type(result)}: {result}")
            return {}
        return result
    
    @staticmethod
    def prepare_metadata_for_chromadb(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подготавливает метаданные для сохранения в ChromaDB
        
        Args:
            metadata: Исходные метаданные
            
        Returns:
            Метаданные, готовые для ChromaDB
        """
        prepared = {}
        
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                # Сериализуем сложные структуры
                prepared[key] = MetadataSerializer.serialize_metadata(value)
            elif isinstance(value, datetime):
                # Конвертируем datetime в ISO строку
                prepared[key] = value.isoformat()
            else:
                # Простые типы оставляем как есть
                prepared[key] = value
                
        return prepared
    
    @staticmethod
    def parse_metadata_from_chromadb(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Парсит метаданные из ChromaDB обратно в Python объекты
        
        Args:
            metadata: Метаданные из ChromaDB
            
        Returns:
            Парсированные метаданные
        """
        parsed = {}
        
        # Поля, которые нужно десериализовать из JSON
        json_fields = {
            'related_concepts', 'tags', 'participants', 'related_memories',
            'properties', 'emotion_data', 'formatted_emotion'
        }
        
        # Поля с датами
        date_fields = {
            'timestamp', 'last_accessed', 'last_updated', 'created_at'
        }
        
        for key, value in metadata.items():
            if key in json_fields and isinstance(value, str):
                # Десериализуем JSON поля
                if key in ['related_concepts', 'tags', 'participants', 'related_memories']:
                    parsed[key] = MetadataSerializer.safe_deserialize_list(value)
                else:
                    parsed[key] = MetadataSerializer.safe_deserialize_dict(value)
            elif key in date_fields and isinstance(value, str):
                # Парсим даты
                try:
                    parsed[key] = datetime.fromisoformat(value)
                except ValueError:
                    logger.warning(f"Не удалось парсить дату {key}: {value}")
                    parsed[key] = value
            else:
                # Остальные поля оставляем как есть
                parsed[key] = value
                
        return parsed

# Глобальные функции для обратной совместимости
def serialize_metadata(data: Any) -> str:
    """Глобальная функция для сериализации метаданных"""
    return MetadataSerializer.serialize_metadata(data)

def deserialize_metadata(json_str: str, default: Any = None) -> Any:
    """Глобальная функция для десериализации метаданных"""
    return MetadataSerializer.deserialize_metadata(json_str, default)

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Безопасная замена для json.loads"""
    return MetadataSerializer.deserialize_metadata(json_str, default)

def safe_json_dumps(data: Any) -> str:
    """Безопасная замена для json.dumps"""
    return MetadataSerializer.serialize_metadata(data)

