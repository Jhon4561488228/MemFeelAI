#!/usr/bin/env python3
"""
Утилиты валидации для системы памяти
"""

from typing import Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

def validate_importance(importance: Any, default: float = 0.5) -> float:
    """
    Валидация и нормализация значения importance
    
    Args:
        importance: Значение importance для валидации
        default: Значение по умолчанию, если валидация не пройдена
        
    Returns:
        float: Валидное значение importance в диапазоне [0.0, 1.0]
    """
    try:
        # Преобразуем в float
        if isinstance(importance, str):
            importance = float(importance)
        elif not isinstance(importance, (int, float)):
            logger.warning(f"Invalid importance type: {type(importance)}, using default: {default}")
            return default
        
        # Проверяем диапазон
        if not (0.0 <= importance <= 1.0):
            logger.warning(f"Importance out of range [0.0, 1.0]: {importance}, clamping to valid range")
            importance = max(0.0, min(1.0, importance))
        
        return float(importance)
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to validate importance '{importance}': {e}, using default: {default}")
        return default

def validate_confidence(confidence: Any, default: float = 0.5) -> float:
    """
    Валидация и нормализация значения confidence
    
    Args:
        confidence: Значение confidence для валидации
        default: Значение по умолчанию, если валидация не пройдена
        
    Returns:
        float: Валидное значение confidence в диапазоне [0.0, 1.0]
    """
    try:
        # Преобразуем в float
        if isinstance(confidence, str):
            confidence = float(confidence)
        elif not isinstance(confidence, (int, float)):
            logger.warning(f"Invalid confidence type: {type(confidence)}, using default: {default}")
            return default
        
        # Проверяем диапазон
        if not (0.0 <= confidence <= 1.0):
            logger.warning(f"Confidence out of range [0.0, 1.0]: {confidence}, clamping to valid range")
            confidence = max(0.0, min(1.0, confidence))
        
        return float(confidence)
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to validate confidence '{confidence}': {e}, using default: {default}")
        return default

def validate_min_confidence(min_confidence: Any, default: float = 0.25) -> float:
    """
    Валидация и нормализация значения min_confidence
    
    Args:
        min_confidence: Значение min_confidence для валидации
        default: Значение по умолчанию, если валидация не пройдена
        
    Returns:
        float: Валидное значение min_confidence в диапазоне [0.0, 1.0]
    """
    try:
        # Преобразуем в float
        if isinstance(min_confidence, str):
            min_confidence = float(min_confidence)
        elif not isinstance(min_confidence, (int, float)):
            logger.warning(f"Invalid min_confidence type: {type(min_confidence)}, using default: {default}")
            return default
        
        # Проверяем диапазон
        if not (0.0 <= min_confidence <= 1.0):
            logger.warning(f"Min_confidence out of range [0.0, 1.0]: {min_confidence}, clamping to valid range")
            min_confidence = max(0.0, min(1.0, min_confidence))
        
        return float(min_confidence)
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to validate min_confidence '{min_confidence}': {e}, using default: {default}")
        return default

def validate_distance_threshold(distance: Any, default: float = 0.8) -> float:
    """
    Валидация и нормализация значения distance threshold
    
    Args:
        distance: Значение distance для валидации
        default: Значение по умолчанию, если валидация не пройдена
        
    Returns:
        float: Валидное значение distance threshold
    """
    try:
        # Преобразуем в float
        if isinstance(distance, str):
            distance = float(distance)
        elif not isinstance(distance, (int, float)):
            logger.warning(f"Invalid distance type: {type(distance)}, using default: {default}")
            return default
        
        # Проверяем, что distance положительное
        if distance < 0.0:
            logger.warning(f"Distance threshold negative: {distance}, using default: {default}")
            return default
        
        return float(distance)
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to validate distance '{distance}': {e}, using default: {default}")
        return default

def validate_metadata_values(metadata: dict) -> dict:
    """
    Валидация всех значений в метаданных
    
    Args:
        metadata: Словарь метаданных для валидации
        
    Returns:
        dict: Валидированный словарь метаданных
    """
    if not isinstance(metadata, dict):
        logger.warning(f"Invalid metadata type: {type(metadata)}, returning empty dict")
        return {}
    
    validated_metadata = metadata.copy()
    
    # Валидируем importance
    if "importance" in validated_metadata:
        validated_metadata["importance"] = validate_importance(validated_metadata["importance"])
    
    # Валидируем confidence
    if "confidence" in validated_metadata:
        validated_metadata["confidence"] = validate_confidence(validated_metadata["confidence"])
    
    return validated_metadata

