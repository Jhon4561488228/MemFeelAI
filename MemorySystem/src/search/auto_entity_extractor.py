"""
Auto Entity Extractor для AIRI Memory System
Главный класс для автоматического извлечения сущностей и построения графа
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from .entity_extractor import EntityExtractor, EntityExtractionResult
from .graph_builder import GraphBuilder, GraphBuildResult
from ..memory_levels.graph_memory_sqlite import GraphMemoryManagerSQLite as GraphMemoryManager
from ..providers.lm_studio_provider import LMStudioProvider

@dataclass
class AutoExtractionResult:
    """Результат автоматического извлечения"""
    extraction_result: EntityExtractionResult
    build_result: GraphBuildResult
    total_entities: int
    total_relationships: int
    success: bool
    processing_time: float

class AutoEntityExtractor:
    """Автоматический извлекатель сущностей и построитель графа"""
    
    def __init__(self, graph_manager: GraphMemoryManager, 
                 memory_orchestrator: Optional[Any] = None,
                 llm_provider: Optional[LMStudioProvider] = None):
        self.graph_manager = graph_manager
        self.memory_orchestrator = memory_orchestrator
        self.entity_extractor = EntityExtractor(llm_provider)
        self.graph_builder = GraphBuilder(graph_manager)
    
    async def extract_and_build_from_text(self, text: str, user_id: str,
                                        importance: float = 0.5) -> AutoExtractionResult:
        """
        Извлечение сущностей из текста и построение графа
        
        Args:
            text: Текст для анализа
            user_id: ID пользователя
            importance: Важность создаваемых узлов
            
        Returns:
            Результат автоматического извлечения
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Starting auto extraction from text (length: {len(text)})")
            
            # 1. Извлекаем сущности и связи
            extraction_result = await self.entity_extractor.extract_entities(text, user_id)
            
            if not extraction_result.entities:
                logger.warning("No entities extracted from text")
                return AutoExtractionResult(
                    extraction_result=extraction_result,
                    build_result=GraphBuildResult(0, 0, 0, 0, 0.0, ["No entities found"]),
                    total_entities=0,
                    total_relationships=0,
                    success=False,
                    processing_time=time.time() - start_time
                )
            
            # 2. Строим граф
            build_result = await self.graph_builder.build_graph_from_text(
                text, user_id, extraction_result, importance
            )
            
            processing_time = time.time() - start_time
            
            result = AutoExtractionResult(
                extraction_result=extraction_result,
                build_result=build_result,
                total_entities=len(extraction_result.entities),
                total_relationships=len(extraction_result.relationships),
                success=build_result.nodes_added > 0 or build_result.edges_added > 0,
                processing_time=processing_time
            )
            
            logger.info(f"Auto extraction completed: {result.total_entities} entities, "
                       f"{result.total_relationships} relationships, "
                       f"{result.build_result.nodes_added} nodes added, "
                       f"{result.build_result.edges_added} edges added, "
                       f"time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Auto extraction failed: {e}")
            processing_time = time.time() - start_time
            
            return AutoExtractionResult(
                extraction_result=EntityExtractionResult([], [], 0.0),
                build_result=GraphBuildResult(0, 0, 0, 0, 0.0, [str(e)]),
                total_entities=0,
                total_relationships=0,
                success=False,
                processing_time=processing_time
            )
    
    async def extract_and_build_from_memories(self, user_id: str, 
                                            memory_ids: List[str],
                                            importance: float = 0.7) -> AutoExtractionResult:
        """
        Извлечение сущностей из записей памяти и построение графа
        
        Args:
            user_id: ID пользователя
            memory_ids: Список ID записей памяти
            importance: Важность создаваемых узлов
            
        Returns:
            Результат автоматического извлечения
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Starting auto extraction from {len(memory_ids)} memory records")
            
            # Извлекаем тексты из записей памяти
            texts = []
            for memory_id in memory_ids:
                try:
                    # Получаем запись памяти через MemoryOrchestrator
                    memory_record = await self._get_memory_record(memory_id)
                    if memory_record and memory_record.get('content'):
                        texts.append(memory_record['content'])
                except Exception as e:
                    logger.warning(f"Failed to get memory record {memory_id}: {e}")
                    continue
            
            if not texts:
                logger.warning("No memory records found or no content extracted")
                return AutoExtractionResult(
                    extraction_result=EntityExtractionResult([], [], 0.0),
                    build_result=GraphBuildResult(0, 0, 0, 0, 0.0, ["No memory content found"]),
                    total_entities=0,
                    total_relationships=0,
                    success=False,
                    processing_time=time.time() - start_time
                )
            
            # Объединяем все тексты
            combined_text = "\n\n".join(texts)
            
            # Извлекаем сущности из объединенного текста
            result = await self.extract_and_build_from_text(
                text=combined_text,
                user_id=user_id,
                importance=importance
            )
            
            processing_time = time.time() - start_time
            
            return AutoExtractionResult(
                extraction_result=EntityExtractionResult([], [], 0.0),
                build_result=GraphBuildResult(0, 0, 0, 0, 0.0, ["Not implemented"]),
                total_entities=0,
                total_relationships=0,
                success=False,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Memory-based auto extraction failed: {e}")
            processing_time = time.time() - start_time
            
            return AutoExtractionResult(
                extraction_result=EntityExtractionResult([], [], 0.0),
                build_result=GraphBuildResult(0, 0, 0, 0, 0.0, [str(e)]),
                total_entities=0,
                total_relationships=0,
                success=False,
                processing_time=processing_time
            )
    
    async def batch_extract_and_build(self, texts: List[str], user_id: str,
                                    importance: float = 0.5) -> List[AutoExtractionResult]:
        """
        Пакетное извлечение сущностей из нескольких текстов
        
        Args:
            texts: Список текстов для анализа
            user_id: ID пользователя
            importance: Важность создаваемых узлов
            
        Returns:
            Список результатов извлечения
        """
        try:
            logger.info(f"Starting batch extraction from {len(texts)} texts")
            
            results = []
            for i, text in enumerate(texts):
                logger.info(f"Processing text {i+1}/{len(texts)}")
                result = await self.extract_and_build_from_text(text, user_id, importance)
                results.append(result)
            
            # Статистика
            total_entities = sum(r.total_entities for r in results)
            total_relationships = sum(r.total_relationships for r in results)
            total_nodes_added = sum(r.build_result.nodes_added for r in results)
            total_edges_added = sum(r.build_result.edges_added for r in results)
            successful_extractions = sum(1 for r in results if r.success)
            
            logger.info(f"Batch extraction completed: {total_entities} entities, "
                       f"{total_relationships} relationships, "
                       f"{total_nodes_added} nodes added, "
                       f"{total_edges_added} edges added, "
                       f"{successful_extractions}/{len(texts)} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            return []
    
    def get_extraction_stats(self, results: List[AutoExtractionResult]) -> Dict[str, Any]:
        """Получение статистики по результатам извлечения"""
        if not results:
            return {}
        
        total_entities = sum(r.total_entities for r in results)
        total_relationships = sum(r.total_relationships for r in results)
        total_nodes_added = sum(r.build_result.nodes_added for r in results)
        total_edges_added = sum(r.build_result.edges_added for r in results)
        total_processing_time = sum(r.processing_time for r in results)
        successful_extractions = sum(1 for r in results if r.success)
        
        avg_confidence = sum(r.extraction_result.confidence for r in results) / len(results)
        avg_processing_time = total_processing_time / len(results)
        
        return {
            'total_texts': len(results),
            'successful_extractions': successful_extractions,
            'success_rate': successful_extractions / len(results),
            'total_entities': total_entities,
            'total_relationships': total_relationships,
            'total_nodes_added': total_nodes_added,
            'total_edges_added': total_edges_added,
            'avg_confidence': avg_confidence,
            'total_processing_time': total_processing_time,
            'avg_processing_time': avg_processing_time
        }
    
    async def _get_memory_record(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить запись памяти по ID
        
        Args:
            memory_id: ID записи памяти
            
        Returns:
            Словарь с данными записи памяти или None
        """
        try:
            logger.debug(f"Getting memory record: {memory_id}")
            
            if not self.memory_orchestrator:
                logger.warning("MemoryOrchestrator not available for memory record retrieval")
                return None
            
            # Получаем запись памяти через MemoryOrchestrator
            # Ищем во всех уровнях памяти
            memory_levels = [
                'working_memory',
                'short_term_memory', 
                'episodic_memory',
                'semantic_memory'
            ]
            
            for level_name in memory_levels:
                try:
                    level_manager = getattr(self.memory_orchestrator, level_name, None)
                    if level_manager and hasattr(level_manager, 'get'):
                        memory_record = await level_manager.get(memory_id)
                        if memory_record:
                            logger.debug(f"Found memory record in {level_name}")
                            return {
                                'id': memory_record.get('id', memory_id),
                                'content': memory_record.get('content', ''),
                                'metadata': memory_record.get('metadata', {}),
                                'level': level_name,
                                'timestamp': memory_record.get('timestamp', ''),
                                'user_id': memory_record.get('user_id', '')
                            }
                except Exception as e:
                    logger.debug(f"Level {level_name} not accessible: {e}")
                    continue
            
            logger.warning(f"Memory record {memory_id} not found in any level")
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get memory record {memory_id}: {e}")
            return None
