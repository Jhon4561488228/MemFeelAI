"""
Graph Builder для AIRI Memory System
Автоматическое построение графа из извлеченных сущностей и связей
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from loguru import logger

from .entity_extractor import ExtractedEntity, ExtractedRelationship, EntityExtractionResult
from ..memory_levels.graph_memory_sqlite import GraphMemoryManagerSQLite as GraphMemoryManager

@dataclass
class GraphBuildResult:
    """Результат построения графа"""
    nodes_added: int
    edges_added: int
    nodes_updated: int
    edges_updated: int
    confidence: float
    errors: List[str]

class GraphBuilder:
    """Построитель графа из извлеченных сущностей"""
    
    def __init__(self, graph_manager: GraphMemoryManager):
        self.graph_manager = graph_manager
        
        # Маппинг типов сущностей на типы узлов
        self.entity_type_mapping = {
            'person': 'person',
            'organization': 'organization', 
            'technology': 'technology',
            'concept': 'concept',
            'project': 'project',
            'service': 'service',
            'component': 'component',
            'feature': 'feature'
        }
        
        # Маппинг типов связей
        self.relationship_type_mapping = {
            'works_for': 'works_for',
            'uses': 'uses',
            'contains': 'contains',
            'implements': 'implements',
            'related_to': 'related_to',
            'has_component': 'has_component',
            'includes': 'includes'
        }
    
    async def _extract_entities_from_content(self, content: str, user_id: str) -> List[ExtractedEntity]:
        """Извлечение сущностей из контента"""
        try:
            from ..search.entity_extractor import EntityExtractor
            
            # Создаем экстрактор сущностей
            extractor = EntityExtractor()
            
            # Извлекаем сущности
            extraction_result = await extractor.extract_entities(content, user_id)
            
            # Преобразуем в ExtractedEntity объекты
            entities = []
            for entity_data in extraction_result.get("entities", []):
                entity = ExtractedEntity(
                    name=entity_data.get("name", ""),
                    entity_type=entity_data.get("type", "unknown"),
                    confidence=entity_data.get("confidence", 0.5),
                    context=entity_data.get("context", ""),
                    start_pos=entity_data.get("start_pos", 0),
                    end_pos=entity_data.get("end_pos", 0)
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities from content: {e}")
            return []
    
    async def _add_entity_to_graph(self, entity: ExtractedEntity, user_id: str) -> str:
        """Добавление сущности в граф"""
        try:
            if not self.graph_manager:
                logger.warning("Graph manager not available")
                return ""
            
            # Определяем тип узла
            node_type = self.entity_type_mapping.get(entity.entity_type, "concept")
            
            # Создаем узел (используем рабочий способ из memory_orchestrator)
            node_id = await self.graph_manager.add_node(
                name=entity.name,
                user_id=user_id,
                node_type=node_type,
                importance=entity.confidence
            )
            
            return node_id
            
        except Exception as e:
            logger.error(f"Failed to add entity to graph: {e}")
            return ""
    
    async def build_graph_from_text(self, text: str, user_id: str, 
                                  extraction_result: EntityExtractionResult,
                                  importance: float = 0.5) -> GraphBuildResult:
        """
        Построение графа из результата извлечения сущностей
        
        Args:
            text: Исходный текст
            user_id: ID пользователя
            extraction_result: Результат извлечения сущностей
            importance: Важность создаваемых узлов
            
        Returns:
            Результат построения графа
        """
        try:
            logger.info(f"Building graph from {len(extraction_result.entities)} entities and {len(extraction_result.relationships)} relationships")
            
            result = GraphBuildResult(
                nodes_added=0,
                edges_added=0,
                nodes_updated=0,
                edges_updated=0,
                confidence=extraction_result.confidence,
                errors=[]
            )
            
            # 1. Создаем/обновляем узлы
            node_mapping = await self._create_or_update_nodes(
                extraction_result.entities, user_id, importance, result
            )
            
            # 2. Создаем/обновляем связи
            await self._create_or_update_edges(
                extraction_result.relationships, node_mapping, user_id, result
            )
            
            logger.info(f"Graph built: {result.nodes_added} nodes added, {result.edges_added} edges added")
            return result
            
        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            result = GraphBuildResult(
                nodes_added=0, edges_added=0, nodes_updated=0, edges_updated=0,
                confidence=0.0, errors=[str(e)]
            )
            return result
    
    async def _create_or_update_nodes(self, entities: List[ExtractedEntity], 
                                    user_id: str, importance: float,
                                    result: GraphBuildResult) -> Dict[str, str]:
        """
        Создание или обновление узлов
        
        Returns:
            Маппинг имя_сущности -> node_id
        """
        node_mapping = {}
        
        for entity in entities:
            try:
                # Определяем тип узла
                node_type = self.entity_type_mapping.get(entity.entity_type, 'concept')
                
                # Создаем узел напрямую (используем рабочий способ)
                node_id = await self.graph_manager.add_node(
                    name=entity.name,
                    user_id=user_id,
                    node_type=node_type,
                    importance=entity.confidence
                )
                
                node_mapping[entity.name] = node_id
                result.nodes_added += 1
                
                logger.debug(f"Created new node: {entity.name} (ID: {node_id})")
                    
            except Exception as e:
                error_msg = f"Failed to create/update node {entity.name}: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)
                continue
        
        return node_mapping
    
    def _find_best_node_match(self, entity_name: str, node_mapping: Dict[str, str]) -> Optional[str]:
        """Умный поиск узла по имени сущности"""
        if not entity_name or not node_mapping:
            return None
        
        entity_lower = entity_name.lower()
        logger.debug(f"Searching for entity: '{entity_name}' in {len(node_mapping)} nodes")
        
        # 1. Точное совпадение (уже проверено выше)
        if entity_name in node_mapping:
            logger.debug(f"Exact match found: {entity_name}")
            return node_mapping[entity_name]
        
        # 2. Поиск по ключевым словам
        entity_words = set(entity_lower.split())
        logger.debug(f"Entity words: {entity_words}")
        
        best_match = None
        best_score = 0
        
        for node_name, node_id in node_mapping.items():
            node_lower = node_name.lower()
            node_words = set(node_lower.split())
            
            # Подсчет пересечения слов
            common_words = entity_words.intersection(node_words)
            if common_words:
                score = len(common_words) / max(len(entity_words), len(node_words))
                logger.debug(f"Checking '{node_name}': common_words={common_words}, score={score:.2f}")
                if score > best_score and score > 0.3:  # Минимум 30% совпадения
                    best_match = node_id
                    best_score = score
                    logger.debug(f"Word match: {entity_name} -> {node_name} (score: {score:.2f})")
        
        # 3. Поиск по подстрокам (если нет совпадений по словам)
        if not best_match:
            for node_name, node_id in node_mapping.items():
                node_lower = node_name.lower()
                
                # Проверяем, содержит ли одно имя другое
                if entity_lower in node_lower or node_lower in entity_lower:
                    best_match = node_id
                    logger.debug(f"Substring match: {entity_name} -> {node_name}")
                    break
        
        if best_match:
            logger.debug(f"Found match for '{entity_name}': {best_match}")
        else:
            logger.debug(f"No match found for '{entity_name}'")
        
        return best_match
    
    async def _find_node_in_database(self, entity_name: str, user_id: str) -> Optional[str]:
        """Поиск узла в базе данных по имени сущности"""
        try:
            if not self.graph_manager:
                return None
            
            # Поиск узлов с похожими именами
            nodes = await self.graph_manager.search_nodes(
                user_id=user_id,
                query=entity_name,
                limit=10
            )
            
            if not nodes:
                return None
            
            # Применяем тот же алгоритм поиска
            entity_lower = entity_name.lower()
            entity_words = set(entity_lower.split())
            
            best_match = None
            best_score = 0
            
            for node in nodes:
                node_lower = node.name.lower()
                node_words = set(node_lower.split())
                
                # Подсчет пересечения слов
                common_words = entity_words.intersection(node_words)
                if common_words:
                    score = len(common_words) / max(len(entity_words), len(node_words))
                    if score > best_score and score > 0.3:  # Минимум 30% совпадения
                        best_match = node.id
                        best_score = score
                        logger.debug(f"Database word match: {entity_name} -> {node.name} (score: {score:.2f})")
            
            # Поиск по подстрокам
            if not best_match:
                for node in nodes:
                    node_lower = node.name.lower()
                    if entity_lower in node_lower or node_lower in entity_lower:
                        best_match = node.id
                        logger.debug(f"Database substring match: {entity_name} -> {node.name}")
                        break
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error searching node in database: {e}")
            return None
    
    async def _create_or_update_edges(self, relationships: List[ExtractedRelationship],
                                    node_mapping: Dict[str, str], user_id: str,
                                    result: GraphBuildResult):
        """Создание или обновление связей"""
        
        for relationship in relationships:
            try:
                # Получаем ID узлов с улучшенным поиском
                source_id = node_mapping.get(relationship.source_entity)
                target_id = node_mapping.get(relationship.target_entity)
                
                # Если не найдены точные совпадения, ищем частичные
                if not source_id:
                    source_id = self._find_best_node_match(relationship.source_entity, node_mapping)
                    if source_id:
                        logger.debug(f"Found partial match for source: {relationship.source_entity}")
                    else:
                        # Поиск в базе данных
                        source_id = await self._find_node_in_database(relationship.source_entity, user_id)
                        if source_id:
                            logger.debug(f"Found database match for source: {relationship.source_entity}")
                
                if not target_id:
                    target_id = self._find_best_node_match(relationship.target_entity, node_mapping)
                    if target_id:
                        logger.debug(f"Found partial match for target: {relationship.target_entity}")
                    else:
                        # Поиск в базе данных
                        logger.debug(f"Searching in database for target: {relationship.target_entity}")
                        target_id = await self._find_node_in_database(relationship.target_entity, user_id)
                        if target_id:
                            logger.debug(f"Found database match for target: {relationship.target_entity}")
                        else:
                            logger.debug(f"No database match found for target: {relationship.target_entity}")
                
                if not source_id or not target_id:
                    logger.warning(f"Missing node IDs for relationship: {relationship.source_entity} -> {relationship.target_entity}")
                    continue
                
                # Определяем тип связи
                edge_type = self.relationship_type_mapping.get(
                    relationship.relationship_type, 'related_to'
                )
                
                # Создаем свойства связи
                edge_properties = {
                    'extracted_confidence': relationship.confidence,
                    'extraction_context': relationship.context,
                    'relationship_type': relationship.relationship_type,
                    'source': 'auto_extraction'
                }
                
                # Проверяем, существует ли связь
                existing_edges = await self.graph_manager.get_node_edges(source_id, user_id)
                
                edge_exists = False
                for edge in existing_edges:
                    if (edge.source_id == source_id and edge.target_id == target_id and 
                        edge.relationship_type == edge_type):
                        edge_exists = True
                        break
                
                if edge_exists:
                    # Обновляем существующую связь
                    for edge in existing_edges:
                        if (edge.source_id == source_id and edge.target_id == target_id and 
                            edge.relationship_type == edge_type):
                            await self.graph_manager.update_edge(
                                edge_id=edge.id,
                                user_id=user_id,
                                properties=edge_properties
                            )
                            break
                    
                    result.edges_updated += 1
                    logger.debug(f"Updated existing edge: {relationship.source_entity} -> {relationship.target_entity}")
                    
                else:
                    # Создаем новую связь
                    edge_id = await self.graph_manager.add_edge(
                        source_id=source_id,
                        target_id=target_id,
                        user_id=user_id,
                        relationship_type=edge_type,
                        properties=edge_properties
                    )
                    
                    result.edges_added += 1
                    logger.debug(f"Created new edge: {relationship.source_entity} -> {relationship.target_entity} (ID: {edge_id})")
                    
            except Exception as e:
                error_msg = f"Failed to create/update edge {relationship.source_entity} -> {relationship.target_entity}: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)
    
    async def build_graph_from_memory(self, user_id: str, 
                                    memory_ids: List[str],
                                    importance: float = 0.7) -> GraphBuildResult:
        """
        Построение графа из существующих записей памяти
        
        Args:
            user_id: ID пользователя
            memory_ids: Список ID записей памяти для анализа
            importance: Важность создаваемых узлов
            
        Returns:
            Результат построения графа
        """
        try:
            logger.info(f"Building graph from {len(memory_ids)} memory records")
            
            result = GraphBuildResult(
                nodes_added=0, edges_added=0, nodes_updated=0, edges_updated=0,
                confidence=0.0, errors=[]
            )
            
            # Извлекаем сущности из записей памяти
            try:
                from ..memory_levels.memory_orchestrator import memory_orchestrator
                if memory_orchestrator:
                    # Получаем записи из всех уровней памяти
                    memories = []
                    for level_name in ["working", "episodic", "semantic", "procedural"]:
                        try:
                            level_memories = await memory_orchestrator.get_user_memories(
                                user_id, level_name=level_name, limit=100
                            )
                            memories.extend(level_memories)
                        except Exception as e:
                            logger.warning(f"Failed to get memories from {level_name}: {e}")
                    
                    # Извлекаем сущности из записей
                    for memory in memories:
                        if memory.get("content"):
                            content = memory["content"]
                            memory_id = memory.get("id", "unknown")
                            
                            # Извлекаем сущности из контента
                            entities = await self._extract_entities_from_content(content, user_id)
                            
                            # Добавляем сущности в граф
                            for entity in entities:
                                await self._add_entity_to_graph(entity, user_id)
                                logger.debug(f"Added entity from memory {memory_id}: {entity.name}")
                            
                            logger.info(f"Extracted {len(entities)} entities from memory {memory_id}")
                
            except Exception as e:
                logger.error(f"Failed to extract entities from memory records: {e}")
            
            logger.info(f"Memory-based graph building completed: {result.nodes_added} nodes, {result.edges_added} edges")
            return result
            
        except Exception as e:
            logger.error(f"Memory-based graph building failed: {e}")
            result = GraphBuildResult(
                nodes_added=0, edges_added=0, nodes_updated=0, edges_updated=0,
                confidence=0.0, errors=[str(e)]
            )
            return result
