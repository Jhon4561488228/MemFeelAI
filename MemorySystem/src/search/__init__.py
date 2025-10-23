"""
Search module for AIRI Memory System.
Provides various search capabilities including FTS5, hybrid and contextual search.
"""

from .fts5_search import FTS5SearchEngine, get_fts5_engine

# Отложенный импорт для избежания циклических зависимостей
def get_hybrid_engine_lazy():
    """Lazy import для HybridSearchEngine"""
    from .hybrid_search import get_hybrid_engine
    return get_hybrid_engine

def get_hybrid_search_engine_lazy():
    """Lazy import для HybridSearchEngine класса"""
    from .hybrid_search import HybridSearchEngine
    return HybridSearchEngine

def get_contextual_engine_lazy():
    """Lazy import для ContextualSearchEngine"""
    from .contextual_search import get_contextual_engine
    return get_contextual_engine

def get_contextual_search_engine_lazy():
    """Lazy import для ContextualSearchEngine класса"""
    from .contextual_search import ContextualSearchEngine
    return ContextualSearchEngine

def get_graph_engine_lazy():
    """Lazy import для GraphSearchEngine"""
    from .graph_search import get_graph_engine
    return get_graph_engine

def get_graph_search_engine_lazy():
    """Lazy import для GraphSearchEngine класса"""
    from .graph_search import GraphSearchEngine
    return GraphSearchEngine

def get_entity_extractor_lazy():
    """Lazy import для EntityExtractor"""
    from .entity_extractor import EntityExtractor
    return EntityExtractor

def get_graph_builder_lazy():
    """Lazy import для GraphBuilder"""
    from .graph_builder import GraphBuilder
    return GraphBuilder

def get_auto_entity_extractor_lazy():
    """Lazy import для AutoEntityExtractor"""
    from .auto_entity_extractor import AutoEntityExtractor
    return AutoEntityExtractor

def get_semantic_engine_lazy():
    """Lazy import для SemanticSearchEngine"""
    from .semantic_search import get_semantic_engine
    return get_semantic_engine

__all__ = [
    "FTS5SearchEngine",
    "get_fts5_engine",
    "get_hybrid_engine_lazy",
    "get_hybrid_search_engine_lazy",
    "get_contextual_engine_lazy",
    "get_contextual_search_engine_lazy",
    "get_graph_engine_lazy",
    "get_graph_search_engine_lazy",
    "get_entity_extractor_lazy",
    "get_graph_builder_lazy",
    "get_auto_entity_extractor_lazy",
    "get_semantic_engine_lazy"
]
