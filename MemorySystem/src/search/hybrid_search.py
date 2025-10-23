"""
Hybrid Search implementation for AIRI Memory System.
Combines semantic search (ChromaDB) with keyword search (SQLite FTS5).
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import time

from .fts5_search import get_fts5_engine
from typing import TYPE_CHECKING

try:
    from ..monitoring.search_metrics import record_search_metrics
except ImportError:
    from monitoring.search_metrics import record_search_metrics

if TYPE_CHECKING:
    from ..memory_levels.memory_orchestrator import MemoryOrchestrator

class HybridSearchEngine:
    """Hybrid search engine combining semantic and keyword search"""
    
    def __init__(self, memory_orchestrator: Optional["MemoryOrchestrator"] = None):
        """
        Initialize hybrid search engine
        
        Args:
            memory_orchestrator: MemoryOrchestrator instance for semantic search
        """
        self.memory_orchestrator = memory_orchestrator
        self._lock = asyncio.Lock()
        logger.info("HybridSearchEngine initialized")
    
    async def search(self, query: str, user_id: str, limit: int = 10,
                    semantic_weight: float = 0.7, keyword_weight: float = 0.3,
                    memory_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search
        
        Args:
            query: Search query
            user_id: User identifier
            limit: Maximum number of results
            semantic_weight: Weight for semantic search results (0.0-1.0)
            keyword_weight: Weight for keyword search results (0.0-1.0)
            memory_types: Filter by memory types (optional)
            
        Returns:
            List of hybrid search results with combined scores
        """
        try:
            async with self._lock:
                start_time = time.time()
                
                # Validate weights
                if abs(semantic_weight + keyword_weight - 1.0) > 0.01:
                    logger.warning(f"Weights don't sum to 1.0: {semantic_weight} + {keyword_weight}")
                    # Normalize weights
                    total = semantic_weight + keyword_weight
                    semantic_weight /= total
                    keyword_weight /= total
                
                # Perform both searches in parallel
                semantic_task = self._semantic_search(query, user_id, limit * 2, memory_types)
                keyword_task = self._keyword_search(query, user_id, limit * 2, memory_types)
                
                semantic_results, keyword_results = await asyncio.gather(
                    semantic_task, keyword_task, return_exceptions=True
                )
                
                # Handle exceptions
                if isinstance(semantic_results, Exception):
                    logger.error(f"Semantic search failed: {semantic_results}")
                    semantic_results = []
                
                if isinstance(keyword_results, Exception):
                    logger.error(f"Keyword search failed: {keyword_results}")
                    keyword_results = []
                
                # Combine results using Reciprocal Rank Fusion (RRF)
                combined_results = self._combine_results(
                    semantic_results, keyword_results,
                    semantic_weight, keyword_weight, limit
                )
                
                search_time = time.time() - start_time
                duration_ms = search_time * 1000
                logger.info(f"Hybrid search completed in {search_time:.3f}s, "
                           f"returned {len(combined_results)} results")
                
                # Записываем метрики поиска
                try:
                    record_search_metrics(
                        query=query,
                        search_type="hybrid",
                        user_id=user_id,
                        duration_ms=duration_ms,
                        results=combined_results,
                        cache_hit=False,
                        error=None
                    )
                except Exception as metrics_error:
                    logger.warning(f"Failed to record hybrid search metrics: {metrics_error}")
                
                return combined_results
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"HybridSearchEngine.search failed: {e}")
            
            # Записываем метрики ошибки
            try:
                record_search_metrics(
                    query=query,
                    search_type="hybrid",
                    user_id=user_id,
                    duration_ms=duration_ms,
                    results=[],
                    cache_hit=False,
                    error=str(e)
                )
            except Exception as metrics_error:
                logger.warning(f"Failed to record hybrid search error metrics: {metrics_error}")
            
            return []
    
    async def _semantic_search(self, query: str, user_id: str, limit: int,
                              memory_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using MemoryOrchestrator"""
        try:
            if not self.memory_orchestrator:
                logger.warning("MemoryOrchestrator not available for semantic search")
                return []
            
            # Use MemoryOrchestrator's search functionality
            from ..memory_levels import MemoryQuery
            memory_query = MemoryQuery(
                query=query,
                user_id=user_id,
                memory_levels=memory_types,
                limit=limit
            )
            results = await self.memory_orchestrator.search_memory(memory_query)
            
            # Convert to standard format
            formatted_results = []
            if hasattr(results, 'results'):
                # UnifiedMemoryResult format
                for level, memory_result in results.results.items():
                    for i, item in enumerate(memory_result.items):
                        formatted_results.append({
                            "memory_id": getattr(item, 'id', f"{level}_{i}"),
                            "content": getattr(item, 'content', str(item)),
                            "user_id": user_id,
                            "memory_type": level,
                            "importance": getattr(item, 'importance', 0.5),
                            "created_at": getattr(item, 'created_at', 0),
                            "semantic_score": memory_result.relevance_scores[i] if i < len(memory_result.relevance_scores) else 0.0,
                            "search_type": "semantic"
                        })
            else:
                # Fallback for other formats
                for result in results:
                    formatted_results.append({
                        "memory_id": result.get("id", ""),
                        "content": result.get("content", ""),
                        "user_id": user_id,
                        "memory_type": result.get("memory_type", "general"),
                        "importance": result.get("importance", 0.5),
                        "created_at": result.get("created_at", 0),
                        "semantic_score": result.get("similarity", 0.0),
                        "search_type": "semantic"
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _keyword_search(self, query: str, user_id: str, limit: int,
                             memory_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform keyword search using FTS5"""
        try:
            fts5_engine = await get_fts5_engine()
            results = await fts5_engine.search(query, user_id, limit, memory_types)
            
            # Convert FTS5 scores to similarity scores (invert and normalize)
            for result in results:
                # FTS5 BM25 scores are lower for better matches, so we invert
                fts_score = result.get("fts_score", 0.0)
                # Convert to similarity score (0.0-1.0)
                result["keyword_score"] = max(0.0, 1.0 - (fts_score / 10.0))
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _combine_results(self, semantic_results: List[Dict[str, Any]], 
                        keyword_results: List[Dict[str, Any]],
                        semantic_weight: float, keyword_weight: float,
                        limit: int) -> List[Dict[str, Any]]:
        """
        Combine search results using Reciprocal Rank Fusion (RRF)
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            semantic_weight: Weight for semantic results
            keyword_weight: Weight for keyword results
            limit: Maximum number of final results
            
        Returns:
            Combined and ranked results
        """
        # Create memory_id to result mapping
        memory_map = {}
        
        # Add semantic results
        for rank, result in enumerate(semantic_results):
            memory_id = result["memory_id"]
            if memory_id not in memory_map:
                memory_map[memory_id] = result.copy()
                memory_map[memory_id]["semantic_rank"] = rank + 1
                memory_map[memory_id]["keyword_rank"] = None
                memory_map[memory_id]["semantic_score"] = result.get("semantic_score", 0.0)
                memory_map[memory_id]["keyword_score"] = 0.0
            else:
                memory_map[memory_id]["semantic_rank"] = rank + 1
                memory_map[memory_id]["semantic_score"] = result.get("semantic_score", 0.0)
        
        # Add keyword results
        for rank, result in enumerate(keyword_results):
            memory_id = result["memory_id"]
            if memory_id not in memory_map:
                memory_map[memory_id] = result.copy()
                memory_map[memory_id]["semantic_rank"] = None
                memory_map[memory_id]["keyword_rank"] = rank + 1
                memory_map[memory_id]["semantic_score"] = 0.0
                memory_map[memory_id]["keyword_score"] = result.get("keyword_score", 0.0)
            else:
                memory_map[memory_id]["keyword_rank"] = rank + 1
                memory_map[memory_id]["keyword_score"] = result.get("keyword_score", 0.0)
        
        # Calculate RRF scores
        k = 60  # RRF constant
        for memory_id, result in memory_map.items():
            rrf_score = 0.0
            
            # Add semantic RRF contribution
            if result["semantic_rank"] is not None:
                semantic_rrf = 1.0 / (k + result["semantic_rank"])
                rrf_score += semantic_weight * semantic_rrf
            
            # Add keyword RRF contribution
            if result["keyword_rank"] is not None:
                keyword_rrf = 1.0 / (k + result["keyword_rank"])
                rrf_score += keyword_weight * keyword_rrf
            
            result["rrf_score"] = rrf_score
            
            # Calculate weighted similarity score
            weighted_similarity = (
                semantic_weight * result["semantic_score"] +
                keyword_weight * result["keyword_score"]
            )
            result["weighted_similarity"] = weighted_similarity
        
        # Sort by RRF score (higher is better)
        combined_results = list(memory_map.values())
        combined_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # Add metadata
        for result in combined_results:
            result["search_type"] = "hybrid"
            result["semantic_weight"] = semantic_weight
            result["keyword_weight"] = keyword_weight
        
        return combined_results[:limit]
    
    async def index_memory(self, memory_id: str, content: str, user_id: str,
                          memory_type: str = "general", importance: float = 0.5) -> bool:
        """
        Index memory for both semantic and keyword search
        
        Args:
            memory_id: Memory identifier
            content: Text content
            user_id: User identifier
            memory_type: Type of memory
            importance: Importance score
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Index for keyword search
            fts5_engine = await get_fts5_engine()
            keyword_success = await fts5_engine.index_memory(
                memory_id, content, user_id, memory_type, importance
            )
            
            # Note: Semantic indexing is handled by MemoryManager automatically
            # when memories are added through the normal flow
            
            return keyword_success
            
        except Exception as e:
            logger.error(f"HybridSearchEngine.index_memory failed: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Remove memory from both search indexes
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from keyword search
            fts5_engine = await get_fts5_engine()
            keyword_success = await fts5_engine.delete_memory(memory_id)
            
            # Note: Semantic removal is handled by MemoryManager automatically
            
            return keyword_success
            
        except Exception as e:
            logger.error(f"HybridSearchEngine.delete_memory failed: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get hybrid search statistics
        
        Returns:
            Dictionary with search statistics
        """
        try:
            fts5_engine = await get_fts5_engine()
            fts5_stats = await fts5_engine.get_stats()
            
            return {
                "keyword_search": fts5_stats,
                "semantic_search": {
                    "available": self.memory_manager is not None,
                    "status": "active" if self.memory_manager else "inactive"
                },
                "hybrid_search": {
                    "status": "active",
                    "combination_method": "Reciprocal Rank Fusion (RRF)"
                }
            }
            
        except Exception as e:
            logger.error(f"HybridSearchEngine.get_stats failed: {e}")
            return {"error": str(e)}

# Global hybrid search engine instance
_hybrid_engine: Optional[HybridSearchEngine] = None

async def get_hybrid_engine(memory_orchestrator: Optional["MemoryOrchestrator"] = None) -> HybridSearchEngine:
    """Get global hybrid search engine instance"""
    global _hybrid_engine
    if _hybrid_engine is None:
        _hybrid_engine = HybridSearchEngine(memory_orchestrator)
    return _hybrid_engine
