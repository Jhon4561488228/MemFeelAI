from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

class BaseChromaStorage:
    """
    Common base for ChromaDB-backed storages.

    Resolves chroma path from parameters or environment variables and creates a shared client.
    """

    def __init__(self, chromadb_path: Optional[str] = None) -> None:
        if chromadb_path is None:
            data_root = os.getenv("AIRI_DATA_DIR", "./data")
            chromadb_path = os.getenv("CHROMADB_DIR", os.path.join(data_root, "chroma_db"))
        # Keep telemetry disabled for privacy/offline usage
        self.client = PersistentClient(path=chromadb_path, settings=Settings(anonymized_telemetry=False))
        # Используем SentenceTransformerEmbeddingFunction для правильных distances
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer, falling back to DefaultEmbeddingFunction: {e}")
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        return self.client.get_or_create_collection(
            name=name, 
            metadata=metadata,
            embedding_function=self.embedding_function
        )
