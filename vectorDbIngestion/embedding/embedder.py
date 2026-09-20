"""
Embedding module using BioBERT for dense vector generation.

Uses sentence-transformers to load BioBERT and generate 768-dimensional
dense embeddings for medical text chunks.
"""

import logging
from typing import List, Dict, Any

from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

# Global model instance (lazy-loaded)
_model = None


def load_model():
    """
    Loads the BioBERT sentence-transformers model into memory if not already loaded.
    """
    global _model
    if _model is None:
        logger.info("Loading BioBERT embedding model: %s", config.EMBEDDING_MODEL)
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info(
            "BioBERT model loaded successfully. Dim=%d",
            _model.get_sentence_embedding_dimension(),
        )
    return _model


def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes a list of chunk dicts, generates BioBERT dense embeddings,
    and returns the chunks augmented with 'dense_vector'.

    Args:
        chunks: List of chunk dictionaries (must contain 'text' key).

    Returns:
        List of augmented chunk dictionaries with 'dense_vector'.
    """
    model = load_model()

    if not chunks:
        return []

    logger.info("Generating embeddings for %d chunks...", len(chunks))

    batch_size = config.EMBEDDING_BATCH_SIZE

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch_chunks = chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch_chunks]

        # Dense vectors via sentence-transformers
        dense_vecs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        for j, chunk in enumerate(batch_chunks):
            chunk["dense_vector"] = dense_vecs[j].tolist()

    return chunks
