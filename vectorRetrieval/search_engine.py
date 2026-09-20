"""
Medical Search Engine — Dense vector search with multi-level fallback.

Connects to the existing Qdrant vector database (populated by vectorDbIngestion)
and provides semantic search using BioBERT embeddings with source-priority re-ranking.
"""

import logging
import re
from typing import Optional, List, Dict, Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


class MedicalSearchEngine:
    """
    Dense search engine for the medical RAG vector database.
    Uses BioBERT embeddings with cosine similarity, metadata filtering,
    source priority re-ranking, and multi-level fallback strategies.
    """

    def __init__(self):
        """Initialize the search engine with Qdrant client and BioBERT model."""
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY if config.QDRANT_API_KEY else None,
        )
        logger.info("Loading BioBERT model '%s'...", config.EMBEDDING_MODEL)
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("BioBERT model loaded. Connected to Qdrant at '%s'.", config.QDRANT_URL)

    def _encode_query(self, query: str) -> list:
        """Encode a query string into a dense vector using BioBERT."""
        dense_vector = self.embedder.encode(query, convert_to_numpy=True)
        return dense_vector.tolist()

    def _simplify_query(self, query: str) -> str:
        """Simplify a query by removing stop words and extracting key medical terms."""
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        key_terms = [w for w in words if w not in config.MEDICAL_STOP_WORDS and len(w) > 2]
        simplified = " ".join(key_terms)
        logger.debug("Simplified query: '%s' -> '%s'", query, simplified)
        return simplified if simplified else query

    def _build_filter(
        self,
        source_type: Optional[str] = None,
        disease_category: Optional[str] = None,
        evidence_level: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> Optional[models.Filter]:
        """Build a Qdrant filter from optional criteria."""
        conditions = []

        if source_type:
            conditions.append(
                models.FieldCondition(
                    key="source_type",
                    match=models.MatchValue(value=source_type),
                )
            )

        if disease_category:
            conditions.append(
                models.FieldCondition(
                    key="disease_category",
                    match=models.MatchValue(value=disease_category),
                )
            )

        if evidence_level:
            conditions.append(
                models.FieldCondition(
                    key="evidence_level",
                    match=models.MatchValue(value=evidence_level),
                )
            )

        if min_year is not None or max_year is not None:
            range_params = {}
            if min_year is not None:
                range_params["gte"] = min_year
            if max_year is not None:
                range_params["lte"] = max_year
            conditions.append(
                models.FieldCondition(
                    key="publication_year",
                    range=models.Range(**range_params),
                )
            )

        if conditions:
            return models.Filter(must=conditions)
        return None

    def dense_search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        **filter_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Perform a dense vector search using BioBERT embeddings with cosine similarity.

        Args:
            query: The search query text.
            limit: Maximum number of results to return.
            score_threshold: Optional minimum cosine similarity threshold.
            **filter_kwargs: Optional filters (source_type, disease_category, etc.)

        Returns:
            List of result dictionaries with 'score', 'text', and metadata.
        """
        dense_vector = self._encode_query(query)
        query_filter = self._build_filter(**filter_kwargs)

        try:
            results = self.client.query_points(
                collection_name=config.COLLECTION_NAME,
                query=dense_vector,
                using="dense",
                query_filter=query_filter,
                limit=limit,
            )

            formatted = []
            for point in results.points:
                result = {
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "source_id": point.payload.get("source_id", ""),
                    "source_type": point.payload.get("source_type", ""),
                    "publication_year": point.payload.get("publication_year"),
                    "disease_category": point.payload.get("disease_category", ""),
                    "evidence_level": point.payload.get("evidence_level", ""),
                    "section_title": point.payload.get("section_title", ""),
                    "journal": point.payload.get("journal", ""),
                }

                if score_threshold is None or result["score"] >= score_threshold:
                    formatted.append(result)

            # Source Priority Re-Ranking
            for result in formatted:
                multiplier = config.SOURCE_PRIORITY.get(result["source_type"], 1.0)
                result["raw_score"] = result["score"]
                result["score"] = round(result["score"] * multiplier, 6)

            formatted.sort(key=lambda x: x["score"], reverse=True)
            return formatted

        except Exception as e:
            logger.error("Dense search failed: %s", e)
            return []

    def search_with_fallback(
        self,
        query: str,
        limit: int = 5,
        **filter_kwargs,
    ) -> Dict[str, Any]:
        """
        Perform a search with multi-level fallback strategy:
        - Level 1: Dense search with cosine threshold >= 0.45
        - Level 2: Lower threshold to 0.30 and retry
        - Level 3: Simplify query (remove stop words) and retry with no threshold

        Returns:
            Dict with 'results', 'fallback_level_used', and 'total_found'.
        """
        # Level 1: Standard search with cosine threshold
        logger.info("Fallback Level 1: cosine threshold %.2f", config.COSINE_THRESHOLD)
        results = self.dense_search(
            query, limit=limit, score_threshold=config.COSINE_THRESHOLD, **filter_kwargs
        )
        if results:
            return {"results": results, "fallback_level_used": 1, "total_found": len(results)}

        # Level 2: Lower the cosine threshold
        logger.info("Fallback Level 2: cosine threshold %.2f", config.COSINE_FALLBACK_THRESHOLD)
        results = self.dense_search(
            query, limit=limit, score_threshold=config.COSINE_FALLBACK_THRESHOLD, **filter_kwargs
        )
        if results:
            return {"results": results, "fallback_level_used": 2, "total_found": len(results)}

        # Level 3: Simplify query and retry with no threshold
        simplified = self._simplify_query(query)
        logger.info("Fallback Level 3: simplified query '%s' (no threshold)", simplified)
        results = self.dense_search(
            simplified, limit=limit, score_threshold=None, **filter_kwargs
        )
        return {"results": results, "fallback_level_used": 3, "total_found": len(results)}

    def close(self):
        """Close the Qdrant client connection."""
        self.client.close()
        logger.info("Search engine connection closed.")
