"""
Query interface for the medical RAG vector database.
Provides dense vector search with filtering and multi-level fallback.
Uses BioBERT via sentence-transformers for query encoding.
"""

import logging
import re
from typing import Optional

from qdrant_client import QdrantClient, models

import config

logger = logging.getLogger(__name__)


# Common medical stop words to remove during query simplification
MEDICAL_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "ought",
    "with", "for", "from", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over", "about", "against",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "because", "as", "until", "while", "of", "at", "by",
    "to", "in", "on", "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "patient", "presents", "presenting", "presented", "history",
}

# Cosine similarity thresholds for dense-only search
# These replace the RRF thresholds used in hybrid search
COSINE_THRESHOLD = 0.45              # Level 1: strong semantic match
COSINE_FALLBACK_THRESHOLD = 0.30     # Level 2: relaxed semantic match


class MedicalSearchEngine:
    """
    Dense search engine for the medical RAG vector database.
    Uses BioBERT embeddings with cosine similarity, metadata filtering,
    source priority re-ranking, and multi-level fallback strategies.
    """

    def __init__(self, embedder=None):
        """
        Initialize the search engine.

        Args:
            embedder: An optional pre-loaded SentenceTransformer instance.
                      If None, the model will be loaded on first search.
        """
        self.client = QdrantClient(url=config.QDRANT_URL)
        self.embedder = embedder
        self._model_loaded = embedder is not None
        logger.info("Search engine initialized with Qdrant at '%s'", config.QDRANT_URL)

    def _ensure_model_loaded(self):
        """Load the BioBERT model if not already loaded."""
        if not self._model_loaded:
            logger.info("Loading BioBERT model '%s'...", config.EMBEDDING_MODEL)
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
            self._model_loaded = True
            logger.info("BioBERT model loaded successfully.")

    def _encode_query(self, query: str) -> list:
        """
        Encode a query string into a dense vector using BioBERT.

        Args:
            query: The search query text.

        Returns:
            Dense vector as list of floats.
        """
        self._ensure_model_loaded()
        dense_vector = self.embedder.encode(query, convert_to_numpy=True)
        return dense_vector.tolist()

    def _simplify_query(self, query: str) -> str:
        """
        Simplify a query by removing stop words and extracting key medical terms.

        Args:
            query: The original query string.

        Returns:
            Simplified query string.
        """
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        key_terms = [w for w in words if w not in MEDICAL_STOP_WORDS and len(w) > 2]
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
        """
        Build a Qdrant filter from optional criteria.

        Args:
            source_type: Filter by source type (e.g., 'medical-paper')
            disease_category: Filter by disease category
            evidence_level: Filter by evidence level
            min_year: Minimum publication year (inclusive)
            max_year: Maximum publication year (inclusive)

        Returns:
            A Qdrant Filter object, or None if no filters specified.
        """
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
        limit: int = 10,
        source_type: Optional[str] = None,
        disease_category: Optional[str] = None,
        evidence_level: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        Perform a dense vector search using BioBERT embeddings with cosine similarity.

        Args:
            query: The search query text.
            limit: Maximum number of results to return.
            source_type: Optional filter by source type.
            disease_category: Optional filter by disease category.
            evidence_level: Optional filter by evidence level.
            min_year: Optional minimum publication year.
            max_year: Optional maximum publication year.
            score_threshold: Optional minimum cosine similarity threshold.

        Returns:
            List of result dictionaries with 'score', 'text', and metadata.
        """
        dense_vector = self._encode_query(query)

        query_filter = self._build_filter(
            source_type=source_type,
            disease_category=disease_category,
            evidence_level=evidence_level,
            min_year=min_year,
            max_year=max_year,
        )

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

            # --- Source Priority Re-Ranking ---
            for result in formatted:
                multiplier = config.SOURCE_PRIORITY.get(result["source_type"], 1.0)
                result["raw_score"] = result["score"]
                result["score"] = result["score"] * multiplier

            formatted.sort(key=lambda x: x["score"], reverse=True)

            return formatted

        except Exception as e:
            logger.error("Dense search failed: %s", e)
            return []

    def search_with_fallback(
        self,
        query: str,
        limit: int = 10,
        source_type: Optional[str] = None,
        disease_category: Optional[str] = None,
        evidence_level: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> list[dict]:
        """
        Perform a search with multi-level fallback strategy:
        - Level 1: Dense search with cosine threshold >= 0.45
        - Level 2: Lower threshold to 0.30 and retry
        - Level 3: Simplify query (remove stop words) and retry with no threshold

        Args:
            query: The search query text.
            limit: Maximum number of results.
            **kwargs: Optional filters passed to dense_search.

        Returns:
            List of result dictionaries.
        """
        filter_kwargs = dict(
            source_type=source_type,
            disease_category=disease_category,
            evidence_level=evidence_level,
            min_year=min_year,
            max_year=max_year,
        )

        # Level 1: Standard search with cosine threshold
        logger.info("Fallback Level 1: Dense search with cosine threshold %.2f", COSINE_THRESHOLD)
        results = self.dense_search(
            query, limit=limit, score_threshold=COSINE_THRESHOLD, **filter_kwargs
        )
        if results:
            logger.info("Level 1 returned %d results.", len(results))
            return results

        # Level 2: Lower the cosine threshold
        logger.info("Fallback Level 2: Lowering cosine threshold to %.2f", COSINE_FALLBACK_THRESHOLD)
        results = self.dense_search(
            query, limit=limit, score_threshold=COSINE_FALLBACK_THRESHOLD, **filter_kwargs
        )
        if results:
            logger.info("Level 2 returned %d results.", len(results))
            return results

        # Level 3: Simplify query and retry with no threshold
        simplified = self._simplify_query(query)
        logger.info("Fallback Level 3: Simplified query '%s' (no threshold)", simplified)
        results = self.dense_search(
            simplified, limit=limit, score_threshold=None, **filter_kwargs
        )
        logger.info("Level 3 returned %d results.", len(results))
        return results

    def close(self):
        """Close the Qdrant client connection."""
        self.client.close()
        logger.info("Search engine connection closed.")


def run_interactive_cli():
    """
    Launch an interactive CLI for testing queries against the database.
    """
    print("\n" + "=" * 70)
    print("  MEDICAL RAG VECTOR DATABASE - Interactive Search (BioBERT)")
    print("=" * 70)
    print("\nCommands:")
    print("  Type a medical query to search")
    print("  'filter' - toggle filter mode")
    print("  'quit'   - exit\n")

    engine = MedicalSearchEngine()
    use_filters = False
    filter_settings = {}

    try:
        while True:
            query = input("\n🔍 Query: ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                break

            if query.lower() == "filter":
                use_filters = not use_filters
                if use_filters:
                    print("\n--- Filter Configuration ---")
                    min_year = input("  Min publication year (enter to skip): ").strip()
                    source = input("  Source type [medical-paper/clinical-trial/symptom-disease-dataset/preprint] (enter to skip): ").strip()
                    filter_settings = {}
                    if min_year:
                        filter_settings["min_year"] = int(min_year)
                    if source:
                        filter_settings["source_type"] = source
                    print(f"  Filters active: {filter_settings}")
                else:
                    filter_settings = {}
                    print("  Filters disabled.")
                continue

            # Perform search
            print("\nSearching (with multi-level fallback)...")
            results = engine.search_with_fallback(
                query, limit=5, **filter_settings
            )

            if not results:
                print("  No results found.")
                continue

            print(f"\n Found {len(results)} results:\n")
            for i, r in enumerate(results, 1):
                print(f"  [{i}] Score: {r['score']:.4f}")
                print(f"      Source: {r['source_id']} ({r['source_type']})")
                if r.get("journal"):
                    print(f"      Journal: {r['journal']}")
                if r.get("publication_year"):
                    print(f"      Year: {r['publication_year']}")
                if r.get("section_title"):
                    print(f"      Section: {r['section_title']}")
                text = r["text"]
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"      Text: {text}")
                print()

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        engine.close()
        print("Search engine closed.")
