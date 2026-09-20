"""
Configuration for the Vector Retrieval Service.
Mirrors the relevant settings from vectorDbIngestion/config.py (read-only access).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# EMBEDDING MODEL (must match what was used during ingestion)
# =============================================================================

EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
DENSE_VECTOR_DIM = 768

# =============================================================================
# QDRANT DATABASE (connects to the existing Docker container)
# =============================================================================

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "medical_knowledge")

# =============================================================================
# SOURCE PRIORITY (Score Re-Ranking Multipliers)
# Must match vectorDbIngestion/config.py to maintain consistency
# =============================================================================

SOURCE_PRIORITY = {
    "clinical-trial":          1.15,
    "medical-paper":           1.10,
    "symptom-disease-dataset": 1.00,
    "preprint":                0.70,
}

# =============================================================================
# SEARCH DEFAULTS
# =============================================================================

DEFAULT_TOP_K = 5
COSINE_THRESHOLD = 0.45
COSINE_FALLBACK_THRESHOLD = 0.30

# =============================================================================
# MEDICAL STOP WORDS (for query simplification in fallback level 3)
# =============================================================================

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
