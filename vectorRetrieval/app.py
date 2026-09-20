"""
FastAPI application for the Vector Retrieval Service.
Provides a REST endpoint to search the Qdrant medical knowledge base.
"""

import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from search_engine import MedicalSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vector Retrieval Service",
    description="Search the medical knowledge Qdrant vector database using BioBERT embeddings.",
    version="1.0.0",
)

# Initialize search engine at startup (loads BioBERT model)
search_engine: Optional[MedicalSearchEngine] = None


@app.on_event("startup")
def startup():
    """Load the BioBERT model and connect to Qdrant on startup."""
    global search_engine
    logger.info("Starting Vector Retrieval Service...")
    try:
        search_engine = MedicalSearchEngine()
        logger.info("Vector Retrieval Service ready.")
    except Exception as e:
        logger.error("Failed to initialize search engine: %s", e)
        raise


@app.on_event("shutdown")
def shutdown():
    """Close the Qdrant connection on shutdown."""
    if search_engine:
        search_engine.close()


# ── Request / Response Models ──────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Request body for the /search endpoint."""
    query: str = Field(..., description="The search query text", min_length=1)
    top_k: int = Field(5, description="Number of results to return", ge=1, le=50)
    source_type: Optional[str] = Field(None, description="Filter by source type (e.g., 'medical-paper')")
    disease_category: Optional[str] = Field(None, description="Filter by disease category")
    min_year: Optional[int] = Field(None, description="Minimum publication year")
    max_year: Optional[int] = Field(None, description="Maximum publication year")


class SearchResult(BaseModel):
    """A single search result."""
    score: float
    raw_score: Optional[float] = None
    text: str
    source_id: str
    source_type: str
    publication_year: Optional[int] = None
    disease_category: str = ""
    evidence_level: str = ""
    section_title: str = ""
    journal: str = ""


class SearchResponse(BaseModel):
    """Response body for the /search endpoint."""
    query: str
    top_k: int
    total_found: int
    fallback_level_used: int
    results: List[SearchResult]


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check."""
    return {"service": "Vector Retrieval Service", "status": "running"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    Search the medical knowledge base using BioBERT semantic search.

    Uses multi-level fallback:
    - Level 1: Cosine similarity >= 0.45
    - Level 2: Cosine similarity >= 0.30
    - Level 3: Simplified query, no threshold
    """
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    # Build optional filters
    filter_kwargs = {}
    if request.source_type:
        filter_kwargs["source_type"] = request.source_type
    if request.disease_category:
        filter_kwargs["disease_category"] = request.disease_category
    if request.min_year:
        filter_kwargs["min_year"] = request.min_year
    if request.max_year:
        filter_kwargs["max_year"] = request.max_year

    result = search_engine.search_with_fallback(
        query=request.query,
        limit=request.top_k,
        **filter_kwargs,
    )

    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        total_found=result["total_found"],
        fallback_level_used=result["fallback_level_used"],
        results=result["results"],
    )
