"""Upstream Evidence Provider Abstraction Layer for CDSS Agent 5.

Defines the abstract interface, routing identifiers, and request/response data contracts
for interacting with future upstream evidence providers (Agent 2, Agent 3).
"""

from abc import ABC, abstractmethod
from enum import Enum
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UpstreamSourceType(str, Enum):
    """Routing identifiers for future upstream evidence providers.
    
    Marked explicitly as FUTURE_UPSTREAM_AGENT to indicate these are routing interfaces,
    not current implementations of Agent 2 or Agent 3.
    """

    FUTURE_UPSTREAM_AGENT_2 = "FUTURE_UPSTREAM_AGENT_2"
    FUTURE_UPSTREAM_AGENT_3 = "FUTURE_UPSTREAM_AGENT_3"


class UpstreamEvidenceItem(BaseModel):
    """Structured evidence item returned by an upstream evidence provider."""

    evidence_id: str = Field(..., description="Unique synthetic or source identifier for evidence item.")
    finding: str = Field(..., description="Clinical finding or text snippet.")
    source: str = Field(..., description="Source attribution name (e.g. Biomedical RAG, PubMed).")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance or confidence score.")
    content: str = Field(..., description="Detailed content or summary of evidence.")
    source_type: str = Field(..., description="Category of evidence source.")


class UpstreamEvidenceRequest(BaseModel):
    """Structured request sent to an upstream evidence provider."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request ID.")
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Traceable correlation ID.")
    source_type: UpstreamSourceType = Field(..., description="Target upstream evidence provider source.")
    query_or_terms: List[str] = Field(default_factory=list, description="Search queries or normalized medical terms.")
    evidence_gaps: List[str] = Field(default_factory=list, description="Specific evidence gaps to fill.")
    retry_count: int = Field(default=0, ge=0, description="Current adaptive loop retry count.")
    agent_2_details: Optional[Dict[str, Any]] = Field(None, description="Optional Agent 2 specific parameters.")
    agent_3_details: Optional[Dict[str, Any]] = Field(None, description="Optional Agent 3 specific parameters.")


class UpstreamEvidenceResponse(BaseModel):
    """Structured response returned by an upstream evidence provider."""

    request_id: str = Field(..., description="Matching request ID.")
    correlation_id: str = Field(..., description="Matching correlation ID.")
    source_type: UpstreamSourceType = Field(..., description="Responding provider source type.")
    evidence_items: List[UpstreamEvidenceItem] = Field(default_factory=list, description="Retrieved evidence items.")
    status: str = Field(default="success", description="Status string ('success', 'error').")
    error_message: Optional[str] = Field(None, description="Error message if retrieval failed.")


class UpstreamProviderError(Exception):
    """Exception raised when an upstream evidence provider fails to retrieve evidence."""

    pass


class UpstreamEvidenceProvider(ABC):
    """Abstract Base Class / Interface for upstream evidence providers.
    
    Future real implementations (e.g., Agent2HTTPProvider, Agent3HTTPProvider)
    must implement this interface to integrate seamlessly with Agent 5's AdaptiveOrchestrator.
    """

    @abstractmethod
    def retrieve(self, request: UpstreamEvidenceRequest) -> UpstreamEvidenceResponse:
        """Retrieve additional evidence based on a structured request.
        
        Args:
            request: Structured UpstreamEvidenceRequest payload.
            
        Returns:
            UpstreamEvidenceResponse containing retrieved evidence items.
            
        Raises:
            UpstreamProviderError: If retrieval fails.
        """
        pass

    @abstractmethod
    def get_source_type(self) -> UpstreamSourceType:
        """Return the source type identifier handled by this provider."""
        pass
