"""Upstream Evidence Provider Abstraction Package for Agent 5."""

from app.services.upstream.base_provider import (
    UpstreamEvidenceItem,
    UpstreamEvidenceProvider,
    UpstreamEvidenceRequest,
    UpstreamEvidenceResponse,
    UpstreamProviderError,
    UpstreamSourceType,
)
from app.services.upstream.mock_providers import (
    MockAgent2Provider,
    MockAgent3Provider,
)
from app.services.upstream.mock_refusion import (
    MockAgent4RefusionError,
    MockAgent4RefusionService,
)

__all__ = [
    "UpstreamSourceType",
    "UpstreamEvidenceItem",
    "UpstreamEvidenceRequest",
    "UpstreamEvidenceResponse",
    "UpstreamProviderError",
    "UpstreamEvidenceProvider",
    "MockAgent2Provider",
    "MockAgent3Provider",
    "MockAgent4RefusionService",
    "MockAgent4RefusionError",
]
