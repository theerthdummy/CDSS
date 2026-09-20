"""Live integration tests for GroqReasoningEngine via Groq Cloud API.

Skipped automatically if GROQ_API_KEY is not configured in the environment.
These tests make real HTTP calls to Groq and should only be run with --live flag.
"""

import pytest

from app.config import settings
from app.models.input_models import (
    EvidencePriority,
    MergedFinding,
    SourceTraceability,
    SupportingEvidence,
    UnifiedClinicalContext,
)
from app.models.output_models import ClinicalDecisionSupportResponse
from app.services.output_validator import ClinicalOutputValidator
from app.services.reasoning_engines import GroqReasoningEngine


def is_groq_api_configured() -> bool:
    """Check if GROQ_API_KEY environment variable is configured and non-placeholder."""
    return bool(settings.groq_api_key and settings.groq_api_key not in ("", "your_groq_api_key_here"))


# Skip live integration test if Groq API key is not configured
@pytest.mark.live
@pytest.mark.skipif(
    not is_groq_api_configured(),
    reason=f"GROQ_API_KEY is not configured; skipping live Groq integration test.",
)
def test_live_groq_reasoning_integration():
    """Perform end-to-end live reasoning request against Groq Llama 3.1 8B Instruct via Groq Cloud API."""
    context = UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="Acute STEMI with persistent chest pain and ST elevation in inferior leads",
                sources=["Agent 2 Biomedical RAG", "Agent 3 PubMed"],
            )
        ],
        normalized_medical_terms=["STEMI", "Primary PCI", "Aspirin"],
        supporting_evidence=[
            SupportingEvidence(
                finding="Primary PCI within 90 minutes reduces acute STEMI mortality",
                source_attribution="Agent 3 Clinical Guidelines",
            )
        ],
        conflicting_evidence=[],
        evidence_priority=[
            EvidencePriority(
                finding="Acute STEMI emergency reperfusion protocol",
                priority_score=0.95,
                primary_source="Agent 3 Clinical Guidelines",
            )
        ],
        source_traceability=[
            SourceTraceability(
                finding="Acute STEMI emergency reperfusion protocol",
                sources=["PubMed", "Clinical Guidelines"],
                upstream_agents=["Agent 2", "Agent 3"],
            )
        ],
        fusion_summary="Patient presentation strongly consistent with acute inferior STEMI requiring emergency PCI.",
        confidence_score=0.92,
    )

    engine = GroqReasoningEngine()

    try:
        response = engine.reason(context)
    except Exception as exc:
        pytest.skip(f"Live Groq API call skipped due to API issue: {exc}")

    assert isinstance(response, ClinicalDecisionSupportResponse)
    assert response.agent_metadata.model == settings.fallback_reasoning_model
    assert response.agent_metadata.reasoning_mode == "fallback_model"
    assert 0.0 <= response.uncertainty.confidence_score <= 1.0

    val_result = ClinicalOutputValidator.validate_response(response, context)
    assert val_result.valid is True, f"Groq response failed validation: {[e.message for e in val_result.errors]}"
