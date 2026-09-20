"""Live integration tests for GeminiReasoningEngine using real Gemini API.

Skipped automatically if GEMINI_API_KEY environment variable is not present.
"""

import os
import pytest

from app.config import settings
from app.models.input_models import (
    ConflictEvidence,
    EvidencePriority,
    MergedFinding,
    SourceTraceability,
    SupportingEvidence,
    UnifiedClinicalContext,
)
from app.models.output_models import ClinicalDecisionSupportResponse
from app.services.reasoning_engines import EngineExecutionError, GeminiReasoningEngine

# Skip integration tests if no real GEMINI_API_KEY is available
pytestmark = pytest.mark.skipif(
    not settings.gemini_api_key,
    reason="GEMINI_API_KEY environment variable is not set; skipping live Gemini integration test.",
)


def test_live_gemini_reasoning_integration():
    """Perform end-to-end live reasoning request against Gemini 3.6 Flash API."""
    context = UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="ST-elevation myocardial infarction (STEMI) requires emergency primary PCI within 90 minutes",
                sources=["Agent 2 Biomedical RAG", "Agent 3 PubMed"],
            )
        ],
        normalized_medical_terms=["STEMI", "Primary PCI", "Aspirin"],
        supporting_evidence=[
            SupportingEvidence(
                finding="Early primary PCI reduces acute 30-day mortality in STEMI",
                source_attribution="Agent 3 Clinical Guidelines",
            )
        ],
        conflicting_evidence=[
            ConflictEvidence(
                type="Treatment Conflict",
                finding="Fibrinolysis",
                description="Fibrinolysis is contraindicated if high bleeding risk or active GI bleeding is present.",
            )
        ],
        evidence_priority=[
            EvidencePriority(
                finding="STEMI emergency reperfusion therapy",
                priority_score=0.95,
                primary_source="Agent 3 Clinical Guidelines",
            )
        ],
        source_traceability=[
            SourceTraceability(
                finding="STEMI emergency reperfusion therapy",
                sources=["PubMed", "Clinical Guidelines"],
                upstream_agents=["Agent 2", "Agent 3"],
            )
        ],
        fusion_summary="Acute STEMI presentation requiring immediate reperfusion protocol evaluation.",
        confidence_score=0.92,
    )

    engine = GeminiReasoningEngine()
    try:
        response = engine.reason(context)
    except EngineExecutionError as exc:
        pytest.skip(f"Live Gemini API call skipped: {exc}")

    assert isinstance(response, ClinicalDecisionSupportResponse)
    assert response.agent_metadata.agent == "agent_5"
    assert response.agent_metadata.model == settings.gemini_model
    assert response.agent_metadata.reasoning_mode == "primary"
    assert response.agent_metadata.status == "success"
    assert len(response.clinical_assessment.primary_interpretation) > 0
    assert len(response.decision_support.recommended_actions) > 0
