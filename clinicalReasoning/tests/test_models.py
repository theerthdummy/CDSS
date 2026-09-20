"""Unit tests for Agent 5 input/output Pydantic models and FastAPI health check."""

import pytest
from pydantic import ValidationError
from app.main import app, health_check
from app.models.input_models import (
    ConflictEvidence,
    EvidencePriority,
    MergedFinding,
    SourceTraceability,
    SupportingEvidence,
    UnifiedClinicalContext,
)
from app.models.output_models import (
    AgentMetadata,
    ClinicalAssessment,
    ClinicalDecisionSupportResponse,
    ClinicalReasoningDetail,
    DecisionSupport,
    SafetyAnalysis,
    TraceabilityLink,
    UncertaintyAssessment,
)


@pytest.fixture
def valid_agent4_unified_context_dict():
    """Returns a dictionary matching actual Agent 4 UnifiedClinicalContext structure."""
    return {
        "merged_findings": [
            {
                "finding": "ST-elevation myocardial infarction (STEMI) requires immediate emergency reperfusion therapy",
                "sources": ["Agent 2 Biomedical RAG", "Agent 3 PubMed"],
            }
        ],
        "normalized_medical_terms": ["STEMI", "Reperfusion Therapy", "Aspirin"],
        "supporting_evidence": [
            {
                "finding": "STEMI requires immediate emergency reperfusion therapy via primary PCI or fibrinolysis",
                "source_attribution": "Agent 2 Biomedical RAG",
            }
        ],
        "conflicting_evidence": [
            {
                "type": "Treatment Conflict",
                "finding": "Fibrinolysis",
                "description": "Fibrinolysis is indicated for STEMI within 12 hours, but contraindicated if active internal bleeding is present.",
            }
        ],
        "evidence_priority": [
            {
                "finding": "ST-elevation myocardial infarction",
                "priority_score": 0.95,
                "primary_source": "Agent 3 Clinical Guidelines",
            }
        ],
        "source_traceability": [
            {
                "finding": "ST-elevation myocardial infarction",
                "sources": ["PubMed", "Clinical Guidelines"],
                "upstream_agents": ["Agent 2", "Agent 3"],
            }
        ],
        "fusion_summary": "Patient presentation and evidence indicate acute STEMI requiring emergency reperfusion.",
        "confidence_score": 0.92,
    }


def test_parse_valid_unified_clinical_context(valid_agent4_unified_context_dict):
    """Verify that a valid Agent 4 UnifiedClinicalContext payload is parsed accurately."""
    context = UnifiedClinicalContext(**valid_agent4_unified_context_dict)
    assert len(context.merged_findings) == 1
    assert context.merged_findings[0].finding.startswith("ST-elevation myocardial infarction")
    assert context.normalized_medical_terms == ["STEMI", "Reperfusion Therapy", "Aspirin"]
    assert len(context.supporting_evidence) == 1
    assert context.supporting_evidence[0].source_attribution == "Agent 2 Biomedical RAG"
    assert len(context.conflicting_evidence) == 1
    assert context.conflicting_evidence[0].type == "Treatment Conflict"
    assert context.evidence_priority[0].priority_score == 0.95
    assert context.source_traceability[0].upstream_agents == ["Agent 2", "Agent 3"]
    assert context.confidence_score == 0.92


def test_unified_clinical_context_missing_required_fields():
    """Verify that missing required fields raise Pydantic ValidationError."""
    incomplete_dict = {
        "merged_findings": [],
        "normalized_medical_terms": [],
        # missing supporting_evidence, conflicting_evidence, etc.
    }
    with pytest.raises(ValidationError) as exc_info:
        UnifiedClinicalContext(**incomplete_dict)
    errors = exc_info.value.errors()
    missing_fields = {err["loc"][0] for err in errors}
    assert "supporting_evidence" in missing_fields
    assert "conflicting_evidence" in missing_fields
    assert "evidence_priority" in missing_fields
    assert "source_traceability" in missing_fields
    assert "fusion_summary" in missing_fields
    assert "confidence_score" in missing_fields


def test_unified_clinical_context_invalid_confidence_score(valid_agent4_unified_context_dict):
    """Verify that confidence scores out of range [0.0, 1.0] are rejected."""
    # Test confidence score > 1.0
    invalid_dict_high = dict(valid_agent4_unified_context_dict, confidence_score=1.5)
    with pytest.raises(ValidationError):
        UnifiedClinicalContext(**invalid_dict_high)

    # Test confidence score < 0.0
    invalid_dict_low = dict(valid_agent4_unified_context_dict, confidence_score=-0.1)
    with pytest.raises(ValidationError):
        UnifiedClinicalContext(**invalid_dict_low)


def test_nested_agent4_models():
    """Verify individual nested Agent 4 submodels can be instantiated and validated."""
    mf = MergedFinding(finding="Chest pain", sources=["Agent 2"])
    assert mf.finding == "Chest pain"
    assert mf.sources == ["Agent 2"]

    se = SupportingEvidence(finding="Elevated Troponin", source_attribution="Lab Data")
    assert se.source_attribution == "Lab Data"

    ce = ConflictEvidence(type="Diagnosis Conflict", finding="Angina", description="Stable vs Unstable")
    assert ce.type == "Diagnosis Conflict"

    ep = EvidencePriority(finding="ECG ST Elevation", priority_score=0.9, primary_source="ECG")
    assert ep.priority_score == 0.9

    st = SourceTraceability(finding="STEMI", sources=["ECG"], upstream_agents=["Agent 2"])
    assert st.upstream_agents == ["Agent 2"]


def test_valid_clinical_decision_support_response_instantiation():
    """Verify that a valid Agent 5 response model can be instantiated and serialized."""
    response = ClinicalDecisionSupportResponse(
        clinical_assessment=ClinicalAssessment(
            primary_interpretation="Acute myocardial infarction presentation.",
            clinical_significance="Critical",
            differential_considerations=["STEMI", "Aortic Dissection"],
        ),
        reasoning=ClinicalReasoningDetail(
            key_findings=["ST Elevation on ECG", "Elevated Troponin"],
            supporting_factors=["Agent 2 RAG findings confirm reperfusion protocol"],
            conflicting_factors=["Bleeding risk warning for thrombolytics"],
            reasoning_summary="Evidence strongly supports STEMI protocol.",
        ),
        decision_support=DecisionSupport(
            recommended_actions=["Immediate Cardiology Consult", "Prepare PCI Lab"],
            additional_information_needed=["Troponin trend", "Baseline renal function"],
            priority_level="Urgent",
        ),
        safety=SafetyAnalysis(
            safety_flags=["High Risk: Thrombolysis bleeding warning"],
            contraindications_or_concerns=["Active GI bleeding screening required"],
        ),
        uncertainty=UncertaintyAssessment(
            confidence_score=0.88,
            uncertainty_factors=["Missing serial cardiac enzymes"],
        ),
        traceability=[
            TraceabilityLink(
                reasoning_item="Immediate Cardiology Consult",
                supported_by_findings=["ST-elevation myocardial infarction requires immediate emergency reperfusion"],
                upstream_sources=["Agent 2 Biomedical RAG", "Agent 3 PubMed"],
            )
        ],
        agent_metadata=AgentMetadata(
            agent="agent_5",
            model="Gemini 3.6 Flash",
            reasoning_mode="gemini_enhanced",
            status="success",
        ),
    )

    # Verify attributes
    assert response.clinical_assessment.clinical_significance == "Critical"
    assert response.decision_support.priority_level == "Urgent"
    assert response.agent_metadata.reasoning_mode == "gemini_enhanced"
    assert response.uncertainty.confidence_score == 0.88

    # Verify serialization
    dumped_dict = response.model_dump()
    assert dumped_dict["agent_metadata"]["agent"] == "agent_5"
    assert dumped_dict["safety"]["safety_flags"] == ["High Risk: Thrombolysis bleeding warning"]

    dumped_json = response.model_dump_json()
    assert '"clinical_significance":"Critical"' in dumped_json


def test_agent5_response_missing_required_fields():
    """Verify that incomplete response models raise Pydantic ValidationError."""
    with pytest.raises(ValidationError):
        ClinicalDecisionSupportResponse(
            clinical_assessment=ClinicalAssessment(
                primary_interpretation="Incomplete interpretation",
                clinical_significance="High",
                differential_considerations=[],
            )
            # missing reasoning, decision_support, safety, uncertainty, traceability, agent_metadata
        )


def test_fastapi_health_endpoint():
    """Verify that the FastAPI health check endpoint returns expected status."""
    response = health_check()
    assert response == {
        "status": "ok",
        "agent": "agent_5",
    }

