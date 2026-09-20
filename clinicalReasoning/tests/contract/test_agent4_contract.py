"""Contract fidelity and JSON round-trip tests between Agent 4 output and Agent 5 input schema."""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from app.models.input_models import UnifiedClinicalContext

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_fixture(filename: str) -> dict:
    """Load JSON fixture file from tests/fixtures."""
    file_path = FIXTURES_DIR / filename
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_load_and_deserialize_high_confidence_fixture():
    """Verify loading and deserializing high confidence Agent 4 fixture preserves 100% field fidelity."""
    data = load_fixture("agent4_high_confidence.json")
    context = UnifiedClinicalContext.model_validate(data)

    assert isinstance(context, UnifiedClinicalContext)
    assert context.confidence_score == 0.92
    assert len(context.merged_findings) == 2
    assert context.merged_findings[0].finding == "Acute ST-elevation myocardial infarction (STEMI) requires immediate emergency reperfusion therapy"
    assert context.merged_findings[0].sources == ["Agent 2 Biomedical RAG", "Agent 3 PubMed"]
    assert len(context.normalized_medical_terms) == 5
    assert len(context.supporting_evidence) == 2
    assert len(context.conflicting_evidence) == 0
    assert len(context.evidence_priority) == 2
    assert context.evidence_priority[0].priority_score == 0.95
    assert len(context.source_traceability) == 1
    assert "acute STEMI requiring emergency primary PCI" in context.fusion_summary


def test_load_and_deserialize_low_confidence_fixture():
    """Verify loading and deserializing low confidence Agent 4 fixture preserves conflicts and scores."""
    data = load_fixture("agent4_low_confidence.json")
    context = UnifiedClinicalContext.model_validate(data)

    assert isinstance(context, UnifiedClinicalContext)
    assert context.confidence_score == 0.45
    assert len(context.conflicting_evidence) == 1
    assert context.conflicting_evidence[0].type == "Diagnostic Conflict"
    assert context.conflicting_evidence[0].finding == "Acute Coronary Syndrome vs Musculoskeletal Pain"


def test_json_round_trip_preserves_fidelity():
    """Verify JSON round-trip (Agent 4 JSON -> Pydantic -> JSON string -> Pydantic) maintains exact equivalence."""
    data = load_fixture("agent4_high_confidence.json")
    context_orig = UnifiedClinicalContext.model_validate(data)

    json_str = context_orig.model_dump_json()
    reloaded_dict = json.loads(json_str)
    context_reloaded = UnifiedClinicalContext.model_validate(reloaded_dict)

    assert context_orig == context_reloaded
    assert context_orig.confidence_score == context_reloaded.confidence_score
    assert context_orig.fusion_summary == context_reloaded.fusion_summary


def test_exact_confidence_score_preservation():
    """Verify float precision of confidence_score is never altered or rounded unexpectedly."""
    data = load_fixture("agent4_low_confidence.json")
    data["confidence_score"] = 0.4735
    context = UnifiedClinicalContext.model_validate(data)
    assert context.confidence_score == 0.4735


def test_malformed_agent4_inputs_rejected():
    """Verify Agent 5 rejects malformed Agent 4 payloads safely during input validation."""
    valid_data = load_fixture("agent4_high_confidence.json")

    # Case 1: Missing confidence_score
    data1 = dict(valid_data)
    del data1["confidence_score"]
    with pytest.raises(ValidationError):
        UnifiedClinicalContext.model_validate(data1)

    # Case 2: String confidence_score
    data2 = dict(valid_data, confidence_score="invalid_string")
    with pytest.raises(ValidationError):
        UnifiedClinicalContext.model_validate(data2)

    # Case 3: confidence_score out of range (> 1.0)
    data3 = dict(valid_data, confidence_score=1.5)
    with pytest.raises(ValidationError):
        UnifiedClinicalContext.model_validate(data3)

    # Case 4: confidence_score out of range (< 0.0)
    data4 = dict(valid_data, confidence_score=-0.1)
    with pytest.raises(ValidationError):
        UnifiedClinicalContext.model_validate(data4)

    # Case 5: Missing required list field
    data5 = dict(valid_data)
    del data5["merged_findings"]
    with pytest.raises(ValidationError):
        UnifiedClinicalContext.model_validate(data5)
