"""Unit tests for input validation and preprocessing routines."""

import pytest
from pydantic import ValidationError

from app.fusion import (
    normalize_medical_term,
    preprocess_agent2_output,
    preprocess_agent3_output,
    preprocess_fusion_request,
    validate_agent2_output,
    validate_agent3_output,
)
from app.schemas import (
    Agent2Output,
    Agent3Output,
    FusionRequest,
    KnowledgeGraphEvidence,
    MedicalEvidence,
    RAGEvidence,
)
from tests.test_data import VALID_AGENT2_OUTPUT, VALID_AGENT3_OUTPUT


def test_valid_input_validation():
    """Verify that valid Agent 2 and Agent 3 structures pass validation without raising errors."""
    validate_agent2_output(VALID_AGENT2_OUTPUT)
    validate_agent3_output(VALID_AGENT3_OUTPUT)


def test_empty_agent2_outputs_raises_error():
    """Verify that ValueError is raised if both RAG and KG evidence in Agent 2 output are empty."""
    invalid_agent2 = Agent2Output(rag_evidence=[], kg_evidence=[])
    with pytest.raises(ValueError) as exc:
        validate_agent2_output(invalid_agent2)
    assert "either RAG evidence or Knowledge Graph evidence" in str(exc.value)


def test_empty_literature_rag_raises_error():
    """Verify that empty literature in RAG evidence raises ValueError."""
    rag_empty_lit = RAGEvidence(
        retrieved_literature=[],
        similarity_score=0.8,
        confidence_score=0.8
    )
    invalid_agent2 = Agent2Output(rag_evidence=[rag_empty_lit], kg_evidence=[])
    with pytest.raises(ValueError) as exc:
        validate_agent2_output(invalid_agent2)
    assert "contains no literature items" in str(exc.value)


def test_whitespace_literature_rag_raises_error():
    """Verify that whitespace-only literature items raise ValueError."""
    rag_whitespace = RAGEvidence(
        retrieved_literature=["   ", "Valid literature"],
        similarity_score=0.8,
        confidence_score=0.8
    )
    invalid_agent2 = Agent2Output(rag_evidence=[rag_whitespace], kg_evidence=[])
    with pytest.raises(ValueError) as exc:
        validate_agent2_output(invalid_agent2)
    assert "contains empty or whitespace-only literature items" in str(exc.value)


def test_empty_disease_info_kg_raises_error():
    """Verify that empty disease info in Knowledge Graph evidence raises ValueError."""
    kg_empty_disease = KnowledgeGraphEvidence(
        disease_info="  ",
        relationships=[{"source": "A", "target": "B"}],
        confidence_score=0.9
    )
    invalid_agent2 = Agent2Output(rag_evidence=[], kg_evidence=[kg_empty_disease])
    with pytest.raises(ValueError) as exc:
        validate_agent2_output(invalid_agent2)
    assert "has an empty disease_info field" in str(exc.value)


def test_empty_agent3_outputs_raises_error():
    """Verify that empty medical evidence list in Agent 3 output raises ValueError."""
    invalid_agent3 = Agent3Output(medical_evidence=[])
    with pytest.raises(ValueError) as exc:
        validate_agent3_output(invalid_agent3)
    assert "must contain at least one medical evidence entry" in str(exc.value)


def test_empty_latest_evidence_agent3_raises_error():
    """Verify that empty latest_evidence string raises ValueError."""
    med_empty = MedicalEvidence(
        latest_evidence="   ",
        pubmed_articles=[],
        europe_pmc_articles=[],
        clinical_guidelines=[],
        abstracts=[],
        source_info={}
    )
    invalid_agent3 = Agent3Output(medical_evidence=[med_empty])
    with pytest.raises(ValueError) as exc:
        validate_agent3_output(invalid_agent3)
    assert "has an empty latest_evidence field" in str(exc.value)


def test_invalid_data_types_pydantic():
    """Verify that Pydantic validation catches invalid typing constraints."""
    with pytest.raises(ValidationError):
        RAGEvidence(
            retrieved_literature=["Good info"],
            similarity_score="invalid_float",  # type: ignore
            confidence_score=0.8
        )


def test_preprocessing_trimming_and_deduplication():
    """Verify that preprocessing strips whitespace and removes duplicates preserving order."""
    rag = RAGEvidence(
        retrieved_literature=["   dup  ", "  dup", "unique  "],
        similarity_score=0.9,
        confidence_score=0.8,
        supporting_research_info=["  ref ", "  ref  "]
    )
    agent2 = Agent2Output(rag_evidence=[rag], kg_evidence=[])
    preprocessed = preprocess_agent2_output(agent2)

    lit = preprocessed.rag_evidence[0].retrieved_literature
    assert lit == ["dup", "unique"]

    sup = preprocessed.rag_evidence[0].supporting_research_info
    assert sup == ["ref"]


def test_medical_term_normalization():
    """Verify that medical acronyms are mapped to standard full titles."""
    assert normalize_medical_term("ami") == "Acute Myocardial Infarction"
    assert normalize_medical_term("sob") == "Shortness of Breath"
    assert normalize_medical_term("ecg") == "Electrocardiogram (ECG)"
    assert normalize_medical_term("custom condition") == "Custom Condition"
