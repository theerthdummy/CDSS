"""Unit tests for the Hybrid Clinical Data Fusion Engine (Rule-Based + Llama 3.1 8B Instruct)."""

import time
from unittest.mock import patch
import pytest

from app.fusion import (
    fuse_clinical_data,
    normalize_medical_term,
    prepare_llm_fusion_input,
    preprocess_fusion_request,
    rule_based_fusion,
)
from app.schemas import FusionRequest
from tests.test_data import VALID_FUSION_REQUEST_DICT, VALID_LLM_FUSION_RESULT


def test_normalization():
    """Verify that common medical terms and acronyms are mapped correctly, and others title-cased."""
    assert normalize_medical_term("ami") == "Acute Myocardial Infarction"
    assert normalize_medical_term("sob") == "Shortness of Breath"
    assert normalize_medical_term("ecg") == "Electrocardiogram (ECG)"
    assert normalize_medical_term("new condition") == "New Condition"


def test_rule_based_fusion_execution_and_timing():
    """Verify that rule-based baseline fusion executes deterministically, extracts categories, tags origins, and measures performance."""
    request = FusionRequest.model_validate(VALID_FUSION_REQUEST_DICT)
    preprocessed = preprocess_fusion_request(request)

    start_time = time.perf_counter()
    context, conflicts, trace_origins = rule_based_fusion(preprocessed)
    duration_ms = (time.perf_counter() - start_time) * 1000

    # Rule-based fusion should execute almost instantaneously (under 50ms)
    assert duration_ms < 50.0

    # Context category assertions
    assert "Acute Myocardial Infarction" in context.normalized_medical_terms
    assert "Shortness of Breath" in context.normalized_medical_terms
    assert "Electrocardiogram (ECG)" in context.normalized_medical_terms
    assert "Aspirin" in context.normalized_medical_terms
    assert len(context.merged_findings) > 0

    # Source attribution assertions
    assert "Knowledge Graph" in trace_origins["Aspirin"]
    assert "Latest Medical Evidence" in trace_origins["Aspirin"]

    # Conflict detection assertions
    assert len(conflicts) > 0
    assert any(c.finding == "Aspirin" for c in conflicts)


def test_prepare_llm_fusion_input():
    """Verify that input preparation formats preprocessed payload into JSON dict for Llama."""
    request = FusionRequest.model_validate(VALID_FUSION_REQUEST_DICT)
    preprocessed = preprocess_fusion_request(request)
    prepared_input = prepare_llm_fusion_input(preprocessed)

    assert "agent2_output" in prepared_input
    assert "agent3_output" in prepared_input


@patch("app.fusion.enhance_clinical_context_with_llama")
def test_hybrid_fusion_llama_enhanced_success(mock_llm_func):
    """Verify that when LLM enhancement succeeds, fusion mode is 'llama_enhanced'."""
    mock_stats = {"latency_ms": 45.2, "token_usage": {"total_tokens": 200}}
    mock_llm_func.return_value = (VALID_LLM_FUSION_RESULT, mock_stats)

    request = FusionRequest.model_validate(VALID_FUSION_REQUEST_DICT)
    preprocessed = preprocess_fusion_request(request)
    context, conflicts, trace_origins, mode, stats, fallback_reason = fuse_clinical_data(preprocessed, correlation_id="test-hybrid")

    mock_llm_func.assert_called_once()
    assert mode == "llama_enhanced"
    assert fallback_reason is None
    assert "Acute Myocardial Infarction" in context.normalized_medical_terms
    assert context.confidence_score == 0.95
    assert stats == mock_stats


@patch("app.fusion.enhance_clinical_context_with_llama")
def test_hybrid_fusion_fallback_on_llm_failure(mock_llm_func):
    """Verify that when LLM enhancement fails (returns None), fusion engine falls back to rule_based_fallback."""
    mock_llm_func.return_value = None

    request = FusionRequest.model_validate(VALID_FUSION_REQUEST_DICT)
    preprocessed = preprocess_fusion_request(request)
    context, conflicts, trace_origins, mode, stats, fallback_reason = fuse_clinical_data(preprocessed, correlation_id="test-fallback")

    assert mode == "rule_based_fallback"
    assert fallback_reason is not None
    assert "Acute Myocardial Infarction" in context.normalized_medical_terms
    assert len(conflicts) > 0
    assert stats == {}


@patch("app.fusion.enhance_clinical_context_with_llama")
def test_hybrid_fusion_fallback_on_schema_validation_failure(mock_llm_func):
    """Verify that when LLM returns JSON failing Pydantic schema validation, system falls back to rule-based fusion."""
    invalid_result = {
        "unified_context": {
            "normalized_medical_terms": ["Acute Myocardial Infarction"]
            # Missing required fields like merged_findings, confidence_score, etc.
        }
    }
    mock_llm_func.return_value = (invalid_result, {"latency_ms": 10.0})

    request = FusionRequest.model_validate(VALID_FUSION_REQUEST_DICT)
    preprocessed = preprocess_fusion_request(request)
    context, conflicts, trace_origins, mode, stats, fallback_reason = fuse_clinical_data(preprocessed, correlation_id="test-schema-fail")

    assert mode == "rule_based_fallback"
    assert "Pydantic schema validation" in fallback_reason
    assert "Acute Myocardial Infarction" in context.normalized_medical_terms
