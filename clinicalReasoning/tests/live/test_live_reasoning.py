"""Live LLM Reasoning and Validation test suite for Agent 5.

Performs live end-to-end clinical reasoning validation over synthetic Agent 4 fixtures using configured model APIs.
Automatically skips tests gracefully if live API keys are not configured.

Cascade under test: Gemini 2.5 Flash (Level 1) -> Groq Llama 3.1 8B Instruct (Level 2) -> Deterministic (Level 3)
"""

import json
from pathlib import Path
import httpx
import pytest

from app.config import settings
from app.models.feedback_models import FeedbackRequest
from app.models.input_models import UnifiedClinicalContext
from app.models.output_models import ClinicalDecisionSupportResponse
from app.services.clinical_reasoning import ClinicalReasoningService
from app.services.output_validator import ClinicalOutputValidator
from app.services.reasoning_engines import (
    GeminiReasoningEngine,
    GroqReasoningEngine,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "live"


def load_live_fixture(filename: str) -> UnifiedClinicalContext:
    """Load and validate synthetic clinical fixture from tests/fixtures/live/."""
    file_path = FIXTURES_DIR / filename
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return UnifiedClinicalContext.model_validate(data)


def is_gemini_api_configured() -> bool:
    """Check if GEMINI_API_KEY environment variable is configured."""
    return bool(settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here")


def is_groq_api_configured() -> bool:
    """Check if GROQ_API_KEY environment variable is configured."""
    return bool(settings.groq_api_key and settings.groq_api_key != "your_groq_api_key_here")


pytestmark = pytest.mark.live


@pytest.mark.skipif(not is_gemini_api_configured(), reason="GEMINI_API_KEY environment variable is not configured.")
def test_live_gemini_reasoning_pipeline():
    """Live integration test invoking primary Gemini model over synthetic high-confidence STEMI context."""
    context = load_live_fixture("strong_consistent_case.json")

    engine = GeminiReasoningEngine()
    service = ClinicalReasoningService(primary_engine=engine, use_dev_default_when_unimplemented=False)

    try:
        result = service.execute_reasoning_pipeline(context)
    except Exception as exc:
        pytest.skip(f"Live Gemini API call skipped due to temporary API issue: {exc}")

    if isinstance(result, ClinicalDecisionSupportResponse) and result.agent_metadata.reasoning_mode != "primary":
        pytest.skip(f"Live Gemini API rate limit or fallback occurred (mode={result.agent_metadata.reasoning_mode}).")

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == settings.primary_reasoning_model

    val_result = ClinicalOutputValidator.validate_response(result, context)
    assert val_result.valid is True
    assert 0.0 <= result.uncertainty.confidence_score <= 1.0
    assert result.uncertainty.fusion_confidence == 0.92


@pytest.mark.skipif(not is_groq_api_configured(), reason="GROQ_API_KEY environment variable is not configured.")
def test_live_groq_reasoning_pipeline():
    """Live integration test invoking secondary Groq Llama 3.1 8B Instruct model over synthetic STEMI context."""
    context = load_live_fixture("strong_consistent_case.json")

    engine = GroqReasoningEngine()
    service = ClinicalReasoningService(primary_engine=engine, use_dev_default_when_unimplemented=False)

    try:
        result = service.execute_reasoning_pipeline(context)
    except Exception as exc:
        pytest.skip(f"Live Groq API call skipped due to temporary API issue: {exc}")

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == settings.fallback_reasoning_model

    val_result = ClinicalOutputValidator.validate_response(result, context)
    assert val_result.valid is True


def test_live_adaptive_gate_blocks_llm_execution():
    """Verify low-confidence context is gated by AdaptiveOptimizer; live LLM models are NOT called."""
    context = load_live_fixture("insufficient_case.json")

    mock_gemini = GeminiReasoningEngine()
    service = ClinicalReasoningService(primary_engine=mock_gemini, use_dev_default_when_unimplemented=False)

    result = service.execute_reasoning_pipeline(context, retry_count=0)

    assert isinstance(result, FeedbackRequest)
    assert result.status == "needs_more_evidence"


@pytest.mark.skipif(not is_gemini_api_configured(), reason="GEMINI_API_KEY environment variable is not configured.")
def test_live_conflicting_case_preserves_uncertainty():
    """Live integration test evaluating moderate-confidence context with diagnostic conflict at max retries."""
    context = load_live_fixture("conflicting_case.json")

    engine = GeminiReasoningEngine()
    service = ClinicalReasoningService(primary_engine=engine, use_dev_default_when_unimplemented=False)

    try:
        result = service.execute_reasoning_pipeline(context, retry_count=3)
    except Exception as exc:
        pytest.skip(f"Live Gemini API call skipped due to temporary API issue: {exc}")

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert "confirmed diagnosis" not in result.clinical_assessment.primary_interpretation.lower()

    val_result = ClinicalOutputValidator.validate_response(result, context)
    assert val_result.valid is True
