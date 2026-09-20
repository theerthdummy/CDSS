"""Unit test suite for Agent 5 Configuration and Environment Management.

Verifies authoritative Settings validation, safe defaults, threshold bounds, provider validation,
secret masking (Groq + Gemini), and test environment isolation.
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from app.config import Settings, settings


def test_default_configuration_values():
    """Test 1: Verify authoritative default settings match expected baseline configuration."""
    s = Settings.from_env(overrides={"gemini_api_key": "", "groq_api_key": ""})

    assert s.primary_reasoning_provider == "ollama"
    assert s.primary_reasoning_model == "mistral:latest"
    assert s.fallback_reasoning_provider == "deterministic"
    assert s.fallback_reasoning_model == "deterministic-fallback"
    assert s.confidence_threshold == 0.70
    assert s.max_adaptive_retries == 3
    assert s.max_retries == 3
    assert s.api_host == "0.0.0.0"
    assert s.api_port == 8005
    assert s.log_level == "INFO"
    assert s.app_env in ["development", "test", "production"]


def test_gemini_configuration_overrides():
    """Test 2: Verify Gemini API key, model, and timeout configuration overrides."""
    s = Settings.from_env(
        overrides={
            "gemini_api_key": "test_api_key_12345",
            "gemini_model": "gemini-2.5-flash-custom",
            "gemini_timeout_seconds": 12.5,
        }
    )

    assert s.gemini_api_key == "test_api_key_12345"
    assert s.gemini_model == "gemini-2.5-flash-custom"
    assert s.gemini_timeout_seconds == 12.5


def test_groq_configuration_overrides():
    """Test 3: Verify Groq API key, model, and timeout configuration overrides."""
    s = Settings.from_env(
        overrides={
            "groq_api_key": "gsk_test_key_1234",
            "groq_timeout_seconds": 20.0,
            "fallback_reasoning_model": "llama-3.1-8b-instruct",
        }
    )

    assert s.groq_api_key == "gsk_test_key_1234"
    assert s.groq_timeout_seconds == 20.0
    assert s.fallback_reasoning_model == "llama-3.1-8b-instruct"


def test_confidence_threshold_configurable():
    """Test 4: Verify CONFIDENCE_THRESHOLD is dynamically configurable."""
    s1 = Settings.from_env(overrides={"confidence_threshold": 0.80})
    assert s1.confidence_threshold == 0.80

    s2 = Settings.from_env(overrides={"confidence_threshold": 0.60})
    assert s2.confidence_threshold == 0.60


def test_invalid_confidence_threshold_rejected():
    """Test 5: Verify invalid confidence threshold (< 0.0 or > 1.0) raises ValidationError."""
    with pytest.raises(ValidationError):
        Settings.from_env(overrides={"confidence_threshold": -0.1})

    with pytest.raises(ValidationError):
        Settings.from_env(overrides={"confidence_threshold": 1.5})


def test_max_adaptive_retries_configurable():
    """Test 6: Verify MAX_ADAPTIVE_RETRIES is configurable and must be >= 0."""
    s1 = Settings.from_env(overrides={"max_adaptive_retries": 2})
    assert s1.max_adaptive_retries == 2
    assert s1.max_retries == 2

    s2 = Settings.from_env(overrides={"max_adaptive_retries": 0})
    assert s2.max_adaptive_retries == 0

    with pytest.raises(ValidationError):
        Settings.from_env(overrides={"max_adaptive_retries": -1})


def test_timeout_validation():
    """Test 7: Verify timeouts must be positive floats."""
    with pytest.raises(ValidationError):
        Settings.from_env(overrides={"gemini_timeout_seconds": 0.0})

    with pytest.raises(ValidationError):
        Settings.from_env(overrides={"groq_timeout_seconds": -5.0})

    with pytest.raises(ValidationError):
        Settings.from_env(overrides={"upstream_timeout_seconds": 0.0})


def test_unsupported_provider_rejected():
    """Test 8: Verify unsupported reasoning providers are rejected during validation."""
    with pytest.raises(ValidationError) as exc_info:
        Settings.from_env(overrides={"primary_reasoning_provider": "unsupported_llm"})
    assert "Invalid provider" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        Settings.from_env(overrides={"fallback_reasoning_provider": "unsupported_llm"})
    assert "Invalid provider" in str(exc_info.value)

    s = Settings.from_env(overrides={"primary_reasoning_provider": "ollama", "fallback_reasoning_provider": "deterministic"})
    assert s.primary_reasoning_provider == "ollama"
    assert s.fallback_reasoning_provider == "deterministic"


def test_supported_providers_accepted():
    """Test 9: Verify all supported providers are accepted."""
    s_ollama = Settings.from_env(overrides={"primary_reasoning_provider": "ollama"})
    assert s_ollama.primary_reasoning_provider == "ollama"

    s_groq = Settings.from_env(overrides={"fallback_reasoning_provider": "groq"})
    assert s_groq.fallback_reasoning_provider == "groq"

    s_det = Settings.from_env(overrides={"fallback_reasoning_provider": "deterministic"})
    assert s_det.fallback_reasoning_provider == "deterministic"


def test_invalid_log_level_rejected():
    """Test 10: Verify arbitrary invalid log level string is rejected."""
    with pytest.raises(ValidationError):
        Settings.from_env(overrides={"log_level": "INVALID_LEVEL"})


def test_gemini_secret_masking_safety():
    """Test 11: Verify masked_gemini_api_key masks secrets safely without exposing raw string."""
    s_empty = Settings.from_env(overrides={"gemini_api_key": ""})
    assert s_empty.masked_gemini_api_key == "<NOT_SET>"

    s_short = Settings.from_env(overrides={"gemini_api_key": "12345"})
    assert s_short.masked_gemini_api_key == "***MASKED***"

    s_full = Settings.from_env(overrides={"gemini_api_key": "AIzaSySecretKey1234567890"})
    assert s_full.masked_gemini_api_key == "AIza...7890"
    assert "SecretKey" not in s_full.masked_gemini_api_key


def test_groq_secret_masking_safety():
    """Test 12: Verify masked_groq_api_key masks secrets safely without exposing raw string."""
    s_empty = Settings.from_env(overrides={"groq_api_key": ""})
    assert s_empty.masked_groq_api_key == "<NOT_SET>"

    s_short = Settings.from_env(overrides={"groq_api_key": "short"})
    assert s_short.masked_groq_api_key == "***MASKED***"

    s_full = Settings.from_env(overrides={"groq_api_key": "gsk_QsjLPMLDoSecretValue1234"})
    assert s_full.masked_groq_api_key == "gsk_...1234"
    assert "Secret" not in s_full.masked_groq_api_key


def test_env_example_consistency():
    """Test 13: Verify .env.example contains all key settings variables with no secrets."""
    example_path = Path(__file__).parent.parent / ".env.example"
    assert example_path.exists()

    content = example_path.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=" in content
    assert "GEMINI_MODEL=gemini-2.5-flash" in content
    assert "PRIMARY_REASONING_PROVIDER=ollama" in content
    assert "GROQ_API_KEY=" in content
    assert "FALLBACK_REASONING_PROVIDER=deterministic" in content
    assert "FALLBACK_REASONING_MODEL=deterministic-fallback" in content
    assert "CONFIDENCE_THRESHOLD=0.70" in content
    assert "MAX_ADAPTIVE_RETRIES=3" in content
    # Verify no real API key secrets exist in .env.example
    assert "AIzaSy" not in content
    assert "gsk_" not in content
    # Verify Ollama is documented for the offline orchestration path
    assert "OLLAMA_BASE_URL=" in content
    assert "OLLAMA_MODEL=" in content


def test_test_environment_isolation():
    """Test 14: Verify APP_ENV=test can be set for test environment isolation."""
    s_test = Settings.from_env(overrides={"app_env": "test"})
    assert s_test.app_env == "test"


def test_adaptive_optimizer_uses_configured_threshold():
    """Test 15: Verify AdaptiveOptimizer uses configured confidence_threshold parameter without code modifications."""
    from app.models.input_models import MergedFinding, SupportingEvidence, UnifiedClinicalContext
    from app.services.adaptive_optimizer import AdaptiveOptimizer

    context = UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="Chest pain with ST elevation",
                sources=["Agent 2"],
            ),
            MergedFinding(
                finding="Troponin elevated",
                sources=["Agent 2"],
            ),
        ],
        normalized_medical_terms=["STEMI"],
        supporting_evidence=[
            SupportingEvidence(
                finding="Troponin elevated",
                source_attribution="Agent 2",
            )
        ],
        conflicting_evidence=[],
        evidence_priority=[],
        source_traceability=[],
        fusion_summary="Acute STEMI presentation",
        confidence_score=0.75,
    )

    opt_high = AdaptiveOptimizer(confidence_threshold=0.85)
    decision_high = opt_high.evaluate_context(context)
    assert decision_high.status == "needs_more_evidence"

    opt_low = AdaptiveOptimizer(confidence_threshold=0.65)
    decision_low = opt_low.evaluate_context(context)
    assert decision_low.status == "ready_for_reasoning"
