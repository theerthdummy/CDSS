"""Unit tests for Llama 3.1 8B Instruct enhancement module (app/llm.py)."""

import json
from unittest.mock import MagicMock, patch
import httpx
import pytest

from app.config import settings
from app.llm import _clean_json_markdown_fences, enhance_clinical_context_with_llama


def test_clean_json_markdown_fences():
    """Verify that markdown code fences are stripped properly from raw LLM responses."""
    raw_markdown = "```json\n{\n  \"unified_context\": {}\n}\n```"
    cleaned = _clean_json_markdown_fences(raw_markdown)
    assert cleaned == '{\n  "unified_context": {}\n}'

    plain_json = '{"unified_context": {}}'
    assert _clean_json_markdown_fences(plain_json) == plain_json


def test_llm_unconfigured_api_key_returns_none():
    """Verify that unconfigured LLM API key returns None (triggering fallback)."""
    with patch.object(settings, "llm_api_key", "mock_key_or_set_real_api_key"):
        res = enhance_clinical_context_with_llama({"test": "data"}, correlation_id="test-unconfig")
        assert res is None


def test_llm_disabled_setting_returns_none():
    """Verify that setting LLM_ENABLED=False returns None (triggering fallback)."""
    with patch.object(settings, "llm_enabled", False):
        res = enhance_clinical_context_with_llama({"test": "data"}, correlation_id="test-disabled")
        assert res is None


def test_llm_rate_limit_429_returns_none():
    """Verify that 429 rate limit response from LLM provider returns None."""
    mock_response = MagicMock()
    mock_response.status_code = 429

    with patch.object(settings, "llm_api_key", "valid_key_123"), \
         patch("httpx.Client.post", return_value=mock_response):
        
        res = enhance_clinical_context_with_llama({"test": "data"}, correlation_id="test-429")
        assert res is None


def test_llm_timeout_returns_none():
    """Verify that network timeout during LLM call returns None."""
    with patch.object(settings, "llm_api_key", "valid_key_123"), \
         patch("httpx.Client.post", side_effect=httpx.TimeoutException("Timeout")):
        
        res = enhance_clinical_context_with_llama({"test": "data"}, correlation_id="test-timeout")
        assert res is None


def test_llm_empty_response_content_returns_none():
    """Verify that empty response content from LLM returns None."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}

    with patch.object(settings, "llm_api_key", "valid_key_123"), \
         patch("httpx.Client.post", return_value=mock_response):
        
        res = enhance_clinical_context_with_llama({"test": "data"}, correlation_id="test-empty")
        assert res is None


def test_llm_invalid_json_returns_none():
    """Verify that invalid JSON returned by LLM returns None."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Not valid JSON content"}}]
    }

    with patch.object(settings, "llm_api_key", "valid_key_123"), \
         patch("httpx.Client.post", return_value=mock_response):
        
        res = enhance_clinical_context_with_llama({"test": "data"}, correlation_id="test-json-err")
        assert res is None


def test_llm_missing_root_key_returns_none():
    """Verify that JSON missing 'unified_context' root key returns None."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "{\"other_key\": 123}"}}]
    }

    with patch.object(settings, "llm_api_key", "valid_key_123"), \
         patch("httpx.Client.post", return_value=mock_response):
        
        res = enhance_clinical_context_with_llama({"test": "data"}, correlation_id="test-missing-key")
        assert res is None


def test_llm_successful_enhancement():
    """Verify that valid markdown-wrapped JSON with token usage stats and latency timing is returned successfully."""
    expected_context = {
        "unified_context": {
            "merged_findings": [{"finding": "Acute Myocardial Infarction", "sources": ["Knowledge Graph"]}],
            "normalized_medical_terms": ["Acute Myocardial Infarction"],
            "supporting_evidence": [{"finding": "Acute Myocardial Infarction", "source_attribution": "Knowledge Graph"}],
            "conflicting_evidence": [],
            "evidence_priority": [{"finding": "Acute Myocardial Infarction", "priority_score": 0.8, "primary_source": "Knowledge Graph"}],
            "source_traceability": [{"finding": "Acute Myocardial Infarction", "sources": ["Knowledge Graph"], "upstream_agents": ["Agent 2"]}],
            "fusion_summary": "Unified clinical context.",
            "confidence_score": 0.92,
        }
    }

    markdown_content = f"```json\n{json.dumps(expected_context)}\n```"
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": markdown_content}}],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 80,
            "total_tokens": 230
        }
    }

    with patch.object(settings, "llm_api_key", "valid_key_123"), \
         patch("httpx.Client.post", return_value=mock_response):
        
        result = enhance_clinical_context_with_llama({"test": "data"}, correlation_id="test-success")
        assert result is not None
        res, stats = result
        assert res == expected_context
        assert stats["latency_ms"] >= 0.0
        assert stats["token_usage"]["total_tokens"] == 230
