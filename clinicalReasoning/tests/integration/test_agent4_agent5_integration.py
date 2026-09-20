"""HTTP Integration test validating real communication between Agent 4 (/api/v1/fuse) and Agent 5 (/api/v1/reason).

Automatically skipped if local Agent 4 or Agent 5 services are not running.
"""

import httpx
import pytest

from app.config import settings

# Helper to check if Agent 4 is running locally
def is_agent4_available() -> bool:
    try:
        resp = httpx.get(f"{settings.agent_4_base_url}/api/v1/", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


# Helper to check if Agent 5 is running locally
def is_agent5_available() -> bool:
    try:
        resp = httpx.get(f"{settings.agent_5_base_url}/health", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (is_agent4_available() and is_agent5_available()),
        reason=f"Live Agent 4 ({settings.agent_4_base_url}) or Agent 5 ({settings.agent_5_base_url}) service is not running; skipping live HTTP integration test.",
    ),
]


def test_live_agent4_to_agent5_http_pipeline():
    """Perform live HTTP request from Agent 4 fusion output into Agent 5 reason endpoint."""
    # Synthetic Agent 4 FusionRequest payload
    agent4_payload = {
        "agent2_output": {
            "rag_evidence": [{"text": "STEMI guidelines support primary PCI", "score": 0.95}],
            "kg_evidence": [{"subject": "STEMI", "predicate": "treated_by", "object": "Primary PCI"}],
        },
        "agent3_output": {
            "medical_evidence": [{"finding": "Primary PCI within 90 mins", "source": "PubMed"}]
        },
    }

    # 1. Send request to Agent 4 /api/v1/fuse
    agent4_url = f"{settings.agent_4_base_url}/api/v1/fuse"
    resp4 = httpx.post(agent4_url, json=agent4_payload, timeout=10.0)
    assert resp4.status_code == 200
    fusion_data = resp4.json()
    assert "unified_context" in fusion_data

    # 2. Extract raw unified_context dictionary produced by Agent 4
    unified_context_json = fusion_data["unified_context"]

    # 3. Send exact unified_context to Agent 5 /api/v1/reason
    agent5_url = f"{settings.agent_5_base_url}/api/v1/reason"
    resp5 = httpx.post(agent5_url, json=unified_context_json, timeout=10.0)
    assert resp5.status_code == 200
    reasoning_data = resp5.json()

    assert "status" in reasoning_data or "agent_metadata" in reasoning_data
