"""Live Local Integration Test for Agent 4 -> Agent 5 REST Communication.

Automatically skipped if live Agent 4 (http://localhost:8004/api/v1/) is not running.
"""

import httpx
import pytest

from app.config import settings

# Helper to check if live Agent 4 service is running locally
def is_live_agent4_available() -> bool:
    try:
        resp = httpx.get(f"{settings.agent_4_base_url}/api/v1/", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.live,
    pytest.mark.integration,
    pytest.mark.skipif(
        not is_live_agent4_available(),
        reason=f"Live Agent 4 service at '{settings.agent_4_base_url}' is not running; skipping live HTTP integration test.",
    ),
]


def test_live_agent4_fuse_to_agent5_reason_contract_pipeline():
    """Send request to live Agent 4 /api/v1/fuse, extract unified_context, and pass to Agent 5 /api/v1/reason."""
    agent4_payload = {
        "agent2_output": {
            "success": True,
            "message": "Retrieved evidence from Agent 2 RAG & KG",
            "data": {
                "query": "Acute STEMI treatment primary PCI",
                "results": [
                    {
                        "text": "ST-elevation myocardial infarction (STEMI) requires immediate emergency reperfusion therapy via primary PCI.",
                        "score": 0.95,
                        "confidence": 0.92,
                        "source": "Biomedical PubMed Vector Store",
                        "document_type": "literature"
                    }
                ]
            }
        },
        "agent3_output": {
            "entities_queried": ["STEMI", "PCI"],
            "summary": "Primary PCI within 90 minutes reduces mortality.",
            "evidence": [
                {
                    "source": "Clinical Guidelines",
                    "title": "ACC/AHA STEMI Reperfusion Guidelines",
                    "abstract": "Emergency catheterization within 90 minutes of medical contact."
                }
            ]
        }
    }

    # 1. Post request to Agent 4 /api/v1/fuse
    agent4_url = f"{settings.agent_4_base_url}/api/v1/fuse"
    resp4 = httpx.post(agent4_url, json=agent4_payload, timeout=10.0)
    assert resp4.status_code == 200
    fusion_data = resp4.json()
    assert "unified_context" in fusion_data

    # 2. Extract unified_context JSON
    unified_context_json = fusion_data["unified_context"]
    assert "merged_findings" in unified_context_json
    assert "confidence_score" in unified_context_json

    # 3. Post extracted context to Agent 5 /api/v1/reason
    agent5_url = f"{settings.agent_5_base_url}/api/v1/reason"
    resp5 = httpx.post(agent5_url, json=unified_context_json, timeout=10.0)
    assert resp5.status_code == 200
    reasoning_data = resp5.json()

    assert "status" in reasoning_data or "agent_metadata" in reasoning_data
