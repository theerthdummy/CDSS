"""Validation script for live Agent 4 -> Agent 5 local HTTP communication."""

import sys
import httpx
from pydantic import ValidationError

from app.config import settings
from app.models.input_models import UnifiedClinicalContext
from app.models.feedback_models import FeedbackRequest


def run_live_validation():
    print("=" * 60)
    print("AGENT 4 -> AGENT 5 LIVE HTTP VALIDATION SCRIPT")
    print("=" * 60)
    print(f"Agent 4 Base URL: {settings.agent_4_base_url}")
    print(f"Agent 5 Base URL: {settings.agent_5_base_url}")
    print("-" * 60)

    # 1. Health Checks
    print("\n[Stage 1] Checking Service Health Endpoints...")
    agent4_health_url = f"{settings.agent_4_base_url}/api/v1/"
    agent5_health_url = f"{settings.agent_5_base_url}/health"

    agent4_online = False
    agent5_online = False

    try:
        r4 = httpx.get(agent4_health_url, timeout=3.0)
        if r4.status_code == 200:
            print(f"  [SUCCESS] Agent 4 Health Check OK: {r4.json()}")
            agent4_online = True
        else:
            print(f"  [WARNING] Agent 4 Health Check returned HTTP {r4.status_code}")
    except Exception as e:
        print(f"  [INFO] Agent 4 service not currently running at '{settings.agent_4_base_url}': {e}")

    try:
        r5 = httpx.get(agent5_health_url, timeout=3.0)
        if r5.status_code == 200:
            print(f"  [SUCCESS] Agent 5 Health Check OK: {r5.json()}")
            agent5_online = True
        else:
            print(f"  [WARNING] Agent 5 Health Check returned HTTP {r5.status_code}")
    except Exception as e:
        print(f"  [INFO] Agent 5 service not currently running at '{settings.agent_5_base_url}': {e}")

    if not agent4_online or not agent5_online:
        print("\n[NOTE] Live services are not both active. Full live HTTP roundtrip skipped.")
        print("To run live HTTP roundtrip, start services in separate terminals:")
        print("  1. Agent 4: uvicorn app.main:app --port 8004 (in D:\\Projects\\CDSS\\agent-4-datafusion)")
        print("  2. Agent 5: uvicorn app.main:app --port 8005 (in D:\\Projects\\CDSS\\agent-5-clinical-reasoning)")
        print("=" * 60)
        return

    # 2. High-Confidence Data Fusion Pipeline
    print("\n[Stage 2] Executing Live Data Fusion (Agent 4) -> Clinical Reasoning (Agent 5)...")
    payload = {
        "agent2_output": {
            "success": True,
            "data": {
                "query": "Acute STEMI PCI treatment",
                "results": [
                    {
                        "text": "STEMI requires primary PCI within 90 minutes of medical contact.",
                        "score": 0.96,
                        "confidence": 0.92,
                        "source": "Biomedical RAG Vector Store"
                    }
                ]
            }
        },
        "agent3_output": {
            "summary": "Administer Aspirin 325mg orally and perform primary PCI emergency catheterization.",
            "evidence": [
                {
                    "source": "Clinical Guidelines",
                    "title": "ACC/AHA STEMI Guidelines 2025",
                    "abstract": "Emergency reperfusion via primary PCI reduces mortality."
                }
            ]
        }
    }

    try:
        fuse_resp = httpx.post(f"{settings.agent_4_base_url}/api/v1/fuse", json=payload, timeout=10.0)
        print(f"  Agent 4 Response HTTP Status: {fuse_resp.status_code}")
        fuse_json = fuse_resp.json()
        unified_context = fuse_json.get("unified_context")
        assert unified_context is not None, "Missing unified_context in Agent 4 response"
        print(f"  [SUCCESS] Received UnifiedClinicalContext from Agent 4 (confidence: {unified_context.get('confidence_score')})")

        # Send context to Agent 5
        reason_resp = httpx.post(f"{settings.agent_5_base_url}/api/v1/reason", json=unified_context, timeout=10.0)
        print(f"  Agent 5 Response HTTP Status: {reason_resp.status_code}")
        reason_json = reason_resp.json()
        print(f"  [SUCCESS] Agent 5 Response Received: {reason_json.get('status') or reason_json.get('agent_metadata', {}).get('status')}")
    except Exception as exc:
        print(f"  [ERROR] Live HTTP roundtrip failed: {exc}")

    print("=" * 60)


if __name__ == "__main__":
    run_live_validation()
