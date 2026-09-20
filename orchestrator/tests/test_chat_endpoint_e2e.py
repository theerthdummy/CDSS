"""Test Case: End-to-End Test for the /chat Endpoint.

Verifies:
1. Multi-turn conversation through the FastAPI endpoint.
2. The fever + hallucinations case produces a grounded, urgent response without symptom fabrication.
3. State accumulates across turns.
"""

import pytest
from fastapi.testclient import TestClient
from orchestrator.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_e2e_fever_hallucination_chat(client):
    """Test the hallucination bug scenario end-to-end through the /chat API."""
    payload = {
        "message": (
            "I'm 55 years old with no previous medical history and I've had a fever of "
            "102°F for 2 days. I started hallucinating."
        ),
        "session_id": "test-e2e-fever-hallucination",
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["session_id"] == "test-e2e-fever-hallucination"
    assert data["urgent_flag"] is True

    # Patient state assertions
    state = data["patient_state"]
    assert state["age"] == "55"
    assert "fever" in state["confirmed_symptoms"]
    assert "hallucinations" in state["confirmed_symptoms"]
    assert "headache" not in state["confirmed_symptoms"]
    assert "neck stiffness" not in state["confirmed_symptoms"]
    assert "vomiting" not in state["confirmed_symptoms"]

    # Model metadata
    metadata = data["llm_metadata"]
    assert metadata["provider"] == "OpenAI"
    assert "gpt" in metadata["model"].lower()

    # Response verification
    resp_text = data["response"].lower()
    # Must advise urgency/emergency evaluation
    assert any(w in resp_text for w in ["emergency", "urgent", "immediate", "medical attention", "evaluation"])


def test_e2e_multiturn_chat_negation(client):
    """Test multi-turn denial in conversational flow."""
    session_id = "test-e2e-multiturn-negation"

    # Turn 1
    r1 = client.post("/chat", json={"message": "I have had a fever for 2 days.", "session_id": session_id})
    assert r1.status_code == 200
    assert "fever" in r1.json()["patient_state"]["confirmed_symptoms"]

    # Turn 2: Deny headache
    r2 = client.post("/chat", json={"message": "No headache.", "session_id": session_id})
    assert r2.status_code == 200
    state2 = r2.json()["patient_state"]
    assert "fever" in state2["confirmed_symptoms"]
    assert "headache" in state2["denied_symptoms"]
    assert "headache" not in state2["confirmed_symptoms"]
