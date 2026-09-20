"""Tests for FastAPI backend API routes."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create a FastAPI TestClient."""
    return TestClient(app)


class TestApiRoutes:
    """Test API endpoint behavior and data flow."""

    def test_root_endpoint(self, client):
        """Test root / endpoint returns 200 and expected status."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Backend Running"}

    def test_test_endpoint(self, client):
        """Test /test endpoint returns success."""
        response = client.get("/test")
        assert response.status_code == 200
        assert response.json() == {"status": "Connected Successfully"}

    def test_clarify_fever_returns_structured_data_and_info_noted(self, client):
        """Test /clarify generates structured data and user-friendly 'Info noted.' message."""
        payload = {"text": "Patient has fever."}
        response = client.post("/clarify", json=payload)

        assert response.status_code == 200
        data = response.json()

        # 1. User message must be "Info noted."
        assert data["message"] == "Info noted."

        # 2. Structured clinical fields are fully preserved
        assert "fever" in data["symptoms"]
        assert data["requires_clarification"] is True
        assert data["clarification_question"] is not None
        assert "session_id" in data
        assert isinstance(data["measurements"], dict)
        assert isinstance(data["medical_history"], list)
        assert isinstance(data["medications"], list)
        assert isinstance(data["missing_information"], list)
        assert isinstance(data["other_information"], list)
        assert isinstance(data["relationships"], list)

    def test_clarify_complete_patient_case(self, client):
        """Test /clarify with complete clinical presentation."""
        payload = {
            "text": "55 y.o female with temp 102°F for 2 days. No history, no medications."
        }
        response = client.post("/clarify", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Confirmation message
        assert data["message"] == "Info noted."

        # Demographics & measurements preserved
        assert data["age"] == "55"
        assert data["gender"] == "female"
        assert "102" in data["measurements"].get("temperature", "")
        assert data["medical_history_status"] == "NONE_REPORTED"
        assert data["medications_status"] == "NONE_REPORTED"

    def test_clarify_multi_turn_session(self, client):
        """Test /clarify maintains session state across multi-turn input."""
        # Turn 1: Enter age
        resp1 = client.post("/clarify", json={"text": "Patient is 45 years old."})
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["message"] == "Info noted."
        session_id = data1["session_id"]
        assert data1["age"] == "45"

        # Turn 2: Enter gender using the same session_id
        resp2 = client.post("/clarify", json={"text": "male", "session_id": session_id})
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["message"] == "Info noted."
        assert data2["age"] == "45"
        assert data2["gender"] == "male"
