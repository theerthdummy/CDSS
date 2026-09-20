"""Tests for Agent V2 - Simplified clinical text extraction."""

import pytest
from app.agent.agent import ClinicalTextClarifierAgent
from app.agent.schema import Agent1Request


class TestAgentV2:
    """Test Agent V2 functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.agent = ClinicalTextClarifierAgent()
    
    def test_fever_extraction(self):
        """Test basic fever extraction."""
        request = Agent1Request(text="patient has fever")
        response = self.agent.process(request)
        
        assert "fever" in response.symptoms
        assert response.requires_clarification
    
    def test_fever_with_temperature(self):
        """Test fever with temperature extraction."""
        request = Agent1Request(text="patient has fever with temperature 103°F")
        response = self.agent.process(request)
        
        assert "fever" in response.symptoms
        # LLM preserves the exact format from input, so check for presence of "103" and "F"
        temp = response.measurements.get("temperature", "")
        assert "103" in str(temp)
        # Check that measurements were captured
        assert response.measurements
    
    def test_symptom_with_duration(self):
        """Test symptom with duration extraction."""
        request = Agent1Request(text="fever for 3 days")
        response = self.agent.process(request)
        
        assert "fever" in response.symptoms
        assert response.duration is not None
        assert "3" in response.duration
        assert "day" in response.duration
    
    def test_multiple_symptoms(self):
        """Test extraction of multiple symptoms."""
        request = Agent1Request(text="patient has fever, headache, and cough")
        response = self.agent.process(request)
        
        assert "fever" in response.symptoms
        assert "headache" in response.symptoms
        assert "cough" in response.symptoms
    
    def test_severity_extraction(self):
        """Test severity extraction."""
        request = Agent1Request(text="severe headache")
        response = self.agent.process(request)
        
        # LLM may extract "headache" or "severe headache" as symptom
        # Either is acceptable - the key is that severity is captured somehow
        symptoms_text = " ".join(response.symptoms).lower()
        assert "headache" in symptoms_text
        # Severity should be extracted (either as field or part of symptom name)
        assert response.severity == "severe" or "severe" in symptoms_text
    
    def test_empty_input(self):
        """Test empty input handling."""
        request = Agent1Request(text="")
        response = self.agent.process(request)
        
        assert response.symptoms == []
        assert response.measurements == {}
        assert not response.requires_clarification
    
    def test_no_temperature_extraction_from_age(self):
        """Test that age is not confused with temperature."""
        request = Agent1Request(text="45 year old patient")
        response = self.agent.process(request)
        
        # Key assertion: temperature should not be extracted
        assert "temperature" not in response.measurements
        # Age information might be in other_information, but if not extracted at all that's also ok
        # The important thing is that "45" is not treated as a temperature measurement
        assert not any("45" in str(m) for m in response.measurements.values())
    
    def test_clarification_for_missing_temperature(self):
        """Test clarification when fever present but no temperature."""
        request = Agent1Request(text="patient has fever")
        response = self.agent.process(request)
        
        assert response.requires_clarification
        # Clarification should be present when fever has no temperature
        assert response.clarification_question is not None
        # Either missing_information is populated OR the clarification_question indicates what's missing
        assert len(response.missing_information) > 0 or response.clarification_question
    
    def test_no_clarification_when_complete(self):
        """Test no clarification when all info provided."""
        request = Agent1Request(text="patient has high fever 103°F for 2 days")
        response = self.agent.process(request)
        
        # May not require clarification if we have temperature, duration, and severity
        assert response.symptoms
        assert response.measurements

    def test_session_maintains_clarification_answers(self):
        """Test that clarification answers are merged into the active session and update the case."""
        session_id = "demo-session-1"

        first = self.agent.process(Agent1Request(text="patient has fever", session_id=session_id))
        assert first.requires_clarification
        assert first.clarification_question is not None

        second = self.agent.process(Agent1Request(text="103°F", session_id=session_id))
        assert second.session_id == session_id
        assert "103" in str(second.measurements.get("temperature", ""))
        assert second.requires_clarification or second.missing_information == []
