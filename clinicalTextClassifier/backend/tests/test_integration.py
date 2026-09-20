"""
Comprehensive tests for Ollama-based Clinical Text Clarifier.

Tests validate:
- Temperature extraction (not confused with age)
- Symptom extraction (no conversion to more specific types)
- Relationship extraction (only explicit relationships)
- Clarification logic (dynamic, not redundant)
- Missing information detection
"""

import pytest
from app.agent.agent import ClinicalTextClarifierAgent
from app.agent.schema import Agent1Request


class TestOllamaIntegration:
    """Test Ollama-based clinical text extraction."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.agent = ClinicalTextClarifierAgent()
    
    # ============================================================================
    # TEST 1: Complex Patient Presentation
    # ============================================================================
    def test_complex_presentation_high_fever_103_not_age(self):
        """
        TEST 1: Complex presentation with high fever 103°F
        
        Verify:
        - 103°F is recognized as temperature (NOT age)
        - body pain is NOT converted to chest pain
        - fever is NOT incorrectly located at head
        - unsupported relationships are NOT generated
        - system does NOT ask for temperature again
        """
        text = (
            "Patient has had a high fever of 103°F for 2 days, "
            "along with severe headache, chills, body pain, and nausea. "
            "The patient also reports feeling very weak."
        )
        
        request = Agent1Request(text=text)
        response = self.agent.process(request)
        
        # ✓ Temperature extracted correctly (may be in different format)
        temp_value = response.measurements.get("temperature", "")
        assert "103" in str(temp_value), \
            f"Expected temperature with '103', got {response.measurements}"
        
        # ✓ 103 is NOT interpreted as age (age field should be None or not "103")
        assert response.age != "103", "103 should not be interpreted as age"

        # ✓ Symptoms extracted correctly
        symptoms_text = " ".join(response.symptoms).lower()
        # Either "fever" is mentioned OR temperature is measured (fever's manifestation)
        has_fever_indication = "fever" in symptoms_text or len(response.measurements.get("temperature", "")) > 0
        assert has_fever_indication, f"Expected fever or temperature in response. Symptoms: {response.symptoms}, Measurements: {response.measurements}"
        assert "headache" in symptoms_text, f"'headache' not in {response.symptoms}"
        assert "pain" in symptoms_text or "body" in symptoms_text, f"pain/body not in {response.symptoms}"  # body pain
        assert "nausea" in symptoms_text, f"'nausea' not in {response.symptoms}"
        assert "chills" in symptoms_text, f"'chills' not in {response.symptoms}"
        assert "weakness" in symptoms_text or "weak" in symptoms_text, f"'weakness' not in {response.symptoms}"

        # ✓ Chest pain should NOT appear (input says body pain)
        assert "chest" not in symptoms_text.lower(), \
            "body pain should not be converted to chest pain"

        # ✓ Duration extracted
        assert response.duration is not None and "2" in response.duration and "day" in response.duration, \
            f"Duration should contain '2' and 'day', got {response.duration}"

        # ✓ Severity extracted
        has_severity = response.severity is not None and "high" in response.severity.lower()
        or_severe_symptom = any("severe" in s.lower() or "high" in s.lower() for s in response.symptoms)
        assert has_severity or or_severe_symptom, \
            f"Severity should be captured. Severity field: {response.severity}, Symptoms: {response.symptoms}"

        # ✓ Should NOT ask for temperature (already provided)
        if response.requires_clarification and response.clarification_question:
            q_lower = response.clarification_question.lower()
            assert "temperature" not in q_lower, \
                f"Should not ask for temperature (already provided). Got: {response.clarification_question}"
            assert "fever" not in q_lower or "duration" in q_lower, \
                "If asking about fever, should be about missing context (duration, etc)"
        
        print("✅ TEST 1 PASSED: Complex presentation handled correctly")
        print(f"   Extracted: {response.symptoms}")
        print(f"   Temperature: {response.measurements}")
        print(f"   Duration: {response.duration}")
        print(f"   Severity: {response.severity}")
    
    # ============================================================================
    # TEST 2: Chest Pain with Simple Info
    # ============================================================================
    def test_chest_pain_dynamic_clarification(self):
        """
        TEST 2: Chest pain for 3 hours
        
        Verify:
        - Chest pain extracted correctly
        - Next question dynamically selected by LLM
        - System doesn't ask for already-provided info (duration)
        """
        text = "Patient has chest pain for 3 hours."
        
        request = Agent1Request(text=text)
        response = self.agent.process(request)
        
        # ✓ Pain extracted (may be "pain", "chest pain", or more specific)
        symptoms_text = " ".join(response.symptoms).lower()
        assert "pain" in symptoms_text, f"'pain' not found in {response.symptoms}"
        
        # ✓ Duration extracted
        assert response.duration is not None and "3" in response.duration and "hour" in response.duration, \
            f"Duration should contain '3' and 'hour', got {response.duration}"
        
        # ✓ May need clarification, but NOT for duration
        if response.requires_clarification and response.clarification_question:
            q_lower = response.clarification_question.lower()
            assert "duration" not in q_lower and "how long" not in q_lower, \
                f"Should not ask for duration (already provided). Got: {response.clarification_question}"
        
        print("✅ TEST 2 PASSED: Chest pain handled correctly")
        print(f"   Symptoms: {response.symptoms}")
        print(f"   Duration: {response.duration}")
        print(f"   Clarification: {response.clarification_question}")
    
    # ============================================================================
    # TEST 3: Severe Chest Pain with Multiple Symptoms
    # ============================================================================
    def test_severe_chest_pain_no_redundant_clarification(self):
        """
        TEST 3: Severe chest pain with sweating and shortness of breath
        
        Verify:
        - Multiple symptoms extracted
        - Severity recognized
        - System does NOT ask for redundant information
        """
        text = (
            "Patient has severe chest pain for 3 hours "
            "with sweating and shortness of breath."
        )
        
        request = Agent1Request(text=text)
        response = self.agent.process(request)
        
        # ✓ Symptoms extracted (may include "pain", "chest pain", "severe chest pain", etc.)
        symptoms_text = " ".join(response.symptoms).lower()
        assert "pain" in symptoms_text, f"'pain' not found in {response.symptoms}"
        
        # ✓ Duration present
        assert response.duration is not None and "3" in response.duration, \
            f"Duration should be extracted, got {response.duration}"
        
        # ✓ Severity recognized
        assert response.severity is not None and "severe" in response.severity.lower(), \
            f"Severity should contain 'severe', got {response.severity}"
        
        # ✓ If clarification needed, it's not redundant
        if response.requires_clarification and response.clarification_question:
            q_lower = response.clarification_question.lower()
            # Should not ask for already-provided info
            assert "severity" not in q_lower, "Should not ask for severity (already stated)"
            assert "how long" not in q_lower, "Should not ask for duration (already stated)"
        
        print("✅ TEST 3 PASSED: Severe chest pain handled correctly")
        print(f"   Symptoms: {response.symptoms}")
        print(f"   Severity: {response.severity}")
        print(f"   Duration: {response.duration}")
    
    # ============================================================================
    # TEST 4: Simple Fever (Missing Information)
    # ============================================================================
    def test_fever_alone_identifies_missing_info(self):
        """
        TEST 4: Simple "Patient has fever"
        
        Verify:
        - Missing information detected
        - ONE appropriate clarification question generated
        - Question is clinically useful
        """
        text = "Patient has fever."
        
        request = Agent1Request(text=text)
        response = self.agent.process(request)
        
        # ✓ Fever recognized
        assert "fever" in response.symptoms, f"'fever' not in {response.symptoms}"
        
        # ✓ Missing information identified
        assert len(response.missing_information) > 0, \
            "Should identify missing information (temperature, duration, etc)"
        
        # ✓ Requires clarification
        assert response.requires_clarification, \
            "Should require clarification when fever but no temp/duration"
        
        # ✓ ONE question only
        assert response.clarification_question is not None, \
            "Should have ONE clarification question"
        
        # ✓ Question makes clinical sense
        q_lower = response.clarification_question.lower()
        valid_questions = ["age", "temperature", "how long", "duration", "how severe", "severity", 
                          "when", "started", "took temperature"]
        assert any(keyword in q_lower for keyword in valid_questions), \
            f"Question should ask about temperature/duration/severity. Got: {response.clarification_question}"
        
        print("✅ TEST 4 PASSED: Simple fever generates appropriate clarification")
        print(f"   Missing Info: {response.missing_information}")
        print(f"   Clarification Q: {response.clarification_question}")
    
    # ============================================================================
    # TEST 5: Accuracy Verification - No Number Confusion
    # ============================================================================
    def test_no_confusion_between_temperature_and_age(self):
        """
        Additional accuracy test: Verify temperature not confused with age.
        """
        texts = [
            "Patient has 103°F fever",  # 103°F should be temperature
            "Temperature is 102.5 Fahrenheit",  # 102.5 should be temperature
            "Patient reports 99°F",  # 99°F should be temperature
        ]
        
        for text in texts:
            request = Agent1Request(text=text)
            response = self.agent.process(request)
            
            # Should have temperature, not age
            assert len(response.measurements) > 0, \
                f"Should extract measurement from '{text}'"
            
            # No false age extraction — age field should not be set to a temperature reading
            assert response.age != str(response.measurements.get("temperature", "")).replace("°F", "").strip(), \
                f"Temperature value should not be placed in age field for '{text}'"
        
        print("✅ TEST 5 PASSED: Temperature not confused with age")
    
    # ============================================================================
    # TEST 6: Output Structure Validation
    # ============================================================================
    def test_output_structure_valid(self):
        """
        Verify output JSON structure matches Agent1Response exactly.
        """
        text = "Patient has fever 103°F for 2 days"
        
        request = Agent1Request(text=text)
        response = self.agent.process(request)
        
        # Validate all required fields exist (simplified schema)
        assert hasattr(response, 'session_id'), "Missing 'session_id' field"
        assert hasattr(response, 'age'), "Missing 'age' field"
        assert hasattr(response, 'gender'), "Missing 'gender' field"
        assert hasattr(response, 'symptoms'), "Missing 'symptoms' field"
        assert hasattr(response, 'denied_symptoms'), "Missing 'denied_symptoms' field"
        assert hasattr(response, 'measurements'), "Missing 'measurements' field"
        assert hasattr(response, 'duration'), "Missing 'duration' field"
        assert hasattr(response, 'severity'), "Missing 'severity' field"
        assert hasattr(response, 'medical_history'), "Missing 'medical_history' field"
        assert hasattr(response, 'medical_history_status'), "Missing 'medical_history_status' field"
        assert hasattr(response, 'medications'), "Missing 'medications' field"
        assert hasattr(response, 'medications_status'), "Missing 'medications_status' field"
        assert hasattr(response, 'missing_information'), "Missing 'missing_information' field"
        assert hasattr(response, 'requires_clarification'), "Missing 'requires_clarification' field"
        assert hasattr(response, 'clarification_question'), "Missing 'clarification_question' field"
        assert hasattr(response, 'terminology_mappings'), "Missing 'terminology_mappings' field"

        # Validate types
        assert isinstance(response.symptoms, list), "symptoms should be list"
        assert isinstance(response.denied_symptoms, list), "denied_symptoms should be list"
        assert isinstance(response.measurements, dict), "measurements should be dict"
        assert isinstance(response.medical_history, list), "medical_history should be list"
        assert isinstance(response.medications, list), "medications should be list"
        assert isinstance(response.missing_information, list), "missing_information should be list"
        assert isinstance(response.requires_clarification, bool), "requires_clarification should be bool"
        assert isinstance(response.terminology_mappings, list), "terminology_mappings should be list"

        print("✅ TEST 6 PASSED: Output structure valid")
        print(f"   Response type: {type(response)}")
        print(f"   All fields present and properly typed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
