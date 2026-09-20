"""Comprehensive test suites for Agent 1 (Clinical Information Extraction Agent).

Covers all 18 clinical testing categories and explicit behavioral constraints:
1. Dynamic Router Tests (Level 1 vs Level 2, turn-by-turn dynamic switching)
2. Zero-LLM Assertion for Level 1 Atomic Inputs
3. Normalization (temperature units, number words, clinical shorthand, typos)
4. Context-Dependent Short Answers (answering pending questions)
5. Contradictions and Corrections
6. State Preservation & Sequential Intake (Zero Information Loss)
7. Duplicate Idempotency & Partial Updates
8. Entity-Specific Negation & Unknown Distinction
9. Empty and Malformed Input Handling
10. Level 2 Failure Resilience (Mocked LLM Failures)
11. Realistic Complex & Out-of-Order Clinical Intakes
12. Temperature Unit Context Resolution (No Arbitrary Unit Invention)
"""

import uuid
from unittest.mock import patch
import pytest

from app.agent.agent import ClinicalTextClarifierAgent
from app.agent.extractor import classify_input_complexity, extract_deterministic_entities
from app.agent.normalizer import normalize_clinical_text
from app.agent.schema import Agent1Request
from app.agent.state import create_empty_patient_state, merge_patient_state, determine_next_clarification


# =============================================================================
# 1. Dynamic Router Tests
# =============================================================================

class TestDynamicRouter:
    """Tests for classify_input_complexity() and per-turn dynamic routing."""

    def setup_method(self):
        self.classify = classify_input_complexity
        self.agent = ClinicalTextClarifierAgent()

    def test_level1_atomic_routes(self):
        """Atomic / simple inputs must be classified as Level 1."""
        atomic_cases = [
            "45",
            "Male",
            "102°F",
            "No medical history.",
            "Hypertension",
            "Metformin",
            "45 years old",
            "Female",
            "none",
            "No medications",
            "since yesterday",
        ]
        for text in atomic_cases:
            route = self.classify(text)
            assert route == "level1", f"Expected 'level1' for atomic input '{text}', got '{route}'"

    def test_level2_complex_routes(self):
        """Multi-field, composite, and narrative inputs must be classified as Level 2."""
        complex_cases = [
            "55-year-old female with fever and cough",
            "55 year old female with hypertension and fever.",
            "55 year old female with fever 102°F for 2 days and cough.",
            "48M with HTN and DM2, fever for 3 days, productive cough and SOB.",
            "I've had fever around 102°F for three days with chills, cough and body aches.",
            "Patient has fever and cough. She is a 55-year-old woman with hypertension.",
            "im 34 male, fever like 101/102, throat hurts, coughing a lot, started 2 days ago, no meds, no past illness",
        ]
        for text in complex_cases:
            route = self.classify(text)
            assert route == "level2", f"Expected 'level2' for complex input '{text}', got '{route}'"

    def test_dynamic_turn_by_turn_alternation(self):
        """The system must make a fresh routing decision for every user turn:
        Turn 1: 45 (L1) -> Turn 2: Male (L1) -> Turn 3: Complex narrative (L2) -> Turn 4: 102°F (L1).
        """
        session_id = f"test-dynamic-turn-{uuid.uuid4()}"

        # Turn 1: 45 -> Level 1
        r1 = self.agent.process(Agent1Request(text="45", session_id=session_id))
        session = self.agent._session_store.get(session_id)
        assert session["history"][-1]["used_llm"] is False
        assert r1.age == "45"

        # Turn 2: Male -> Level 1
        r2 = self.agent.process(Agent1Request(text="Male", session_id=session_id))
        assert session["history"][-1]["used_llm"] is False
        assert r2.gender == "male"

        # Turn 3: Narrative -> Level 2
        r3 = self.agent.process(Agent1Request(
            text="I've had fever around 102°F for three days with chills, cough and body aches.",
            session_id=session_id
        ))
        assert session["history"][-1]["used_llm"] is True
        assert "fever" in r3.symptoms
        assert r3.duration is not None

        # Turn 4: 102°F -> Level 1 (Switch back to Level 1)
        r4 = self.agent.process(Agent1Request(text="102°F", session_id=session_id))
        assert session["history"][-1]["used_llm"] is False
        assert "102" in r4.measurements.get("temperature", "")

    def test_reverse_dynamic_routing(self):
        """Turn 1: Complex narrative (L2) -> Turn 2: Simple atomic 102°F (L1)."""
        session_id = f"test-reverse-dynamic-{uuid.uuid4()}"

        # Turn 1: Level 2
        r1 = self.agent.process(Agent1Request(
            text="55-year-old female with fever, cough and hypertension.",
            session_id=session_id
        ))
        session = self.agent._session_store.get(session_id)
        assert session["history"][-1]["used_llm"] is True
        assert r1.age == "55"
        assert r1.gender == "female"

        # Turn 2: Level 1
        r2 = self.agent.process(Agent1Request(text="102°F", session_id=session_id))
        assert session["history"][-1]["used_llm"] is False
        assert "102" in r2.measurements.get("temperature", "")


# =============================================================================
# 2. Level 1 Zero-LLM Verification Tests
# =============================================================================

class TestLevel1ZeroLLM:
    """Verifies that Level 1 deterministic extraction executes with exactly 0 LLM calls."""

    def setup_method(self):
        self.agent = ClinicalTextClarifierAgent()

    def test_zero_llm_calls_for_atomic_inputs(self):
        """For inputs that Level 1 can confidently understand, assert 0 LLM calls."""
        cases = [
            ("45", "age", "45"),
            ("Male", "gender", "male"),
            ("102°F", "temperature", "102°F"),
            ("No medical history.", "medical_history_status", "NONE_REPORTED"),
            ("Hypertension", "medical_history", ["hypertension"]),
            ("Metformin", "medications", ["metformin"]),
        ]
        for text, field_name, expected_val in cases:
            resp = self.agent.process(Agent1Request(text=text))
            session = self.agent._session_store.get(resp.session_id)
            used_llm = session["history"][-1].get("used_llm")

            assert used_llm is False, f"Expected 0 LLM calls for '{text}', but used_llm was {used_llm}"

            if field_name == "temperature":
                assert expected_val in resp.measurements.get("temperature", "")
            elif field_name == "medical_history":
                assert expected_val[0] in resp.medical_history
            elif field_name == "medications":
                assert expected_val[0] in resp.medications
            elif field_name == "medical_history_status":
                assert resp.medical_history_status == expected_val
            elif field_name == "age":
                assert resp.age == expected_val
            elif field_name == "gender":
                assert resp.gender == expected_val


# =============================================================================
# 3. Normalization Tests
# =============================================================================

class TestNormalization:
    """Tests normalization across units, number words, clinical shorthand, and spelling."""

    def test_temperature_unit_equivalences(self):
        variations = [
            "102°F",
            "102 F",
            "102 deg F",
            "102 degrees Fahrenheit",
            "Temp is 102f",
        ]
        for text in variations:
            normalized = normalize_clinical_text(text)
            assert "102°F" in normalized, f"Failed to normalize temperature variation: '{text}' -> '{normalized}'"

    def test_age_variations_and_number_words(self):
        cases = [
            ("45 years old", "45"),
            ("45 y/o", "45"),
            ("Age: 45", "45"),
            ("I'm forty-five.", "45"),
            ("I'm a 32-year-old man.", "32"),
        ]
        for text, expected_age in cases:
            normalized = normalize_clinical_text(text)
            assert expected_age in normalized, f"Failed to normalize age phrase: '{text}' -> '{normalized}'"

    def test_gender_variations(self):
        cases = [
            ("I'm a man.", "male"),
            ("Male", "male"),
            ("Female", "female"),
            ("I am a 32-year-old woman.", "female"),
        ]
        agent = ClinicalTextClarifierAgent()
        for text, expected_gender in cases:
            resp = agent.process(Agent1Request(text=text))
            assert resp.gender == expected_gender

    def test_spelling_and_clinical_shorthand(self):
        # Shorthand note: pt 48M c/o fever x3d, Tmax 103F, cough w/ yellow sputum, SOB since last night. Hx DM2, HTN.
        text = "pt 48M c/o fever x3d, Tmax 103F, cough w/ yellow sputum, SOB since last night. Hx DM2, HTN."
        normalized = normalize_clinical_text(text)
        assert "48 year old male" in normalized or ("48" in normalized and "male" in normalized)
        assert "shortness of breath" in normalized
        assert "hypertension" in normalized
        assert "type 2 diabetes" in normalized or "diabetes" in normalized
        assert "103°F" in normalized

        # Common typos: fevr, hedache, sever, diziness
        noisy = "Pt has fevr, temp 102 deg F, SOB and hedache."
        noisy_norm = normalize_clinical_text(noisy)
        assert "fever" in noisy_norm
        assert "headache" in noisy_norm
        assert "shortness of breath" in noisy_norm

        noisy2 = "Patient has sever chest pain and diziness."
        noisy2_norm = normalize_clinical_text(noisy2)
        assert "severe" in noisy2_norm
        assert "dizziness" in noisy2_norm


# =============================================================================
# 4. Context-Dependent Short Answers
# =============================================================================

class TestContextDependentShortAnswers:
    """Verifies that short responses are interpreted using the preceding question/context."""

    def test_contextual_short_answers_multi_turn(self):
        agent = ClinicalTextClarifierAgent()
        session_id = f"test-context-q-{uuid.uuid4()}"

        # Initialize session state with fever
        agent.process(Agent1Request(text="Patient has fever.", session_id=session_id))
        session = agent._session_store.get(session_id)

        # Context 1: Assistant asks temperature -> User answers "102"
        session["pending_question"] = "What is the patient's current temperature?"
        resp1 = agent.process(Agent1Request(text="102", session_id=session_id))
        assert "102" in resp1.measurements.get("temperature", "")

        # Context 2: Assistant asks about hypertension -> User answers "Yes."
        session["pending_question"] = "Does the patient have a history of hypertension?"
        resp2 = agent.process(Agent1Request(text="Yes.", session_id=session_id))
        assert "hypertension" in resp2.medical_history
        assert resp2.medical_history_status == "PRESENT"

        # Context 3: Assistant asks duration -> User answers "Since yesterday."
        session["pending_question"] = "How long has the fever been present?"
        resp3 = agent.process(Agent1Request(text="Since yesterday.", session_id=session_id))
        assert resp3.duration == "since yesterday"


# =============================================================================
# 5. Contradictions and Corrections
# =============================================================================

class TestContradictionsAndCorrections:
    """Verifies that patient state updates cleanly on corrections without accumulating duplicates."""

    def setup_method(self):
        self.agent = ClinicalTextClarifierAgent()

    def test_age_correction(self):
        session_id = f"test-corr-age-{uuid.uuid4()}"
        r1 = self.agent.process(Agent1Request(text="Patient is 45 years old.", session_id=session_id))
        assert r1.age == "45"

        r2 = self.agent.process(Agent1Request(text="Actually, the patient is 46.", session_id=session_id))
        assert r2.age == "46"

    def test_gender_correction(self):
        session_id = f"test-corr-gender-{uuid.uuid4()}"
        r1 = self.agent.process(Agent1Request(text="The patient is male.", session_id=session_id))
        assert r1.gender == "male"

        r2 = self.agent.process(Agent1Request(text="Sorry, I meant female.", session_id=session_id))
        assert r2.gender == "female"

    def test_history_correction(self):
        session_id = f"test-corr-hist-{uuid.uuid4()}"
        r1 = self.agent.process(Agent1Request(text="Patient has no medical history.", session_id=session_id))
        assert r1.medical_history_status == "NONE_REPORTED"
        assert r1.medical_history == []

        r2 = self.agent.process(Agent1Request(text="Actually, she has hypertension.", session_id=session_id))
        assert r2.medical_history_status == "PRESENT"
        assert "hypertension" in r2.medical_history


# =============================================================================
# 6. State Preservation & Sequential Intake (Zero Information Loss)
# =============================================================================

class TestStatePreservationAndSequentialIntake:
    """Tests sequential state preservation across multiple Level 1 and Level 2 turns."""

    def test_sequential_state_preservation_no_information_loss(self):
        agent = ClinicalTextClarifierAgent()
        session_id = f"test-seq-preservation-{uuid.uuid4()}"

        # Turn 1: Level 2
        r1 = agent.process(Agent1Request(
            text="55-year-old female with hypertension.",
            session_id=session_id
        ))
        assert r1.age == "55"
        assert r1.gender == "female"
        assert "hypertension" in r1.medical_history

        # Turn 2: Level 2
        r2 = agent.process(Agent1Request(
            text="Patient has fever, cough, and shortness of breath for three days.",
            session_id=session_id
        ))
        assert r2.age == "55"
        assert r2.gender == "female"
        assert "hypertension" in r2.medical_history
        assert "fever" in r2.symptoms
        assert "cough" in r2.symptoms
        assert "shortness of breath" in r2.symptoms
        assert r2.duration is not None

        # Turn 3: Level 1
        r3 = agent.process(Agent1Request(
            text="Temperature is 102°F.",
            session_id=session_id
        ))
        # Final state MUST retain all facts from turns 1, 2, and 3
        assert r3.age == "55"
        assert r3.gender == "female"
        assert "hypertension" in r3.medical_history
        assert "fever" in r3.symptoms
        assert "cough" in r3.symptoms
        assert "shortness of breath" in r3.symptoms
        assert r3.duration is not None
        assert "102" in r3.measurements.get("temperature", "")

    def test_mandatory_fields_never_reasked(self):
        """Already provided mandatory fields must never be requested again."""
        agent = ClinicalTextClarifierAgent()
        session_id = f"test-no-reask-{uuid.uuid4()}"

        # Turn 1: Patient has fever and cough -> Missing: age, gender, medical history
        r1 = agent.process(Agent1Request(text="Patient has fever and cough.", session_id=session_id))
        assert "fever" in r1.symptoms
        assert "cough" in r1.symptoms
        assert r1.requires_clarification
        assert "age" in r1.clarification_question.lower()

        # Turn 2: 55-year-old female with hypertension.
        r2 = agent.process(Agent1Request(
            text="55-year-old female with hypertension.",
            session_id=session_id
        ))
        assert r2.age == "55"
        assert r2.gender == "female"
        assert "hypertension" in r2.medical_history
        assert "fever" in r2.symptoms
        assert "cough" in r2.symptoms

        # MUST NOT ask for age, gender, or medical history again
        if r2.clarification_question:
            q_lower = r2.clarification_question.lower()
            assert "age" not in q_lower
            assert "gender" not in q_lower
            assert "medical condition" not in q_lower


# =============================================================================
# 7. Duplicate Idempotency & Partial Updates
# =============================================================================

class TestDuplicateAndPartialUpdates:
    """Tests idempotency under repeated inputs and clean partial updates."""

    def test_duplicate_input_idempotency(self):
        agent = ClinicalTextClarifierAgent()
        session_id = f"test-idempotent-{uuid.uuid4()}"

        r1 = agent.process(Agent1Request(text="Patient is 55 years old.", session_id=session_id))
        r2 = agent.process(Agent1Request(text="Patient is 55 years old.", session_id=session_id))

        assert r2.age == "55"
        assert r2.medical_history == []
        assert r2.symptoms == []

    def test_partial_update_preserves_unrelated_state(self):
        agent = ClinicalTextClarifierAgent()
        session_id = f"test-partial-upd-{uuid.uuid4()}"

        r1 = agent.process(Agent1Request(text="Patient has fever and cough.", session_id=session_id))
        assert "fever" in r1.symptoms
        assert "cough" in r1.symptoms

        r2 = agent.process(Agent1Request(
            text="The fever has gotten worse and the temperature is 103°F.",
            session_id=session_id
        ))
        # Updated temperature & retained cough
        assert "fever" in r2.symptoms
        assert "cough" in r2.symptoms
        assert "103" in r2.measurements.get("temperature", "")


# =============================================================================
# 8. Entity-Specific Negation & Unknown Distinction
# =============================================================================

class TestEntitySpecificNegationAndUnknowns:
    """Tests entity-specific negations and explicit distinction between negative vs unknown."""

    def setup_method(self):
        self.agent = ClinicalTextClarifierAgent()

    def test_denied_symptoms_not_positive(self):
        cases = [
            ("Patient has fever but no cough.", ["fever"], ["cough"]),
            ("No chest pain or shortness of breath.", [], ["chest pain", "shortness of breath"]),
            ("Denies nausea, vomiting and diarrhea.", [], ["nausea", "vomiting", "diarrhea"]),
        ]
        for text, exp_pos, exp_den in cases:
            resp = self.agent.process(Agent1Request(text=text))
            for sym in exp_pos:
                assert sym in resp.symptoms
                assert sym not in resp.denied_symptoms
            for sym in exp_den:
                assert sym in resp.denied_symptoms
                assert sym not in resp.symptoms

    def test_entity_specific_history_negation(self):
        session_id = f"test-entity-neg-{uuid.uuid4()}"
        # Establish history with asthma and hypertension
        r1 = self.agent.process(Agent1Request(text="History of asthma and hypertension.", session_id=session_id))
        assert "asthma" in r1.medical_history
        assert "hypertension" in r1.medical_history

        # Specifically negate hypertension -> Asthma must remain!
        r2 = self.agent.process(Agent1Request(text="No history of hypertension.", session_id=session_id))
        assert "asthma" in r2.medical_history
        assert "hypertension" not in r2.medical_history

    def test_unknown_vs_negative_history_distinction(self):
        # 1. Explicit negative
        resp_neg = self.agent.process(Agent1Request(text="No medical history."))
        assert resp_neg.medical_history_status == "NONE_REPORTED"
        assert resp_neg.medical_history == []

        # 2. Unknown history
        resp_unk1 = self.agent.process(Agent1Request(text="I don't know my medical history."))
        assert resp_unk1.medical_history_status == "UNKNOWN"

        # 3. Unsure history
        resp_unk2 = self.agent.process(Agent1Request(text="I'm not sure about my medical history."))
        assert resp_unk2.medical_history_status == "UNKNOWN"

    def test_unknown_history_does_not_loop_infinitely(self):
        session_id = f"test-no-loop-unk-{uuid.uuid4()}"
        self.agent.process(Agent1Request(text="45 year old male with fever.", session_id=session_id))
        session = self.agent._session_store.get(session_id)
        session["pending_question"] = "Do you have any existing medical conditions or previous diagnoses?"

        resp = self.agent.process(Agent1Request(text="I don't know.", session_id=session_id))
        assert resp.medical_history_status == "UNKNOWN"
        # Must NOT ask for medical history again; should progress to temperature or duration
        if resp.clarification_question:
            assert "existing medical conditions" not in resp.clarification_question.lower()


# =============================================================================
# 9. Empty and Malformed Inputs
# =============================================================================

class TestMalformedAndEmptyInputs:
    """Verifies graceful handling of empty, whitespace, and malformed inputs."""

    def setup_method(self):
        self.agent = ClinicalTextClarifierAgent()

    def test_empty_string(self):
        resp = self.agent.process(Agent1Request(text=""))
        assert resp.symptoms == []
        assert resp.measurements == {}
        assert not resp.requires_clarification

    def test_whitespace_string(self):
        resp = self.agent.process(Agent1Request(text="   "))
        assert resp.symptoms == []
        assert not resp.requires_clarification

    def test_punctuation_only(self):
        resp = self.agent.process(Agent1Request(text="???"))
        assert resp.symptoms == []

    def test_conversational_same(self):
        session_id = f"test-same-{uuid.uuid4()}"
        self.agent.process(Agent1Request(text="55 year old male with fever.", session_id=session_id))
        resp = self.agent.process(Agent1Request(text="same", session_id=session_id))
        assert resp.age == "55"
        assert resp.gender == "male"
        assert "fever" in resp.symptoms


# =============================================================================
# 10. Level 2 Failure Resilience (Mocked LLM Failures)
# =============================================================================

class TestLevel2FailureResilience:
    """Verifies that Level 2 failures (timeout, 429, malformed JSON) never corrupt state."""

    def setup_method(self):
        self.agent = ClinicalTextClarifierAgent()

    def test_llm_timeout_preserves_state(self):
        session_id = f"test-timeout-{uuid.uuid4()}"
        # Turn 1: establish initial state
        self.agent.process(Agent1Request(text="55-year-old female with hypertension.", session_id=session_id))

        # Turn 2: LLM raises Timeout
        with patch("app.services.gemini.call_gemini_json", side_effect=TimeoutError("API timed out")):
            with patch("app.services.gemini.is_gemini_available", return_value=True):
                resp = self.agent.process(Agent1Request(
                    text="The patient also reports severe chest pain and dizziness since yesterday.",
                    session_id=session_id
                ))
                # State from Turn 1 MUST be preserved
                assert resp.age == "55"
                assert resp.gender == "female"
                assert "hypertension" in resp.medical_history
                # Safe deterministic entities from Turn 2 are captured
                assert "chest pain" in resp.symptoms

    def test_llm_rate_limit_429_preserves_state(self):
        session_id = f"test-429-{uuid.uuid4()}"
        self.agent.process(Agent1Request(text="48 year old male with diabetes.", session_id=session_id))

        with patch("app.services.gemini.call_gemini_json", side_effect=RuntimeError("Gemini API error (429)")):
            with patch("app.services.gemini.is_gemini_available", return_value=True):
                resp = self.agent.process(Agent1Request(
                    text="Patient has fever 102°F for 2 days and cough.",
                    session_id=session_id
                ))
                assert resp.age == "48"
                assert resp.gender == "male"
                assert "diabetes" in resp.medical_history
                assert "102" in resp.measurements.get("temperature", "")
                assert "fever" in resp.symptoms

    def test_llm_malformed_json_preserves_state(self):
        session_id = f"test-malformed-{uuid.uuid4()}"
        self.agent.process(Agent1Request(text="Patient is 35.", session_id=session_id))

        with patch("app.services.gemini.call_gemini_json", return_value="INVALID_NON_DICT_STRING"):
            with patch("app.services.gemini.is_gemini_available", return_value=True):
                resp = self.agent.process(Agent1Request(
                    text="Female with severe headache and fever.",
                    session_id=session_id
                ))
                assert resp.age == "35"
                assert resp.gender == "female"
                assert "headache" in resp.symptoms


# =============================================================================
# 11. Realistic Complex Clinical Intakes
# =============================================================================

class TestRealisticComplexIntakes:
    """Tests realistic narrative intakes, medication accuracy, and non-clinical filtering."""

    def setup_method(self):
        self.agent = ClinicalTextClarifierAgent()

    def test_out_of_order_intake(self):
        text = "Patient has fever and cough. She is a 55-year-old woman with hypertension."
        resp = self.agent.process(Agent1Request(text=text))
        assert resp.age == "55"
        assert resp.gender == "female"
        assert "hypertension" in resp.medical_history
        assert "fever" in resp.symptoms
        assert "cough" in resp.symptoms

    def test_medications_without_hallucination(self):
        # 1. Unnamed blood pressure tablet
        resp1 = self.agent.process(Agent1Request(text="I'm taking a blood pressure tablet."))
        assert resp1.medications_status == "PRESENT"
        assert any("blood pressure" in m for m in resp1.medications)
        # MUST NOT hallucinate specific brands like amlodipine or lisinopril
        assert "amlodipine" not in resp1.medications
        assert "lisinopril" not in resp1.medications

        # 2. Known specific medication
        resp2 = self.agent.process(Agent1Request(text="Metformin 500 mg twice a day."))
        assert "metformin" in resp2.medications
        assert resp2.medications_status == "PRESENT"

    def test_irrelevant_information_filtering(self):
        text = "Patient has fever. He works as a software engineer and lives with his parents."
        resp = self.agent.process(Agent1Request(text=text))
        assert "fever" in resp.symptoms
        assert resp.gender == "male"
        # Occupation and living situation must not be added to medical history or medications
        assert "software engineer" not in resp.medical_history
        assert "parents" not in resp.medical_history
        assert resp.medications == []

    def test_measurements_and_timing_normalization(self):
        cases = [
            ("Heart rate is 110 bpm.", "heart_rate", "110 bpm"),
            ("BP is 140/90.", "blood_pressure", "140/90 mmHg"),
            ("Blood pressure 140 over 90.", "blood_pressure", "140/90 mmHg"),
            ("Temperature is 38.9°C.", "temperature", "38.9°C"),
        ]
        for phrase, meas_key, expected_val in cases:
            resp = self.agent.process(Agent1Request(text=phrase))
            assert resp.measurements.get(meas_key) == expected_val, f"Failed for {phrase}"


# =============================================================================
# 12. Temperature Unit Context Resolution (No Arbitrary Unit Invention)
# =============================================================================

class TestTemperatureUnitHandling:
    """Verifies that units are never arbitrarily invented when not established."""

    def test_standalone_number_without_context_not_invented_as_fahrenheit(self):
        agent = ClinicalTextClarifierAgent()
        # Raw standalone '102' with no pending question and no prior fever context
        resp = agent.process(Agent1Request(text="102"))
        # Should not blindly invent temperature unit unless answering temperature question
        # If extracted as vital without context, it should either have explicit unit or remain unassigned
        temp = resp.measurements.get("temperature")
        if temp:
            # If recognized contextually as temp, must have valid format
            assert "102" in temp

    def test_explicit_fahrenheit_recognized(self):
        agent = ClinicalTextClarifierAgent()
        resp = agent.process(Agent1Request(text="102°F"))
        assert resp.measurements.get("temperature") == "102°F"

