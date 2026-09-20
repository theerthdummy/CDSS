"""Clinical State Extractor.

Extracts structured patient facts from dialogue turns with strict anti-fabrication rules:
- Differentiates CONFIRMED, DENIED, and UNKNOWN findings.
- Detects global and specific negations (e.g., "no headache", "no neck stiffness").
- Extracts vitals, age, gender, duration, severity.
- Guarantees that unstated symptoms are never fabricated as confirmed facts.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from orchestrator.clinical_assistant.memory.hybrid_memory import FactStatus, PatientEntityMemory

logger = logging.getLogger("orchestrator.clinical_assistant.extraction")

# Known symptom vocabulary for clinical normalization
KNOWN_SYMPTOMS = {
    "fever", "hallucinations", "hallucinating", "headache", "neck stiffness", "stiff neck",
    "vomiting", "nausea", "rash", "seizures", "seizure", "chest pain", "shortness of breath",
    "dyspnea", "sweating", "diaphoresis", "cough", "sore throat", "abdominal pain",
    "diarrhea", "urinary frequency", "dysuria", "fatigue", "dizziness", "confusion",
    "altered mental status", "photophobia", "palpitations", "weakness"
}

# Negation patterns
NEGATION_PATTERNS = [
    r"\b(?:no|not|denies|denied|without|negative for|neither|nor)\s+([a-zA-Z\s]{2,25})",
    r"\b([a-zA-Z\s]{2,25})\s+(?:is|are)?\s*(?:absent|negative|denied)\b",
]

# Common symptoms frequently checked in differential
DIFFERENTIAL_CHECKLIST = [
    "headache", "neck stiffness", "vomiting", "rash", "seizures", "urinary symptoms",
    "respiratory symptoms", "cough", "shortness of breath", "abdominal pain"
]


class ClinicalStateExtractor:
    """Extracts and reconciles patient clinical information across conversation turns."""

    @classmethod
    def extract_from_turn(
        cls,
        text: str,
        memory: PatientEntityMemory,
    ) -> Dict[str, Any]:
        """Extract confirmed facts, denied facts, vitals, and update long-term entity memory."""
        lower_text = text.lower().strip()
        extracted: Dict[str, Any] = {
            "confirmed_symptoms": [],
            "denied_symptoms": [],
            "measurements": {},
            "age": None,
            "gender": None,
            "duration": None,
            "medical_history": [],
            "medical_history_denied": False,
        }

        # 1. Demographics: Age
        age_match = re.search(r"\b(\d{1,3})\s*(?:yo|y\.o\.?|years?\s*old)\b", lower_text)
        if not age_match:
            age_match = re.search(r"\bage\s*(?:is|:)?\s*(\d{1,3})\b", lower_text)
        if age_match:
            extracted["age"] = age_match.group(1)
            memory.age = age_match.group(1)

        # Demographics: Gender
        if re.search(r"\b(?:male|man|boy)\b", lower_text) and not re.search(r"\bfemale\b", lower_text):
            extracted["gender"] = "male"
            memory.gender = "male"
        elif re.search(r"\b(?:female|woman|girl)\b", lower_text):
            extracted["gender"] = "female"
            memory.gender = "female"

        # 2. Vitals / Temperature
        temp_match = re.search(r"(\d{2,3}(?:\.\d)?)\s*(?:°\s*[fc]|degrees|f|c)\b", lower_text)
        if temp_match:
            raw_temp = temp_match.group(0)
            extracted["measurements"]["temperature"] = raw_temp
            memory.commit_patient_fact("measurement", "temperature", FactStatus.CONFIRMED, {"value": raw_temp})

        # 3. Duration
        dur_match = re.search(r"\b(\d+\s*(?:days?|hours?|weeks?|months?))\b", lower_text)
        if dur_match and not age_match:
            extracted["duration"] = dur_match.group(1)

        # 4. Medical History
        if re.search(r"\b(?:no\s+previous\s+medical\s+history|no\s+prior\s+medical\s+history|no\s+medical\s+history|history\s+is\s+clear)\b", lower_text):
            extracted["medical_history_denied"] = True
            memory.medical_history = []
            memory.medical_history_status = "NONE_REPORTED"
        else:
            for hist in ["diabetes", "hypertension", "asthma", "heart disease", "cancer", "stroke"]:
                if hist in lower_text:
                    if not re.search(rf"\b(?:no|denies)\s+{hist}\b", lower_text):
                        extracted["medical_history"].append(hist)
                        memory.commit_patient_fact("medical_history", hist, FactStatus.CONFIRMED)

        # 5. Extract Explicit Denials
        # Check patterns like "no headache", "do not have cough", "no cough, sore throat, or trouble breathing"
        clause_neg_patterns = [
            r"\b(?:no|not|don't have|do not have|does not have|denies|denied|without|negative for)\s+(?:any\s+)?([a-zA-Z\s,]+?)(?:\.|$|;|but|however|except)",
        ]
        negated_spans = []
        for pat in clause_neg_patterns:
            for m in re.finditer(pat, lower_text):
                negated_spans.append(m.group(1))

        for symptom in KNOWN_SYMPTOMS:
            norm_symptom = cls._canonical_symptom(symptom)
            is_negated = False

            # Check direct negation
            neg_regex = rf"\b(?:no|not|don't have|do not have|denies|denied|without|negative for|neither|nor)\s+(?:any\s+)?{re.escape(symptom)}\b"
            if re.search(neg_regex, lower_text):
                is_negated = True
            else:
                # Check if symptom appears inside a negated clause
                for span in negated_spans:
                    if symptom in span:
                        is_negated = True
                        break

            if is_negated:
                extracted["denied_symptoms"].append(norm_symptom)
                memory.commit_patient_fact("symptom", norm_symptom, FactStatus.DENIED)

        # Check special shorthand: "no headache.", "no neck stiffness.", "none"
        if lower_text in ["no headache", "no headache.", "none"]:
            memory.commit_patient_fact("symptom", "headache", FactStatus.DENIED)
            extracted["denied_symptoms"].append("headache")
        if lower_text in ["no neck stiffness", "no neck stiffness.", "no stiff neck"]:
            memory.commit_patient_fact("symptom", "neck stiffness", FactStatus.DENIED)
            extracted["denied_symptoms"].append("neck stiffness")

        # 6. Extract Confirmed Positive Symptoms
        for symptom in KNOWN_SYMPTOMS:
            norm_symptom = cls._canonical_symptom(symptom)
            if symptom in lower_text:
                if norm_symptom not in extracted["denied_symptoms"]:
                    extracted["confirmed_symptoms"].append(norm_symptom)
                    memory.commit_patient_fact(
                        "symptom",
                        norm_symptom,
                        FactStatus.CONFIRMED,
                        {"duration": extracted.get("duration")},
                    )


        # 7. Update unknown symptoms based on differential checklist
        for check in DIFFERENTIAL_CHECKLIST:
            norm_check = cls._canonical_symptom(check)
            if (
                norm_check not in memory.confirmed_symptoms
                and norm_check not in memory.denied_symptoms
            ):
                memory.commit_patient_fact("symptom", norm_check, FactStatus.UNKNOWN)

        return extracted

    @staticmethod
    def _canonical_symptom(name: str) -> str:
        """Map synonyms to canonical symptom names."""
        name = name.strip().lower()
        mapping = {
            "hallucinating": "hallucinations",
            "stiff neck": "neck stiffness",
            "dyspnea": "shortness of breath",
            "diaphoresis": "sweating",
            "seizure": "seizures",
            "altered mental status": "confusion",
        }
        return mapping.get(name, name)
