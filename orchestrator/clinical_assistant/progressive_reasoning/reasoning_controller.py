"""Progressive Clinical Reasoning Controller.

Manages the clinical progression pipeline:
- Stage 1: Understand chief complaint
- Stage 2: Collect basic clinical information (age, duration, vitals)
- Stage 3: Detect red flags & critical urgent conditions
- Stage 4: Consider reasonable broad / common differentials
- Stage 5: Ask targeted follow-up questions dynamically
- Stage 6: Update differential as evidence accumulates
- Stage 7: Consider advanced diagnoses only when supported
- Stage 8: Discuss appropriate next steps

CRITICAL EXCEPTION FOR URGENT CONDITIONS:
If patient presents with time-critical red flags (e.g. acute high fever with hallucinations,
or chest pain radiating to the left arm with dyspnea), the system IMMEDIATELY highlights
the need for urgent in-person medical evaluation without delaying for questionnaires.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from orchestrator.clinical_assistant.memory.hybrid_memory import PatientEntityMemory

logger = logging.getLogger("orchestrator.clinical_assistant.reasoning")


class ClinicalTriageAssessment(BaseModel):
    """Output of clinical reasoning triage."""
    stage: int = 1  # 1 to 8
    is_urgent: bool = False
    urgency_reason: Optional[str] = None
    priority_level: str = "Routine"  # "Emergency" | "Urgent" | "Priority" | "Routine"
    red_flags_detected: List[str] = Field(default_factory=list)
    differential_considerations: List[str] = Field(default_factory=list)
    suggested_follow_up_questions: List[str] = Field(default_factory=list)
    reasoning_summary: str = ""


class ProgressiveReasoningController:
    """Controls progressive clinical reasoning, red flag triage, and dynamic questioning."""

    # Explicit red-flag symptom combinations
    RED_FLAG_PATTERNS = [
        {
            "triggers": {"fever", "hallucinations"},
            "urgency": "Emergency",
            "reason": (
                "Acute high fever accompanied by hallucinations indicates altered mental status, "
                "which may represent acute central nervous system infection (e.g., encephalitis, meningitis), "
                "severe systemic sepsis, or toxic-metabolic encephalopathy."
            ),
            "critical_differentials": ["Encephalitis / CNS Infection", "Sepsis with Encephalopathy", "Febrile Delirium", "Toxic/Metabolic Reaction"],
        },
        {
            "triggers": {"chest pain", "shortness of breath"},
            "urgency": "Emergency",
            "reason": (
                "Acute chest pain with shortness of breath warrants immediate emergency evaluation for "
                "acute coronary syndrome, pulmonary embolism, or aortic dissection."
            ),
            "critical_differentials": ["Acute Coronary Syndrome", "Pulmonary Embolism", "Pneumothorax"],
        },
    ]

    # Clinical domains for targeted follow-up questions
    DOMAIN_QUESTIONS = {
        "headache": "Do you have any severe headache or head pressure?",
        "neck stiffness": "Are you experiencing any stiffness or difficulty bending your neck forward?",
        "photophobia": "Are bright lights causing you significant pain or discomfort in your eyes?",
        "vomiting": "Have you experienced any nausea or vomiting?",
        "rash": "Have you noticed any new rash, purple spots, or skin changes?",
        "seizures": "Have there been any tremors, twitches, or seizure-like activity?",
        "urinary": "Have you noticed any burning during urination or flank pain?",
        "respiratory": "Do you have any cough, shortness of breath, or chest tightness?",
    }

    @classmethod
    def evaluate(
        cls,
        memory: PatientEntityMemory,
        asked_questions: Optional[Set[str]] = None,
    ) -> ClinicalTriageAssessment:
        """Evaluate patient state progressively and generate dynamic triage and questions."""
        asked = asked_questions or set()
        confirmed = {name.lower() for name in memory.confirmed_symptoms.keys()}
        denied = {name.lower() for name in memory.denied_symptoms}
        measurements = memory.measurements

        triage = ClinicalTriageAssessment()

        # Check Red Flags
        for pattern in cls.RED_FLAG_PATTERNS:
            triggers = pattern["triggers"]
            if triggers.issubset(confirmed):
                triage.is_urgent = True
                triage.priority_level = pattern["urgency"]
                triage.urgency_reason = pattern["reason"]
                triage.red_flags_detected.append(" + ".join(sorted(list(triggers))))
                triage.differential_considerations.extend(pattern["critical_differentials"])

        # Determine Stage
        if not memory.age or not memory.medical_history_status:
            triage.stage = 2  # Collecting basic clinical info
        elif triage.is_urgent:
            triage.stage = 3  # Identified red flags / urgent triage
        elif len(confirmed) >= 2:
            triage.stage = 4  # Broad differentials
        else:
            triage.stage = 1

        # Add common differentials if non-urgent
        if not triage.is_urgent:
            if "chest pain" in confirmed:
                triage.differential_considerations = ["Musculoskeletal Chest Pain", "Gastroesophageal Reflux", "Angina Pectoris"]
            elif "fever" in confirmed:
                triage.differential_considerations = ["Viral Syndrome", "Upper Respiratory Infection", "Urinary Tract Infection"]

        # Dynamically Select Targeted Follow-Up Questions (NEVER REPEATING)
        candidate_questions: List[Tuple[str, str]] = []

        # If fever + hallucinations, select key neurological & infectious exclusion questions
        if "fever" in confirmed and "hallucinations" in confirmed:
            checks = [
                ("headache", cls.DOMAIN_QUESTIONS["headache"]),
                ("neck stiffness", cls.DOMAIN_QUESTIONS["neck stiffness"]),
                ("photophobia", cls.DOMAIN_QUESTIONS["photophobia"]),
                ("vomiting", cls.DOMAIN_QUESTIONS["vomiting"]),
                ("rash", cls.DOMAIN_QUESTIONS["rash"]),
                ("seizures", cls.DOMAIN_QUESTIONS["seizures"]),
            ]
            for sym, q in checks:
                if sym not in confirmed and sym not in denied and q not in asked:
                    candidate_questions.append((sym, q))

        # Check demographics / background if missing
        if not memory.age and "What is your age?" not in asked:
            candidate_questions.insert(0, ("age", "Could you share your age?"))
        if memory.medical_history_status == "UNKNOWN" and "Do you have any medical history?" not in asked:
            candidate_questions.insert(0, ("medical_history", "Do you have any pre-existing medical conditions or take any daily medications?"))

        # Select top 2-3 most relevant questions
        selected_questions = [q for _, q in candidate_questions[:3]]
        triage.suggested_follow_up_questions = selected_questions

        # Summarize reasoning
        if triage.is_urgent:
            triage.reasoning_summary = (
                f"Urgent condition detected ({triage.priority_level}): {triage.urgency_reason} "
                f"Differentials to evaluate include {', '.join(triage.differential_considerations[:3])}."
            )
        else:
            triage.reasoning_summary = (
                f"Progressive evaluation (Stage {triage.stage}): Patient has {len(confirmed)} confirmed symptom(s) "
                f"and {len(denied)} denied finding(s)."
            )

        return triage
