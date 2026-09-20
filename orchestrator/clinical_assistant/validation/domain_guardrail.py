"""Domain Guardrail for Clinical Decision Support System.

Strictly enforces that the Clinical Assistant only operates within the clinical,
medical, pharmacological, and healthcare domain. Rejects all out-of-domain queries
(e.g., geography, trivia, coding, sports, weather, politics, recipes, etc.).
"""

import re
from typing import Optional, Tuple
from orchestrator.clinical_assistant.memory.hybrid_memory import PatientEntityMemory

# Explicit non-medical intent patterns (geography, coding, pop culture, trivia, etc.)
OUT_OF_DOMAIN_PATTERNS = [
    # Geography & distances
    r"\b(?:distance\s+between|how\s+far\s+is|flight\s+time\s+from|capital\s+of|population\s+of|where\s+is\s+[a-zA-Z\s]+located)\b",
    r"\b(?:continent|ocean|mountain|river|bordering\s+countries|timezone\s+of)\b",
    
    # Coding & technical computing
    r"\b(?:write\s+(?:python|javascript|code|script|sql|html|css|c\+\+|java)|debug\s+this\s+code|regex\s+for|git\s+commit)\b",
    
    # Math & physics calculations unrelated to medicine
    r"\b(?:solve\s+for\s+x|calculate\s+the\s+area|pythagorean|speed\s+of\s+light|derivative\s+of|integral\s+of)\b",
    
    # Trivia, politics, sports, entertainment, movies, games
    r"\b(?:who\s+is\s+the\s+president|prime\s+minister|who\s+won\s+the|world\s+cup|super\s+bowl|nba|olympics)\b",
    r"\b(?:box\s+office|movie\s+plot|actor\s+in|who\s+directed|lyrics\s+to|album\s+by)\b",
    
    # Cooking, recipes, food preparation
    r"\b(?:recipe\s+for|how\s+to\s+cook|how\s+to\s+bake|ingredients\s+for\s+cake|pasta\s+recipe)\b",
    
    # Weather & finance
    r"\b(?:weather\s+in|forecast\s+for|stock\s+price|bitcoin\s+price|crypto\s+rate|exchange\s+rate)\b",
]

# Medical and clinical terminology indicating valid in-domain queries
MEDICAL_KEYWORDS = {
    # Symptoms & findings
    "fever", "pain", "headache", "cough", "breath", "breathing", "dyspnea", "rash",
    "nausea", "vomiting", "vomit", "diarrhea", "stool", "constipation", "dizzy",
    "dizziness", "vertigo", "fatigue", "tired", "weakness", "lethargy", "chill", "chills",
    "sweat", "sweating", "diaphoresis", "swelling", "edema", "sore", "throat", "burn",
    "burning", "urination", "urinate", "urine", "dysuria", "hematuria", "itching",
    "pruritus", "seizure", "seizures", "convulsion", "hallucination", "hallucinations",
    "confusion", "confused", "stiff", "stiffness", "palpitation", "palpitations",
    "chest", "abdomen", "abdominal", "back", "flank", "joint", "muscle", "ear", "eye",
    "vision", "hearing", "numbness", "tingling", "paresthesia", "cramp", "cramps",
    
    # Measurements & vitals
    "temperature", "temp", "blood pressure", "bp", "pulse", "heart rate", "hr",
    "oxygen", "saturation", "spo2", "weight", "height", "bmi", "glucose", "a1c",
    "degrees", "°f", "°c",
    
    # Anatomy & organs
    "heart", "lung", "lungs", "liver", "kidney", "kidneys", "brain", "skin", "stomach",
    "bowel", "colon", "bladder", "spine", "neck", "arm", "leg", "foot", "hand", "finger",
    "artery", "vein", "nerve", "throat", "sinus", "sinuses", "pelvis",
    
    # Disease & conditions
    "infection", "infectious", "viral", "virus", "bacterial", "bacteria", "pneumonia",
    "meningitis", "encephalitis", "sepsis", "infarction", "stroke", "diabetes",
    "hypertension", "asthma", "copd", "cancer", "tumor", "allergy", "allergic",
    "anaphylaxis", "appendicitis", "hepatitis", "covid", "influenza", "flu", "migraine",
    "syndrome", "disorder", "disease", "illness", "condition", "triage",
    
    # Medications & treatments
    "medication", "medications", "medicine", "drug", "drugs", "antibiotic", "antibiotics",
    "aspirin", "ibuprofen", "tylenol", "acetaminophen", "paracetamol", "penicillin",
    "steroid", "insulin", "dose", "dosage", "prescription", "side effect", "contraindication",
    "vaccine", "vaccination", "immunization", "therapy", "treatment", "hospital", "clinic",
    "doctor", "nurse", "physician", "patient", "pediatric", "geriatric", "emergency", "urgent",
}

# Conversational responses valid in clinical turn contexts
CLINICAL_CONVERSATIONAL_WORDS = {
    "yes", "no", "none", "neither", "nor", "both", "maybe", "sometimes",
    "constantly", "intermittently", "worse", "better", "mild", "severe", "moderate",
    "started", "days", "hours", "weeks", "yesterday", "today", "ago",
}

# Standardized out-of-domain refusal message
OUT_OF_DOMAIN_RESPONSE = (
    "**Out of Domain**\n"
    "- I am a specialized Clinical Decision Support Assistant designed solely for medical and healthcare guidance.\n"
    "- I cannot answer non-medical questions such as general geography, trivia, math, or coding.\n\n"
    "**How I can help**\n"
    "- Please provide patient symptoms, clinical notes, medications, or health concerns, and I will be glad to assist you."
)


class DomainGuardrail:
    """Evaluates whether an incoming user query falls within the medical/clinical domain."""

    @classmethod
    def is_out_of_domain(cls, text: str, memory: Optional[PatientEntityMemory] = None) -> Tuple[bool, str]:
        """Check if query is non-medical or out-of-domain.
        
        Returns:
            (is_out_of_domain: bool, reason: str)
        """
        if not text or not text.strip():
            return False, "empty"

        lower = text.lower().strip()

        # 1. Check for explicit out-of-domain patterns (geography, coding, pop trivia, recipes, etc.)
        for pattern in OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, lower):
                return True, f"Matched out-of-domain pattern: {pattern}"

        # 2. Check if the text contains any medical keywords or clinical concepts
        words = set(re.findall(r"\b[a-z]{3,}\b", lower))
        has_medical_keyword = bool(words & MEDICAL_KEYWORDS)

        # Check for numeric vitals/measurements (e.g., 102F, 120/80, 98.6)
        has_vitals = bool(re.search(r"\b(?:\d{2,3}(?:\.\d)?\s*(?:°\s*[fc]|degrees|f|c)\b|\d{2,3}\s*/\s*\d{2,3})", lower))

        if has_medical_keyword or has_vitals:
            return False, "in_domain_medical"

        # 3. Check for standard clinical conversational turn replies in ongoing sessions
        # (e.g., "no headache", "yes for 2 days", "none of those", "since this morning")
        if memory and (memory.confirmed_symptoms or memory.measurements or memory.medical_history):
            # If in an active case, short replies to questions are valid clinical turn answers
            tokens = set(re.findall(r"\b[a-z]+\b", lower))
            if tokens & CLINICAL_CONVERSATIONAL_WORDS or len(lower.split()) <= 6:
                return False, "in_domain_conversational_reply"

        # 4. Check for pure polite greetings
        greeting_patterns = [
            r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|help(?:\s+me)?)[.!]?$",
            r"^(?:can\s+you\s+help\s+me|how\s+does\s+this\s+work)[?.]?$",
        ]
        for g in greeting_patterns:
            if re.match(g, lower):
                return False, "in_domain_greeting"

        # If it has 0 medical keywords, no vitals, no prior context, and is not a greeting -> OUT OF DOMAIN
        return True, "No clinical or medical entities detected in query"

    @classmethod
    def get_refusal_response(cls) -> str:
        """Returns the standardized, empathetic clinical refusal response."""
        return OUT_OF_DOMAIN_RESPONSE
