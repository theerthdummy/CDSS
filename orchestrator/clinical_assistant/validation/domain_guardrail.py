"""Domain Guardrail for Clinical Decision Support System.

Strictly enforces that the Clinical Assistant only operates within the clinical,
medical, pharmacological, and healthcare domain. Rejects all out-of-domain queries
(e.g., geography, trivia, coding, sports, weather, politics, recipes, etc.).
"""

import re
from typing import Optional, Tuple
from orchestrator.clinical_assistant.memory.hybrid_memory import PatientEntityMemory

# Explicit non-medical intent patterns (geography, astronomy, coding, pop culture, trivia, etc.)
OUT_OF_DOMAIN_PATTERNS = [
    # Astronomy, space & celestial distances (e.g. "how far is the moon from earth")
    r"\b(?:distance\s+to\s+the\s+moon|how\s+far\s+is\s+(?:the\s+)?moon|moon\s+from\s+earth|distance\s+to\s+(?:mars|sun|jupiter|saturn)|solar\s+system|speed\s+of\s+light|black\s+hole|galaxy|milky\s+way)\b",
    
    # Geography, cities, countries & terrestrial distances
    r"\b(?:distance\s+between|how\s+far\s+is\s+[a-zA-Z\s]+\s+from|flight\s+time\s+from|capital\s+of|population\s+of|where\s+is\s+[a-zA-Z\s]+located)\b",
    r"\b(?:continent|ocean|mountain\s+range|bordering\s+countries|timezone\s+of)\b",
    
    # Coding & technical computing
    r"\b(?:write\s+(?:python|javascript|typescript|code|script|sql|html|css|c\+\+|java|rust)|debug\s+this\s+code|regex\s+for|git\s+commit|git\s+push|npm\s+install)\b",
    
    # Math & physics calculations unrelated to medicine
    r"\b(?:solve\s+for\s+x|calculate\s+the\s+area|pythagorean|derivative\s+of|integral\s+of|quadratic\s+formula)\b",
    
    # Trivia, politics, sports, entertainment, movies, games
    r"\b(?:who\s+is\s+the\s+president|prime\s+minister|who\s+won\s+the|world\s+cup|super\s+bowl|nba|olympics|champions\s+league)\b",
    r"\b(?:box\s+office|movie\s+plot|actor\s+in|who\s+directed|lyrics\s+to|album\s+by|video\s+game)\b",
    
    # Cooking, recipes, food preparation
    r"\b(?:recipe\s+for|how\s+to\s+cook|how\s+to\s+bake|ingredients\s+for\s+cake|pasta\s+recipe)\b",
    
    # Weather & finance
    r"\b(?:weather\s+in|forecast\s+for|stock\s+price|bitcoin\s+price|crypto\s+rate|exchange\s+rate)\b",
]

# Medical and clinical terminology indicating valid in-domain queries
# Extensively includes colloquial, subjective, and vague patient symptom expressions
MEDICAL_KEYWORDS = {
    # Subjective feelings & vague health descriptions
    "sick", "unwell", "ill", "illness", "hurt", "hurts", "hurting", "ache", "aches",
    "aching", "ached", "pain", "pains", "painful", "discomfort", "uneasy", "weird",
    "off", "bad", "awful", "terrible", "poorly", "queasy", "nauseous", "nausea",
    "shiver", "shivers", "shivering", "fever", "feverish", "hot", "cold", "burning",
    "clammy", "sweat", "sweating", "sweats", "sweaty", "diaphoresis", "chill", "chills",
    "dizzy", "dizziness", "lightheaded", "faint", "fainting", "syncope", "passed out",
    "blackout", "tired", "tiredness", "exhausted", "exhaustion", "fatigue", "weak",
    "weakness", "drained", "lethargic", "lethargy", "groggy", "dazed", "confused",
    "confusion", "foggy", "fog", "malaise", "achey",
    
    # Anatomy & body areas (vague or specific)
    "head", "forehead", "temple", "eye", "eyes", "vision", "ear", "ears", "hearing",
    "nose", "throat", "mouth", "tongue", "jaw", "tooth", "teeth", "neck", "shoulder",
    "chest", "breast", "rib", "ribs", "back", "spine", "flank", "stomach", "tummy",
    "belly", "gut", "abdomen", "abdominal", "pelvis", "groin", "arm", "elbow", "wrist",
    "hand", "finger", "fingers", "leg", "thigh", "knee", "calf", "ankle", "foot",
    "feet", "toe", "toes", "skin", "muscle", "muscles", "joint", "joints", "bone",
    "heart", "lung", "lungs", "liver", "kidney", "kidneys", "brain", "bladder", "bowel",
    
    # Specific symptoms & clinical signs
    "headache", "migraine", "cough", "coughing", "breath", "breathing", "dyspnea",
    "shortness", "wheezing", "phlegm", "mucus", "congestion", "congested", "sneeze",
    "sneezing", "runny", "sore", "stiff", "stiffness", "rash", "itching", "itchy",
    "pruritus", "hives", "bump", "lumps", "lump", "swelling", "swollen", "edema",
    "vomit", "vomiting", "diarrhea", "constipation", "cramp", "cramps", "cramping",
    "bloating", "bloated", "gas", "reflux", "heartburn", "indigestion", "burn", "burning",
    "urination", "urinate", "urine", "pee", "dysuria", "hematuria", "stool", "poop",
    "seizure", "seizures", "convulsion", "hallucination", "hallucinations", "delusion",
    "palpitation", "palpitations", "fluttering", "numbness", "numb", "tingling",
    "paresthesia", "tremor", "tremors", "shaky", "shaking", "twitch", "twitching",
    "bleeding", "bleed", "blood", "bruise", "bruising", "wound", "cut", "injury", "injured",
    "fall", "fell", "lesion", "ulcer", "blister",
    
    # Measurements & vitals
    "temperature", "temp", "blood pressure", "bp", "pulse", "heart rate", "hr",
    "oxygen", "saturation", "spo2", "weight", "height", "bmi", "glucose", "sugar",
    "a1c", "degrees", "°f", "°c",
    
    # Disease, conditions, infections & categories
    "infection", "infectious", "viral", "virus", "bacterial", "bacteria", "pneumonia",
    "meningitis", "encephalitis", "sepsis", "infarction", "stroke", "diabetes",
    "hypertension", "asthma", "copd", "cancer", "tumor", "allergy", "allergic",
    "allergies", "anaphylaxis", "appendicitis", "hepatitis", "covid", "influenza",
    "flu", "cold", "strep", "syndrome", "disorder", "disease", "illness", "condition",
    "triage", "emergency", "urgent", "chronic", "acute",
    
    # Medications, treatments & healthcare terms
    "medication", "medications", "medicine", "medicines", "drug", "drugs", "antibiotic",
    "antibiotics", "aspirin", "ibuprofen", "advil", "motrin", "tylenol", "acetaminophen",
    "paracetamol", "penicillin", "steroid", "insulin", "dose", "dosage", "pill", "pills",
    "prescription", "prescribed", "side effect", "contraindication", "vaccine",
    "vaccination", "immunization", "shot", "injection", "therapy", "treatment",
    "hospital", "clinic", "er", "doctor", "nurse", "physician", "patient", "pediatric",
    "geriatric", "health", "healthy", "healthcare", "test", "labs", "scan", "x-ray", "mri",
}

# Conversational responses valid in clinical turn contexts
CLINICAL_CONVERSATIONAL_WORDS = {
    "yes", "no", "none", "neither", "nor", "both", "maybe", "sometimes",
    "constantly", "intermittently", "worse", "better", "mild", "severe", "moderate",
    "started", "days", "hours", "weeks", "yesterday", "today", "ago", "morning", "night",
    "left", "right", "sharp", "dull", "throbbing", "burning", "feels", "feeling",
}

# Standardized out-of-domain refusal message
OUT_OF_DOMAIN_RESPONSE = (
    "**Out of Domain**\n"
    "- I am a dedicated Clinical Decision Support Assistant designed solely for medical, symptom, and healthcare guidance.\n"
    "- I cannot answer non-medical questions such as general geography, trivia, math, or coding (such as distances to the moon, sports, or programming).\n\n"
    "**How I can help**\n"
    "- Please describe any patient symptoms, feelings of being unwell, clinical notes, medications, or health concerns, and I will be glad to assist you."
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

        # 1. Check for explicit out-of-domain patterns (geography, astronomy, coding, pop trivia, recipes, etc.)
        for pattern in OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, lower):
                return True, f"Matched out-of-domain pattern: {pattern}"

        # 2. Check if the text contains any medical keywords or clinical concepts
        words = set(re.findall(r"\b[a-z]{2,}\b", lower))
        has_medical_keyword = bool(words & MEDICAL_KEYWORDS)

        # Check for multi-word clinical phrases
        has_phrase = any(
            phrase in lower
            for phrase in [
                "feel sick", "feeling sick", "not feeling well", "feel unwell",
                "feels weird", "feels off", "don't feel good", "not feeling good",
                "feel bad", "feeling bad", "chest pain", "stomach ache", "short of breath",
                "blood pressure", "heart rate", "side effect", "high temp", "low temp"
            ]
        )

        # Check for numeric vitals/measurements (e.g., 102F, 120/80, 98.6)
        has_vitals = bool(re.search(r"\b(?:\d{2,3}(?:\.\d)?\s*(?:°\s*[fc]|degrees|f|c)\b|\d{2,3}\s*/\s*\d{2,3})", lower))

        if has_medical_keyword or has_phrase or has_vitals:
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
