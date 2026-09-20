"""Context-Aware Deterministic Clinical Extractor.

Extracts structured clinical facts without an LLM when rules/patterns can
confidently explain the input:
- Age & Gender (with context guards against numbers used for vitals/durations)
- Contextual answers to pending questions (e.g., '55' -> age, 'female' -> gender, 'none' -> no history)
- Measurements (Temperature, Blood Pressure, Heart Rate, SpO2)
- Explicit medical history and negation of medical history (global and entity-specific)
- Explicit medications and negation of medications (global and entity-specific)
- Positive symptoms vs. explicitly denied symptoms
- Duration & severity
- Evaluates whether the input is fully understood deterministically (Level 1) or requires LLM semantic interpretation (Level 2).

Routing:
- classify_input_complexity() → "level1" (single atomic field) | "level2" (multi-field / narrative)
  Every user message is independently classified on each turn.
"""

import re
from typing import Dict, Any, List, Optional, Tuple

KNOWN_SYMPTOM_PATTERNS: Dict[str, str] = {
    "fever": r"\bfevers?\b",
    "headache": r"\bheadaches?\b",
    "cough": r"\bcoughs?(?:ing)?\b",
    "chest pain": r"\bchest\s+pain\b|\bchest\s+discomfort\b|\bchest\s+pressure\b",
    "body pain": r"\bbody\s+pains?\b",
    "body ache": r"\bbody\s+aches?\b|\bmuscle\s+aches?\b|\bmyalgia\b",
    "nausea": r"\bnausea\b",
    "vomiting": r"\bvomit(?:ing|s)?\b|\bpuk(?:ing|e)\b|\bthrow(?:ing)?\s+up\b",
    "diarrhea": r"\bdiarrhea\b|\bloose\s+stools?\b",
    "chills": r"\bchills?\b",
    "fatigue": r"\bfatigue\b|\btired(?:ness)?\b|\bexhaust(?:ion|ed)\b",
    "weakness": r"\bweak(?:ness)?\b",
    "dizziness": r"\bdizz(?:y|iness)\b|\blightheaded(?:ness)?\b",
    "numbness": r"\bnumb(?:ness)?\b",
    "tingling": r"\btingl(?:ing)?\b",
    "sore throat": r"\bsore\s+throat\b|\bthroat\s+hurts\b|\bthroat\s+pain\b",
    "runny nose": r"\brunny\s+nose\b|\brhinorrhea\b",
    "congestion": r"\bcongestion\b|\bstuffy\s+nose\b|\bnasal\s+congestion\b",
    "shortness of breath": r"\bshortness\s+of\s+breath\b|\bdyspnea\b|\bbreathing\s+difficulty\b|\bsob\b",
    "palpitations": r"\bpalpitations?\b",
    "rash": r"\brash(?:es)?\b",
    "swelling": r"\bswelling\b|\bedema\b",
    "sweating": r"\bsweat(?:ing)?\b|\bdiaphoresis\b",
    "back pain": r"\bback\s+pain\b",
    "abdominal pain": r"\babdominal\s+pain\b|\bstomach\s+pain\b|\bbelly\s+pain\b|\bpain\s+in\s+the\s+(?:lower\s+|upper\s+)?(?:right|left)?\s*abdomen\b",
    "joint pain": r"\bjoint\s+pains?\b|\barthr(?:itis|algia)\b",
    "fainting": r"\bfaint(?:ing|ed)?\b|\bsyncope\b",
    "decreased appetite": r"\b(?:decreased|loss\s+of|poor)\s+appetite\b|\banorexia\b",
}

NEGATIVE_CONFIRMATION_PATTERNS = [
    r"^none(?:\s+reported|\s+at\s+all)?$",
    r"^no$",
    r"^no\s+(?:prior\s+|past\s+|known\s+)?(?:medical\s+)?history$",
    r"^no\s+(?:known\s+)?(?:medical\s+)?conditions?$",
    r"^no\s+(?:previous\s+|prior\s+)?diagnos(?:is|es)$",
    r"^no\s+prior\s+medical\s+history$",
    r"^i\s+don'?t\s+have\s+any(?:\s+medical\s+conditions?)?$",
    r"^i\s+don'?t\s+take\s+any(?:\s+medications?|\s+meds?)?$",
    r"^no\s+(?:current\s+)?medications?$",
    r"^no\s+(?:current\s+)?meds?$",
    r"^no\s+history\s*[,;]?\s*(?:and\s+)?no\s+(?:medications?|meds?)$",
    r"^not\s+taking\s+any(?:\s+medications?|\s+meds?)?$",
    r"^no\s+known\s+conditions?$",
    r"^negative$",
    r"^nothing$",
    r"^denies$",
]

UNKNOWN_CONFIRMATION_PATTERNS = [
    r"^i\s+don'?t\s+know(?:\s+my\s+(?:medical\s+)?history|\s+my\s+medications?)?$",
    r"^i'?m\s+not\s+sure(?:\s+about\s+my\s+(?:medical\s+)?history|\s+about\s+medications?)?$",
    r"^not\s+sure$",
    r"^don'?t\s+know$",
    r"^unknown$",
    r"^unsure$",
    r"^not\s+certain$",
]

COMMON_CONDITIONS = [
    "diabetes", "diabetes mellitus", "type 1 diabetes", "type 2 diabetes", "diabetic",
    "hypertension", "high blood pressure", "htn",
    "asthma", "copd", "chronic obstructive pulmonary disease",
    "coronary artery disease", "cad", "heart disease", "myocardial infarction",
    "stroke", "tia", "hyperlipidemia", "high cholesterol",
    "chronic kidney disease", "ckd", "cancer", "thyroid disease", "hypothyroidism",
    "anemia", "iron-deficiency anemia", "iron deficiency anemia",
    "gerd", "acid reflux", "depression", "anxiety", "arthritis", "atrial fibrillation", "afib"
]

COMMON_MEDICATIONS = [
    "metformin", "amlodipine", "lisinopril", "atorvastatin", "simvastatin",
    "levothyroxine", "omeprazole", "losartan", "albuterol", "gabapentin",
    "hydrochlorothiazide", "metoprolol", "aspirin", "insulin", "ibuprofen",
    "paracetamol", "acetaminophen", "amoxicillin", "azithromycin", "prednisone",
    "cough syrup", "albuterol inhaler", "blood pressure tablet", "blood pressure medication",
    "diabetes medication"
]

# ---------------------------------------------------------------------------
# Input Complexity Router
# ---------------------------------------------------------------------------

_MULTI_FIELD_SIGNALS: list = [
    # Age + demographic combo words
    r"\b(?:i['']?m|im|i\s+am|patient\s+is)\s+\d{1,3}\s*(?:year|yr|y\.?o|male|female|man|woman)",
    r"\b\d{1,3}[- ]?(?:year[- ]old|y\.?o\.?|yo)\s+(?:male|female|man|woman)",
    # Multiple symptom words
    r"\b(?:and|also|plus|with)\b.{0,60}\b(?:fever|cough|pain|ache|nausea|vomiting|chills|fatigue|dizz|breath|swelling|rash)\b",
    # Symptom + duration together
    r"\b(?:fever|cough|pain|ache|nausea|vomiting|chills|fatigue|headache|rash)\b.{0,80}\b(?:for|since|over|past|ago|started|days?|weeks?|hours?)\b",
    # Symptom + measurement together
    r"\b(?:fever|temperature|temp)\b.{0,40}\b\d{2,3}(?:\.\d+)?(?:\s*°?\s*[FCfc])?\b",
    # Medical history + symptom together
    r"\b(?:history\s+of|hx\s+of|known|diagnosed\s+with|has\s+(?:diabetes|hypertension|asthma|copd|htn|dm))\b.{0,80}\b(?:fever|cough|pain|symptoms|presenting)\b",
    # Medication + symptom together
    r"\b(?:taking|on|uses?)\b.{0,40}\b(?:fever|cough|pain|ache|nausea)\b",
    # Clinical note shorthand with multiple fields: "48M with HTN, fever..."
    r"\b\d{1,3}\s*[MFmf]\b.*\b(?:with|hx|history|fever|cough|pain|sob|dyspnea)\b",
    # Conjunction between two clinical facts
    r"\b(?:fever|cough|pain|nausea|vomiting|headache|chills|rash|ache)\b.{0,60}\b(?:fever|cough|pain|nausea|vomiting|headache|chills|rash|ache|throat|breath)\b",
    # Any sentence-length multi-clause input
    r"\b(?:since|because|also|however|although|despite|started|began|noticed|presents\s+with)\b",
]

_MULTI_FIELD_RE = [re.compile(p, re.IGNORECASE) for p in _MULTI_FIELD_SIGNALS]

# Patterns that are clearly atomic / single-field inputs.
# IMPORTANT: Longer / more specific alternatives must come BEFORE shorter ones in the
# same alternation group, otherwise regex short-circuits (e.g. bare "no" wins before
# "no medical history").
_ATOMIC_PATTERNS: list = [
    r"^(?:i['']?m|i\s+am|age\s*(?:is|:)?\s*)?\d{1,3}(?:\s*(?:years?\s*old|yrs?|y\.?o\.?|yo))?$",  # pure age / age with unit
    r"^(?:male|female|woman|man|boy|girl|non-binary|f|m)$",                                           # pure gender
    r"^\d{2,3}(?:\.\d+)?\s*°?\s*[FCfc]?$",                                                           # pure temperature / number
    # Negation — longer phrases must appear before bare "no" to avoid short-circuiting
    r"^(?:none|negative|nothing"
    r"|not\s+taking\s+any(?:\s+meds?|\s+medications?)?"
    r"|no\s+(?:prior\s+|past\s+|known\s+)?(?:medical\s+)?history"
    r"|no\s+(?:current\s+)?(?:meds?|medications?)"
    r"|no\s+medications?"
    r"|no\s+history"
    r"|no)$",
    r"^(?:i\s+don'?t\s+know(?:\s+my\s+(?:medical\s+)?history|\s+my\s+medications?)?|i'?m\s+not\s+sure|not\s+sure|don'?t\s+know|unknown)$", # unknown
    r"^(?:hypertension|diabetes|asthma|copd|htn|dm\d?|t[12]dm|cad|chf|ckd|gerd|afib|osa|ptsd|bph|high\s+blood\s+pressure|high\s+cholesterol)$", # single known condition
    r"^(?:metformin|amlodipine|lisinopril|atorvastatin|aspirin|albuterol|insulin|ibuprofen|paracetamol|acetaminophen)$", # single medication
    r"^(?:fever|cough|headache|chest\s+pain|nausea|vomiting|dizziness|chills|fatigue|weakness|rash)$", # single symptom
    r"^(?:for\s+\d+\s+days?|since\s+yesterday|since\s+last\s+night|\d+\s+days?|\d+\s+hours?|\d+\s+weeks?)$", # single duration
    r"^(?:yes|yeah|yep|correct|true|nope)$", # direct boolean confirmation (bare "no" handled in negation pattern above)
]
_ATOMIC_RE = [re.compile(p, re.IGNORECASE) for p in _ATOMIC_PATTERNS]


def classify_input_complexity(
    text: str,
    pending_question: Optional[str] = None,
) -> str:
    """Classify user input as 'level1' (atomic / single-field) or 'level2' (multi-field / narrative).

    Decision rules (evaluated in order):
    1. If there is a pending question AND the input looks like a direct answer → level1.
    2. If the input matches any known atomic single-field pattern            → level1.
    3. If the input matches any multi-field signal                           → level2.
    4. Word-count heuristic: 6+ content words with mixed clinical signals   → level2.
    5. Default                                                               → level1.

    Returns:
        "level1"  – handle with deterministic extractor only (0 LLM).
        "level2"  – handle with LLM semantic interpretation.
    """
    if not text:
        return "level1"

    clean = text.strip().lower().rstrip(".!?,")

    # ── Rule 1: Direct answer to a pending question ─────────────────────────
    if pending_question:
        word_count = len(clean.split())
        has_extra_clinical = any(
            re.search(p, clean) for p in [
                r"\b(?:fever|cough|pain|ache|nausea|vomiting|chills|headache|rash|dyspnea|sob)\b.{0,40}\b(?:fever|cough|pain|ache|nausea|vomiting|chills|headache|rash|dyspnea|sob)\b",
                r"\b(?:fever|cough|pain)\b.{0,40}\b(?:for|since|ago|days?|weeks?|hours?)\b",
                r"\b(?:history|hx|diagnosed|taking|on\s+medication)\b",
            ]
        )
        if word_count <= 6 and not has_extra_clinical:
            return "level1"

    # ── Rule 2: Known atomic patterns ────────────────────────────────────────
    for pat in _ATOMIC_RE:
        if pat.match(clean):
            return "level1"

    # ── Rule 3: Multi-field signals ──────────────────────────────────────────
    for pat in _MULTI_FIELD_RE:
        if pat.search(clean):
            return "level2"

    # ── Rule 4: Word-count + clinical content heuristic ──────────────────────
    field_count = 0
    if re.search(r"\b\d{1,3}\s*(?:years?\s*old|y\.?o|yo|yr)\b|\b(?:i['']?m|im)\s+\d{1,3}\b", clean):
        field_count += 1
    if re.search(r"\b(?:male|female|man|woman|boy|girl)\b", clean):
        field_count += 1
    if re.search(r"\b(?:fever|cough|pain|ache|nausea|vomiting|chills|headache|rash|fatigue|dizz)\b", clean):
        field_count += 1
    if re.search(r"\b\d{2,3}(?:\.\d+)?\s*°?\s*[FCfc]\b|\btemperature\b|\btemp\b", clean):
        field_count += 1
    if re.search(r"\b(?:for|since|ago|days?|weeks?|hours?|months?)\b", clean):
        field_count += 1
    if re.search(r"\b(?:history|hx|hypertension|diabetes|asthma|htn|dm)\b", clean):
        field_count += 1

    if field_count >= 2:
        return "level2"

    # ── Rule 5: Default ───────────────────────────────────────────────────────
    return "level1"


def _empty_extracted_dict() -> Dict[str, Any]:
    return {
        "age": None,
        "gender": None,
        "medical_history": [],
        "denied_history": [],
        "medical_history_status": "UNKNOWN",
        "medications": [],
        "denied_medications": [],
        "medications_status": "UNKNOWN",
        "symptoms": [],
        "denied_symptoms": [],
        "measurements": {},
        "duration": None,
        "severity": None,
        "other_information": [],
        "relationships": [],
    }


def extract_duration_phrase(text: str) -> Optional[str]:
    """Extract clinical duration expression from text."""
    if not text:
        return None
    clean = text.lower().strip()

    # Range with units: "2-3 weeks", "2 to 3 days", "2 - 3 wks", "1-2 months", "4-5 hours", "1-2 years"
    range_unit_pat = r'\b(\d+\s*(?:-|to|or)\s*\d+\s*(?:days?|hours?|hrs?|weeks?|wks?|months?|mos?|minutes?|mins?|years?|yrs?))\b'
    # Ago pattern: "3 days ago", "2-3 weeks ago", "about 6 hours ago", "seven days ago"
    ago_pat = r'\b((?:about\s+|approx\s+|approximately\s+)?\d+\s*(?:-|to|or)?\s*\d*\s*(?:days?|hours?|hrs?|weeks?|wks?|months?|mins?)\s+ago)\b'
    # Single unit pattern with prefix: "for 3 days", "about 2 hours", "for one week"
    for_unit_pat = r'\b((?:for|over|past|about|approx|approximately)\s+\d+\s*(?:days?|hours?|hrs?|weeks?|wks?|months?|mos?|minutes?|mins?|years?|yrs?))\b'
    # Single unit pattern (guard against 'X years old' or 'X yo')
    single_unit_pat = r'\b(\d+\s*(?:days?|hours?|hrs?|weeks?|wks?|months?|mos?|minutes?|mins?))\b'
    # Years duration only if explicit duration context like 'for 2 years'
    years_duration_pat = r'\b(?:for|over|past|since|x)\s+(\d+\s*(?:years?|yrs?))\b'
    # Relative expressions: "a couple of days", "few weeks", "several days", "for a while"
    relative_pat = r'\b((?:for\s+)?(?:a\s+)?(?:couple|few|several|a\s+while)\s*(?:of\s+)?(?:days?|weeks?|wks?|months?|hours?|hrs?)?)\b'
    # Since expressions: "since yesterday", "since last week", "since 2 days", "since last night"
    since_pat = r'\b(since\s+(?:yesterday|last\s+night|this\s+morning|last\s+week|last\s+month|\d+\s*(?:days?|weeks?|months?|hours?))|since\s+yesterdy)\b'
    # Fixed time points
    time_points = r'\b(yesterday|last\s+night|this\s+morning)\b'

    for pat in [range_unit_pat, ago_pat, for_unit_pat, years_duration_pat, single_unit_pat, relative_pat, since_pat, time_points]:
        m = re.search(pat, clean)
        if m:
            val = m.group(1).strip()
            # Guard: ensure not part of 'X years old' or 'X yo'
            age_m = re.search(r'\b(\d+)\s*(?:years?\s*old|y\.?o\.?|yo)\b', clean)
            if age_m and val.startswith(age_m.group(1)) and ('year' in val or 'yr' in val):
                continue
            return val
    return None


def extract_contextual_answer(
    text: str,
    pending_question: Optional[str],
    current_state: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Handle direct, concise answers to the immediately preceding question."""
    if not pending_question:
        return None, False

    clean = text.strip().lower().rstrip(".!?,")
    q_lower = pending_question.lower()

    # 1. Answering Age Question
    if "age" in q_lower or "how old" in q_lower:
        age_match = re.match(r"^(?:patient\s+is\s+|i\s+am\s+|age\s*(?:is|:)?\s*)?(\d{1,3})(?:\s*years?\s*old|\s*yrs?|\s*yo)?$", clean)
        if age_match:
            val = int(age_match.group(1))
            if 0 <= val <= 130:
                res = _empty_extracted_dict()
                res["age"] = str(val)
                return res, True

    # 2. Answering Gender Question
    if "gender" in q_lower or "sex" in q_lower:
        if clean in ["female", "woman", "girl", "f", "i'm a woman", "i am female", "i'm female"]:
            res = _empty_extracted_dict()
            res["gender"] = "female"
            return res, True
        if clean in ["male", "man", "boy", "m", "i'm a man", "i am male", "i'm male"]:
            res = _empty_extracted_dict()
            res["gender"] = "male"
            return res, True
        if clean in ["non-binary", "nonbinary"]:
            res = _empty_extracted_dict()
            res["gender"] = "non-binary"
            return res, True

    # 3. Answering Medical History / Diagnoses Question
    if "medical condition" in q_lower or "previous diagnos" in q_lower or "medical history" in q_lower or "history of" in q_lower:
        # Check negative confirmation
        for pat in NEGATIVE_CONFIRMATION_PATTERNS:
            if re.match(pat, clean):
                res = _empty_extracted_dict()
                res["medical_history_status"] = "NONE_REPORTED"
                if "medication" in q_lower or "treatment" in q_lower:
                    res["medications_status"] = "NONE_REPORTED"
                return res, True

        # Check unknown confirmation
        for pat in UNKNOWN_CONFIRMATION_PATTERNS:
            if re.match(pat, clean):
                res = _empty_extracted_dict()
                res["medical_history_status"] = "UNKNOWN"
                res["other_information"] = ["medical history unknown to patient"]
                return res, True

        # Check affirmative answer to specific condition question: e.g. "Does the patient have a history of hypertension?" + "Yes"
        if clean in ["yes", "yeah", "yep", "correct", "true", "i do", "she does", "he does"]:
            for cond in COMMON_CONDITIONS:
                if cond in q_lower:
                    res = _empty_extracted_dict()
                    res["medical_history"] = [cond]
                    res["medical_history_status"] = "PRESENT"
                    return res, True

        # Check direct single condition name in answer
        for cond in COMMON_CONDITIONS:
            if clean == cond or clean == f"history of {cond}" or clean == f"i have {cond}":
                res = _empty_extracted_dict()
                res["medical_history"] = [cond]
                res["medical_history_status"] = "PRESENT"
                return res, True

    # 4. Answering Medication Question
    if "medication" in q_lower or "prescript" in q_lower or "taking any" in q_lower:
        for pat in NEGATIVE_CONFIRMATION_PATTERNS:
            if re.match(pat, clean):
                res = _empty_extracted_dict()
                res["medications_status"] = "NONE_REPORTED"
                return res, True

        for pat in UNKNOWN_CONFIRMATION_PATTERNS:
            if re.match(pat, clean):
                res = _empty_extracted_dict()
                res["medications_status"] = "UNKNOWN"
                res["other_information"] = ["medication history unknown to patient"]
                return res, True

        # Check affirmative to specific medication
        if clean in ["yes", "yeah", "yep", "correct", "true"]:
            for med in COMMON_MEDICATIONS:
                if med in q_lower:
                    res = _empty_extracted_dict()
                    res["medications"] = [med]
                    res["medications_status"] = "PRESENT"
                    return res, True

        # Check direct single medication name in answer
        for med in COMMON_MEDICATIONS:
            if clean == med or clean == f"taking {med}" or clean == f"on {med}":
                res = _empty_extracted_dict()
                res["medications"] = [med]
                res["medications_status"] = "PRESENT"
                return res, True

    # 5. Answering Temperature Question
    if "temperature" in q_lower or "temp" in q_lower:
        temp_match = re.match(r"^(?:temp(?:erature)?\s*(?:is\s*)?)?(\d{2,3}(?:\.\d+)?)\s*(?:°?\s*([fcFC]))?$", clean)
        if temp_match:
            num_str = temp_match.group(1)
            num = float(num_str)
            unit = temp_match.group(2)
            if unit:
                formatted_temp = f"{num_str}°{unit.upper()}"
            else:
                # When unit is not provided but answering a temperature question:
                # Follow clinical scale convention (>=50 is Fahrenheit, <50 is Celsius)
                unit_str = "F" if num >= 50 else "C"
                formatted_temp = f"{num_str}°{unit_str}"
            res = _empty_extracted_dict()
            res["measurements"]["temperature"] = formatted_temp
            return res, True

    # 6. Answering Duration Question
    if "how long" in q_lower or "duration" in q_lower or "when did" in q_lower:
        dur = extract_duration_phrase(clean)
        if dur:
            res = _empty_extracted_dict()
            res["duration"] = dur
            return res, True

    # 7. Answering Symptoms Question
    if "symptom" in q_lower or "complaint" in q_lower:
        for sym_name, sym_pat in KNOWN_SYMPTOM_PATTERNS.items():
            if re.search(sym_pat, clean):
                res = _empty_extracted_dict()
                res["symptoms"] = [sym_name]
                return res, True

    return None, False


def extract_deterministic_entities(
    text: str,
    pending_question: Optional[str] = None,
    current_state: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], bool]:
    """Deterministically extract all supported clinical entities from normalized text."""
    # First check contextual single-value response
    contextual_res, is_exact = extract_contextual_answer(text, pending_question, current_state)
    if is_exact and contextual_res is not None:
        return contextual_res, True

    result = _empty_extracted_dict()
    text_lower = text.lower()

    # -------------------------------------------------------------
    # 1. MEASUREMENTS (Temperature, BP, HR, SpO2)
    # (Extract first so numbers with explicit units are not confused with age)
    # -------------------------------------------------------------
    # Explicit Temperature (requires °F, °C, deg F, or temp keyword)
    temp_patterns = [
        r'(\d{2,3}(?:\.\d+)?)\s*°\s*([FCfc])\b',
        r'(\d{2,3}(?:\.\d+)?)\s*°?\s*f(?:ahrenheit)?\b',
        r'(\d{2,3}(?:\.\d+)?)\s*°?\s*c(?:elsius)?\b',
        r'(?:temp(?:erature)?\s*(?:is|of|around|like)?\s*)(\d{2,3}(?:\.\d+)?)\s*°?\s*([FCfc])?',
    ]
    for pat in temp_patterns:
        match = re.search(pat, text_lower)
        if match:
            temp_num = match.group(1)
            num_val = float(temp_num)
            unit = "F"
            if len(match.groups()) > 1 and match.group(2):
                unit = match.group(2).upper()
            elif "c" in match.group(0).lower():
                unit = "C"
            elif num_val < 50:
                unit = "C"
            result["measurements"]["temperature"] = f"{temp_num}°{unit}"
            break

    # Temperature Range: "100.4°F to 102.8°F"
    temp_range_match = re.search(r'(\d{2,3}(?:\.\d+)?\s*°?[FCfc])\s*(?:to|-)\s*(\d{2,3}(?:\.\d+)?\s*°?[FCfc])', text_lower)
    if temp_range_match:
        result["measurements"]["temperature"] = f"{temp_range_match.group(1)} - {temp_range_match.group(2)}"

    # Blood Pressure
    bp_match = re.search(r'\b(?:bp|blood pressure)?\s*(\d{2,3}\s*/\s*\d{2,3})\s*(?:mm\s*hg)?\b', text_lower)
    if bp_match and "/" in bp_match.group(1):
        clean_bp = bp_match.group(1).replace(" ", "")
        result["measurements"]["blood_pressure"] = f"{clean_bp} mmHg"

    # Heart Rate
    hr_match = re.search(r'\b(?:hr|heart rate|pulse)\s*(?:is|of)?\s*(\d{2,3})\s*(?:bpm)?\b', text_lower)
    if hr_match:
        result["measurements"]["heart_rate"] = f"{hr_match.group(1)} bpm"

    # SpO2 / Oxygen Saturation
    spo2_match = re.search(r'\b(?:spo2|o2 sat(?:uration)?|oxygen saturation)\s*(?:is|of)?\s*(\d{2,3})\s*%\b', text_lower)
    if spo2_match:
        result["measurements"]["spo2"] = f"{spo2_match.group(1)}%"

    # -------------------------------------------------------------
    # 2. AGE EXTRACTION
    # -------------------------------------------------------------
    age_patterns = [
        r"\b(?:i['’]?m|im|i am)\s+(?:a\s+)?(\d{1,3})\s*(?:-|\s)?(?:years?[- ]old|yrs?|yr|yo|y/o|male|female|man|woman)\b",
        r"\b(?:i['’]?m|im|i am)\s+(\d{1,3})\b",
        r"\b(\d{1,3})[- ]?(?:years?[- ]old|y\.?o\.?|y/o|yo)\b",
        r"\b(\d{1,3})\s*(?:yrs?|yr)\b",
        r"\bage\s*(?:is|:)?\s*(\d{1,3})\b",
        r"\bpatient\s+is\s+(?:a\s+)?(\d{1,3})(?:\s*|\.|\,|$|\s+years|\s+yo|\s+y\.o)",
        r"^(\d{1,3})$",  # standalone number in pure age entry
    ]
    for pat in age_patterns:
        match = re.search(pat, text_lower)
        if match:
            age_candidate = match.group(1)
            val = int(age_candidate)
            # Guard: ensure not a temperature or other vital unit
            surrounding = text_lower[max(0, match.start()-15):min(len(text_lower), match.end()+15)]
            if not any(k in surrounding for k in ["°", "temp", "deg", "bpm", "mmhg", "%", "mg", "hours", "days", "weeks"]):
                if 0 <= val <= 130:
                    # If it's a standalone number between 95 and 108 without age units,
                    # ensure it's not ambiguous temperature if fever is in current state
                    if pat == r"^(\d{1,3})$" and 95 <= val <= 108:
                        if current_state and "fever" in current_state.get("symptoms", []):
                            # In fever context, standalone 102 is temperature, not age
                            result["measurements"]["temperature"] = f"{val}°F"
                            break
                    result["age"] = str(val)
                    break

    # -------------------------------------------------------------
    # 3. GENDER EXTRACTION
    # -------------------------------------------------------------
    if re.search(r'\b(?:female|woman|girl)\b', text_lower):
        result["gender"] = "female"
    elif re.search(r'\b(?:male|man|boy)\b', text_lower):
        result["gender"] = "male"
    elif re.search(r'\bnon[- ]binary\b', text_lower):
        result["gender"] = "non-binary"
    elif re.search(r'\b(?:he|his|him)\b', text_lower) and not re.search(r'\b(?:she|her)\b', text_lower):
        result["gender"] = "male"
    elif re.search(r'\b(?:she|her)\b', text_lower) and not re.search(r'\b(?:he|his|him)\b', text_lower):
        result["gender"] = "female"

    # -------------------------------------------------------------
    # 4. DURATION & SEVERITY
    # -------------------------------------------------------------
    dur = extract_duration_phrase(text_lower)
    if dur:
        result["duration"] = dur

    sev_match = re.search(r'\b(mild|moderate|severe|intense|sharp|dull|high|pretty\s+bad|bad)\b', text_lower)
    if sev_match:
        result["severity"] = sev_match.group(1).lower()

    # -------------------------------------------------------------
    # 5. NEGATIONS & DENIED SYMPTOMS
    # -------------------------------------------------------------
    negation_clauses = re.findall(
        r'\b(?:no|denies|without|negative for)\s+([a-zA-Z\s,]+?)(?:[.;]|\band\b\s+(?:currently|takes|taking|has|reports|with)|$)',
        text_lower
    )
    denied_found = set()
    for clause in negation_clauses:
        clause = clause.strip()
        for sym_name, sym_pat in KNOWN_SYMPTOM_PATTERNS.items():
            if re.search(sym_pat, clause):
                denied_found.add(sym_name)

    result["denied_symptoms"] = list(denied_found)

    # -------------------------------------------------------------
    # 6. POSITIVE SYMPTOMS (exclude any denied symptom)
    # -------------------------------------------------------------
    positive_found = set()
    for sym_name, sym_pat in KNOWN_SYMPTOM_PATTERNS.items():
        if sym_name not in denied_found and re.search(sym_pat, text_lower):
            pos_matches = list(re.finditer(sym_pat, text_lower))
            is_negated = False
            for m in pos_matches:
                prefix = text_lower[max(0, m.start()-25):m.start()]
                if re.search(r'\b(?:no|denies|without|negative for|denied)\s+(?:any\s+)?$', prefix):
                    is_negated = True
                    break
            if not is_negated:
                positive_found.add(sym_name)

    result["symptoms"] = list(positive_found)

    # -------------------------------------------------------------
    # 7. MEDICAL HISTORY & CONDITIONS (Global and Entity-Specific Negation)
    # -------------------------------------------------------------
    # 7a. Check entity-specific negation first: e.g. "no history of hypertension"
    denied_conditions = set()
    for cond in COMMON_CONDITIONS:
        neg_cond_pat = rf'\b(?:no\s+(?:history\s+of\s+|hx\s+of\s+)?|denies\s+(?:history\s+of\s+)?|without\s+){re.escape(cond)}\b'
        if re.search(neg_cond_pat, text_lower):
            denied_conditions.add("diabetes" if cond == "diabetic" else cond)
    result["denied_history"] = list(denied_conditions)

    # 7b. Check general negative confirmation (only if not a targeted negation like "no history of hypertension")
    if re.search(r'\b(?:no\s+(?:prior\s+|past\s+|known\s+)?(?:medical\s+)?history(?:\s+(?:reported|at\s+all))?|no\s+(?:known\s+)?(?:medical\s+)?conditions?|no\s+previous\s+diagnos|denies\s+(?:all\s+)?medical\s+history|no\s+past\s+illness)\b', text_lower) and not denied_conditions and not re.search(r'\bno\s+history\s+of\b', text_lower):
        result["medical_history_status"] = "NONE_REPORTED"
    elif re.search(r'\b(?:i\s+don\'?t\s+know\s+my\s+(?:medical\s+)?history|not\s+sure\s+about\s+my\s+(?:medical\s+)?history)\b', text_lower):
        result["medical_history_status"] = "UNKNOWN"
        result["other_information"].append("medical history unknown to patient")
    else:
        history_found = set()
        hist_headers = re.findall(r'(?:history of|hx of|pmh of|diagnosed with|known)\s+([a-zA-Z\s,]+?)(?:[;.]|\band\b\s+(?:currently|takes|taking|on)|$)', text_lower)
        for segment in hist_headers:
            for cond in COMMON_CONDITIONS:
                if cond in segment and cond not in denied_conditions:
                    clean_cond = "diabetes" if cond == "diabetic" else cond
                    history_found.add(clean_cond)

        # Standalone condition or mentioned in context
        for cond in COMMON_CONDITIONS:
            if cond not in denied_conditions and re.search(rf'\b{re.escape(cond)}\b', text_lower):
                # Verify not preceded by negation
                matches = list(re.finditer(rf'\b{re.escape(cond)}\b', text_lower))
                is_neg = False
                for m in matches:
                    prefix = text_lower[max(0, m.start()-25):m.start()]
                    if re.search(r'\b(?:no|denies|without|negative for)\s+(?:history\s+of\s+)?$', prefix):
                        is_neg = True
                        break
                if not is_neg:
                    clean_cond = "diabetes" if cond == "diabetic" else cond
                    history_found.add(clean_cond)

        if history_found:
            result["medical_history"] = list(history_found)
            result["medical_history_status"] = "PRESENT"


    # -------------------------------------------------------------
    # 8. MEDICATIONS (Global and Entity-Specific Negation)
    # -------------------------------------------------------------
    _no_meds_pattern = re.compile(
        r"\b(?:"
        r"no\s+medications?|"
        r"no\s+meds?|"
        r"not\s+taking\s+any\s+(?:medications?|meds?)|"
        r"don(?:'|')?t\s+take\s+any\s+(?:medications?|meds?)|"
        r"not\s+on\s+any\s+(?:medications?|meds?)|"
        r"i\s+don(?:'|')?t\s+take\s+any\s+(?:medications?|meds?)|"
        r"no\s+current\s+(?:medications?|meds?)|"
        r"denies\s+(?:medications?|meds?)|"
        r"none\s+at\s+this\s+time"
        r")\b",
        re.IGNORECASE
    )
    if _no_meds_pattern.search(text_lower):
        result["medications_status"] = "NONE_REPORTED"
    else:
        denied_meds = set()
        for med in COMMON_MEDICATIONS:
            neg_med_pat = rf'\b(?:no\s+(?:current\s+use\s+of\s+)?|not\s+taking\s+|denies\s+|without\s+){re.escape(med)}\b'
            if re.search(neg_med_pat, text_lower):
                denied_meds.add(med)
        result["denied_medications"] = list(denied_meds)

        meds_found = set()
        for med in COMMON_MEDICATIONS:
            if med not in denied_meds and re.search(rf'\b{re.escape(med)}\b', text_lower):
                meds_found.add(med)
        if meds_found:
            result["medications"] = list(meds_found)
            result["medications_status"] = "PRESENT"

    # -------------------------------------------------------------
    # 9. EVALUATE WHETHER DETERMINISTIC EXTRACTION EXPLAINS INPUT
    # -------------------------------------------------------------
    clean_rem = text_lower
    for sym in result["symptoms"] + result["denied_symptoms"] + result["medical_history"] + result["denied_history"] + result["medications"] + result["denied_medications"]:
        clean_rem = re.sub(rf'\b{re.escape(sym)}\b', '', clean_rem)

    noise_words = [
        "patient", "has", "had", "is", "a", "an", "the", "with", "and", "or", "also", "reports",
        "feeling", "very", "for", "past", "since", "yesterday", "along", "of", "hx", "history",
        "known", "currently", "taking", "on", "c/o", "complains", "years", "old", "yo", "yr", "male", "female",
        "non-binary", "temp", "temperature", "deg", "f", "c", "no", "denies", "without", "mild", "moderate",
        "severe", "high", "hrs", "hours", "hour", "hr", "days", "day", "weeks", "week", "wks", "wk",
        "months", "month", "mos", "mo", "mins", "minutes", "min", "pain", "due", "to", "feel", "feels",
        "don", "take", "any", "not", "i", "at", "this", "time", "like", "about", "around", "approx",
        "approximately", "nearly", "almost", "couple", "few", "several", "ago", "started", "been", "over",
        "sure", "know", "my", "she", "he", "her", "his", "him", "yes", "yeah", "correct", "true", "none"
    ]
    tokens = [w for w in re.findall(r'[a-zA-Z]+', clean_rem) if len(w) > 2 and w not in noise_words]
    has_unrecognized_text = len(tokens) > 3

    is_fully_understood = not has_unrecognized_text and (
        bool(result["age"]) or
        bool(result["gender"]) or
        bool(result["symptoms"]) or
        bool(result["denied_symptoms"]) or
        bool(result["measurements"]) or
        bool(result["duration"]) or
        bool(result["severity"]) or
        bool(result["medical_history"]) or
        bool(result["denied_history"]) or
        bool(result["medications"]) or
        bool(result["denied_medications"]) or
        result["medical_history_status"] != "UNKNOWN" or
        result["medications_status"] != "UNKNOWN"
    )

    return result, is_fully_understood

