"""Lightweight Clinical Text Normalizer.

Applies deterministic, high-confidence normalization BEFORE processing:
- Whitespace & punctuation cleanup
- Unit standardization (e.g., deg F, degrees Fahrenheit -> °F, 140 over 90 -> 140/90)
- Number word conversion in conversational phrases (e.g., forty-five -> 45, three days -> 3 days)
- Common medical abbreviation / shorthand expansion (e.g., HTN -> hypertension, SOB -> shortness of breath, 48M -> 48 year old male, x3d -> for 3 days)
- Common typographical / spelling corrections (e.g., fevr -> fever, sever -> severe, hedache -> headache, dizzines -> dizziness)
- Preserves original input text without ambiguous substring replacement.
"""

import re
from typing import Dict

# High-confidence medical spelling corrections (whole word match only)
SPELLING_CORRECTIONS: Dict[str, str] = {
    "fevr": "fever",
    "feveer": "fever",
    "feverr": "fever",
    "sever": "severe",
    "dizzines": "dizziness",
    "diziness": "dizziness",
    "dizines": "dizziness",
    "dizzyness": "dizziness",
    "dizy": "dizzy",
    "diabtes": "diabetes",
    "diabeetes": "diabetes",
    "diabietes": "diabetes",
    "nausia": "nausea",
    "nauseea": "nausea",
    "nauseous": "nausea",
    "yesterdy": "yesterday",
    "yesturday": "yesterday",
    "headach": "headache",
    "hedache": "headache",
    "vomitin": "vomiting",
    "vomtiing": "vomiting",
    "shortnes": "shortness",
    "palpatations": "palpitations",
    "palpatation": "palpitation",
    "cought": "cough",
    "coff": "cough",
    "faigue": "fatigue",
    "weekness": "weakness",
    "breathin": "breathing",
}


# High-confidence medical abbreviations and shorthand
MEDICAL_ABBREVIATIONS: Dict[str, str] = {
    "HTN": "hypertension",
    "DM": "diabetes",
    "DM2": "type 2 diabetes",
    "T2DM": "type 2 diabetes",
    "DM1": "type 1 diabetes",
    "T1DM": "type 1 diabetes",
    "SOB": "shortness of breath",
    "MI": "myocardial infarction",
    "COPD": "chronic obstructive pulmonary disease",
    "CXR": "chest X-ray",
    "CAD": "coronary artery disease",
    "CKD": "chronic kidney disease",
    "hx of": "history of",
    "hx": "history",
    "c/o": "complains of",
    "pt": "patient",
    "w/": "with",
    "w/o": "without",
    "BP": "blood pressure",
    "HR": "heart rate",
    "pmh": "past medical history",
}

# English number words (0-100)
_NUMBER_WORDS_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19
}

_NUMBER_WORDS_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}


def normalize_number_words(text: str) -> str:
    """Convert compound and single number words (e.g. 'forty-five', 'thirty two', 'three') to digits."""
    # Compound numbers: "forty-five", "forty five", "thirty-two", etc.
    for tens_word, tens_val in _NUMBER_WORDS_TENS.items():
        for ones_word, ones_val in _NUMBER_WORDS_ONES.items():
            if 0 < ones_val < 10:
                compound_val = str(tens_val + ones_val)
                pat = rf'\b{tens_word}[- ]{ones_word}\b'
                text = re.sub(pat, compound_val, text, flags=re.IGNORECASE)

    # Pure tens: "twenty", "thirty", "forty", etc.
    for tens_word, tens_val in _NUMBER_WORDS_TENS.items():
        pat = rf'\b{tens_word}\b'
        text = re.sub(pat, str(tens_val), text, flags=re.IGNORECASE)

    # Standalone ones/teens in conversational phrases
    for ones_word, ones_val in _NUMBER_WORDS_ONES.items():
        if ones_val == 0:
            continue
        if ones_val == 1:
            text = re.sub(rf'\b(for|about|approx|past|since|started)\s+one\s+(day|week|month|year|hr|hour)\b', r'\1 1 \2', text, flags=re.IGNORECASE)
            text = re.sub(rf'\bone\s+hundred\b', '100', text, flags=re.IGNORECASE)
        else:
            text = re.sub(rf'\b{ones_word}\b', str(ones_val), text, flags=re.IGNORECASE)

    return text


def normalize_units(text: str) -> str:
    """Normalize temperature, blood pressure, and vital sign unit expressions."""
    # Blood pressure "140 over 90" -> "140/90"
    text = re.sub(
        r'\b(\d{2,3})\s+over\s+(\d{2,3})\b',
        r'\1/\2',
        text,
        flags=re.IGNORECASE
    )

    # Degrees Fahrenheit variations: '102 deg F', '102 degrees fahrenheit', '102 fahrenheit', '102 deg. F', '102f', '102 F'
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*(?:deg(?:rees?)?\.?\s*)?(?:f(?:ahrenheit)?)\b',
        r'\1°F',
        text,
        flags=re.IGNORECASE
    )

    # Degrees Celsius variations: '38.5 deg C', '38.5 degrees celsius', '38.5 celsius', '38.5c', '38.5 C'
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*(?:deg(?:rees?)?\.?\s*)?(?:c(?:elsius)?)\b',
        r'\1°C',
        text,
        flags=re.IGNORECASE
    )

    # Tmax shorthand: "Tmax 103F", "Tmax 103" -> "temp 103°F"
    text = re.sub(
        r'\btmax\s*(\d+(?:\.\d+)?)\s*°?\s*([fc])?\b',
        lambda m: f"temp {m.group(1)}°{m.group(2).upper() if m.group(2) else 'F'}",
        text,
        flags=re.IGNORECASE
    )

    return text


def normalize_abbreviations_and_spelling(text: str) -> str:
    """Standardize common abbreviations, clinical shorthand, and spelling errors safely."""
    # Demographic shorthand like '48M', '55F', '32m', '22f' -> '48 year old male', '55 year old female'
    text = re.sub(r'\b(\d{1,3})\s*([Mm])\b', r'\1 year old male', text)
    text = re.sub(r'\b(\d{1,3})\s*([Ff])\b', r'\1 year old female', text)

    # Duration shorthand like 'x3d', 'x 3d', 'x3 days' -> 'for 3 days'
    text = re.sub(r'\bx\s*(\d+)\s*d(?:ays?)?\b', r'for \1 days', text, flags=re.IGNORECASE)

    # Multi-word / slash abbreviations
    for abbr, full in [("hx of", "history of"), ("c/o", "complains of"), ("w/", "with "), ("w/o", "without "), ("pmh of", "history of")]:
        text = re.sub(re.escape(abbr), full, text, flags=re.IGNORECASE)

    # Clean age expressions like '55 y.o', '55 yo', '55 y/o', '55 yrs' -> '55 years old'
    text = re.sub(r'\b(\d+)\s*(?:y\.?o\.?|y/o)\b', r'\1 years old', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d+)\s*(?:yrs?|yr)\b', r'\1 years', text, flags=re.IGNORECASE)

    # Word-level dictionary replacements
    tokens = re.split(r'(\W+)', text)
    result_tokens = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in SPELLING_CORRECTIONS:
            result_tokens.append(SPELLING_CORRECTIONS[token_lower])
        elif token in MEDICAL_ABBREVIATIONS:
            result_tokens.append(MEDICAL_ABBREVIATIONS[token])
        elif token_lower in MEDICAL_ABBREVIATIONS:
            result_tokens.append(MEDICAL_ABBREVIATIONS[token_lower])
        else:
            result_tokens.append(token)

    return "".join(result_tokens)


def normalize_clinical_text(raw_text: str) -> str:
    """Run full deterministic normalization pipeline on raw user input."""
    if not raw_text or not raw_text.strip():
        return ""

    # 1. Clean whitespace and tabs
    text = re.sub(r'[\t\r\n]+', ' ', raw_text.strip())
    text = re.sub(r' +', ' ', text)

    # 2. Normalize number words
    text = normalize_number_words(text)

    # 3. Normalize units and vital signs
    text = normalize_units(text)

    # 4. Normalize abbreviations, shorthand & spelling
    text = normalize_abbreviations_and_spelling(text)

    # 5. Final spacing cleanup
    text = re.sub(r'\s+([,.;!?])', r'\1', text)
    text = re.sub(r' +', ' ', text).strip()

    return text

