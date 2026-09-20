import re

ABBREVIATIONS = {
    "SOB": "shortness of breath",
    "HTN": "hypertension",
    "DM": "diabetes mellitus",
    "COPD": "chronic obstructive pulmonary disease",
    "MI": "myocardial infarction",
    "BP": "blood pressure",
    "HR": "heart rate",
    "CXR": "chest X-ray",
}


def expand_abbreviations(text: str) -> str:
    if text is None:
        return ""

    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(key) for key in sorted(ABBREVIATIONS, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )

    def replace(match):
        key = match.group(0)
        return ABBREVIATIONS.get(key.upper(), key)

    return pattern.sub(replace, text)
