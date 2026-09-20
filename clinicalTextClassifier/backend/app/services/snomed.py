"""Snowstorm SNOMED CT terminology lookup service.

Provides a staged terminology lookup service separate from LLM reasoning:
- Stage 1: Normalized clinical term
- Stage 2: Original term (if different)
- Stage 3: Abbreviation-expanded term
- Stage 4: Contextual candidate scoring and disambiguation

Integrates:
1. Live Snowstorm terminology server (when running locally or via URL)
2. Embedded SNOMED CT Core reference mapping for standard clinical concepts
3. Safe fallback when terminology server is offline.
"""

import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv()

# In-memory concept cache to prevent repeated network requests
_CONCEPT_CACHE: Dict[str, Dict[str, Any]] = {}

# Standard SNOMED CT International Core Reference Concepts (for instant local validation)
SNOMED_CORE_REFERENCE: Dict[str, Dict[str, str]] = {
    # Symptoms & Findings
    "fever": {"code": "386661006", "display": "Fever (finding)"},
    "high fever": {"code": "386661006", "display": "Fever (finding)"},
    "headache": {"code": "25064002", "display": "Headache (finding)"},
    "severe headache": {"code": "25064002", "display": "Headache (finding)"},
    "chest pain": {"code": "29857009", "display": "Chest pain (finding)"},
    "mild chest pain": {"code": "29857009", "display": "Chest pain (finding)"},
    "severe chest pain": {"code": "29857009", "display": "Chest pain (finding)"},
    "dizziness": {"code": "404640003", "display": "Dizziness (finding)"},
    "nausea": {"code": "422587007", "display": "Nausea (finding)"},
    "vomiting": {"code": "422400008", "display": "Vomiting (finding)"},
    "cough": {"code": "49727002", "display": "Cough (finding)"},
    "shortness of breath": {"code": "267036007", "display": "Dyspnea (finding)"},
    "dyspnea": {"code": "267036007", "display": "Dyspnea (finding)"},
    "chills": {"code": "43724002", "display": "Chill (finding)"},
    "fatigue": {"code": "84229001", "display": "Fatigue (finding)"},
    "weakness": {"code": "13791008", "display": "Asthenia (finding)"},
    "body pain": {"code": "68962001", "display": "Muscle pain (finding)"},
    "body ache": {"code": "68962001", "display": "Muscle pain (finding)"},
    "palpitations": {"code": "80313002", "display": "Palpitations (finding)"},
    "sweating": {"code": "52559000", "display": "Hyperhidrosis (finding)"},
    "sore throat": {"code": "267102003", "display": "Sore throat symptom (finding)"},
    "runny nose": {"code": "64531003", "display": "Rhinorrhea (finding)"},
    "congestion": {"code": "68235000", "display": "Nasal congestion (finding)"},
    "numbness": {"code": "44077006", "display": "Numbness (finding)"},
    "tingling": {"code": "62507009", "display": "Pins and needles (finding)"},
    "back pain": {"code": "161891005", "display": "Backache (finding)"},
    "abdominal pain": {"code": "21522000", "display": "Abdominal pain (finding)"},
    "joint pain": {"code": "57676002", "display": "Joint pain (finding)"},
    "rash": {"code": "271807003", "display": "Skin rash (finding)"},
    "swelling": {"code": "65124004", "display": "Swelling (finding)"},
    
    # Diseases & Disorders
    "diabetes": {"code": "73211009", "display": "Diabetes mellitus (disorder)"},
    "diabetes mellitus": {"code": "73211009", "display": "Diabetes mellitus (disorder)"},
    "hypertension": {"code": "38341003", "display": "Hypertensive disorder, systemic arterial (disorder)"},
    "high blood pressure": {"code": "38341003", "display": "Hypertensive disorder, systemic arterial (disorder)"},
    "asthma": {"code": "195967001", "display": "Asthma (disorder)"},
    "copd": {"code": "13645005", "display": "Chronic obstructive lung disease (disorder)"},
    "chronic obstructive pulmonary disease": {"code": "13645005", "display": "Chronic obstructive lung disease (disorder)"},
    "myocardial infarction": {"code": "22298006", "display": "Myocardial infarction (disorder)"},
    "heart disease": {"code": "56265001", "display": "Heart disease (disorder)"},
    "stroke": {"code": "230690007", "display": "Cerebrovascular accident (disorder)"},
    "hyperlipidemia": {"code": "55822004", "display": "Hyperlipidemia (disorder)"},

    # Medical Acronyms & Clinical Abbreviations (mapped to official SNOMED CT Concepts)
    "htn": {"code": "38341003", "display": "Hypertensive disorder, systemic arterial (disorder)"},
    "dm": {"code": "73211009", "display": "Diabetes mellitus (disorder)"},
    "dm2": {"code": "44054006", "display": "Type 2 diabetes mellitus (disorder)"},
    "t2dm": {"code": "44054006", "display": "Type 2 diabetes mellitus (disorder)"},
    "dm1": {"code": "46635009", "display": "Type 1 diabetes mellitus (disorder)"},
    "t1dm": {"code": "46635009", "display": "Type 1 diabetes mellitus (disorder)"},
    "sob": {"code": "267036007", "display": "Dyspnea (finding)"},
    "mi": {"code": "22298006", "display": "Myocardial infarction (disorder)"},
    "cad": {"code": "53741008", "display": "Coronary arteriosclerosis (disorder)"},
    "chf": {"code": "88805009", "display": "Chronic congestive heart failure (disorder)"},
    "hf": {"code": "84114007", "display": "Heart failure (disorder)"},
    "ckd": {"code": "709044004", "display": "Chronic kidney disease (disorder)"},
    "esrd": {"code": "46177005", "display": "End-stage renal disease (disorder)"},
    "gerd": {"code": "235595009", "display": "Gastroesophageal reflux disease (disorder)"},
    "tia": {"code": "266257000", "display": "Transient ischemic attack (disorder)"},
    "cva": {"code": "230690007", "display": "Cerebrovascular accident (disorder)"},
    "afib": {"code": "49436004", "display": "Atrial fibrillation (disorder)"},
    "a-fib": {"code": "49436004", "display": "Atrial fibrillation (disorder)"},
    "dvt": {"code": "128053003", "display": "Deep venous thrombosis (disorder)"},
    "pe": {"code": "59282003", "display": "Pulmonary embolism (disorder)"},
    "uti": {"code": "68566005", "display": "Urinary tract infection (disorder)"},
    "uri": {"code": "54150009", "display": "Upper respiratory tract infection (disorder)"},
    "ra": {"code": "69896004", "display": "Rheumatoid arthritis (disorder)"},
    "oa": {"code": "396275006", "display": "Osteoarthritis (disorder)"},
    "osa": {"code": "78275009", "display": "Obstructive sleep apnea (disorder)"},
    "ptsd": {"code": "47505003", "display": "Posttraumatic stress disorder (disorder)"},
    "bph": {"code": "266569009", "display": "Benign prostatic hyperplasia (disorder)"},
    # Medications & Substances
    "metformin": {"code": "372567009", "display": "Metformin (substance)"},
    "amlodipine": {"code": "386864001", "display": "Amlodipine (substance)"},
    "aspirin": {"code": "387458008", "display": "Aspirin (substance)"},
    "lisinopril": {"code": "387207008", "display": "Lisinopril (substance)"},
    "atorvastatin": {"code": "387584000", "display": "Atorvastatin (substance)"},
    "insulin": {"code": "67866001", "display": "Insulin (substance)"},
    "albuterol": {"code": "372897005", "display": "Salbutamol (substance)"},
    "omeprazole": {"code": "387124004", "display": "Omeprazole (substance)"},
    "ibuprofen": {"code": "387207008", "display": "Ibuprofen (substance)"},
    "paracetamol": {"code": "387517004", "display": "Paracetamol (substance)"},
    "acetaminophen": {"code": "387517004", "display": "Paracetamol (substance)"},
}


def get_snowstorm_base_url() -> str:
    """Read the Snowstorm base URL from the environment."""
    return (os.getenv("SNOWSTORM_BASE_URL") or "").strip().rstrip("/")


def _normalize_term(term: str) -> str:
    """Normalize concept text for matching."""
    return re.sub(r"[^a-z0-9\s]", " ", (term or "").lower()).strip()


def _candidate_urls(term: str) -> List[str]:
    """Build candidate Snowstorm browser/search endpoints."""
    base = get_snowstorm_base_url()
    if not base:
        return []
    query = quote(term)
    return [
        f"{base}/browser/MAIN/concepts?term={query}&limit=5",
        f"{base}/browser/MAIN/descriptions?term={query}&limit=5",
    ]


def _extract_display(candidate: Dict[str, Any]) -> Optional[str]:
    """Extract a display label from a Snowstorm candidate result."""
    if not isinstance(candidate, dict):
        return None

    for key in ["pt", "display", "preferredTerm", "term", "fsn", "name"]:
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested_key in ["term", "fsn", "name", "display"]:
                nested_value = value.get(nested_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()

    nested = candidate.get("concept")
    if isinstance(nested, dict):
        for key in ["pt", "display", "preferredTerm", "term", "fsn", "name"]:
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                for nested_key in ["term", "fsn", "name", "display"]:
                    nested_value = value.get(nested_key)
                    if isinstance(nested_value, str) and nested_value.strip():
                        return nested_value.strip()
    return None


def _extract_code(candidate: Dict[str, Any]) -> Optional[str]:
    """Extract a SNOMED CT conceptId from a candidate result."""
    if not isinstance(candidate, dict):
        return None

    for key in ["conceptId", "concept_id", "id", "code", "conceptCode"]:
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    nested = candidate.get("concept")
    if isinstance(nested, dict):
        for key in ["conceptId", "concept_id", "id", "code", "conceptCode"]:
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _coerce_result(raw: Any) -> List[Dict[str, Any]]:
    """Coerce Snowstorm JSON payload into candidate concept dictionaries."""
    if isinstance(raw, list):
        return [v for v in raw if isinstance(v, dict)]
    if isinstance(raw, dict):
        for key in ["items", "matches", "results", "concepts", "data"]:
            value = raw.get(key)
            if isinstance(value, list):
                return [v for v in value if isinstance(v, dict)]
        if any(isinstance(v, dict) for v in raw.values()):
            return [v for v in raw.values() if isinstance(v, dict)]
    return []


def _score_candidate(search_term: str, candidate: Dict[str, Any]) -> float:
    """Calculate terminology match confidence score."""
    if not isinstance(candidate, dict):
        return 0.0

    display = (_extract_display(candidate) or "").lower()
    term = _normalize_term(search_term)
    if not term or not display:
        return 0.0

    # Clean display of semantic tags like '(finding)' or '(disorder)'
    clean_display = re.sub(r"\s*\([^)]*\)$", "", display).strip()

    if clean_display == term:
        return 1.0
    if term in clean_display:
        return 0.90
    if clean_display in term:
        return 0.85

    return round(SequenceMatcher(None, term, clean_display).ratio() * 0.75, 2)


def staged_snomed_lookup(
    normalized_term: str,
    original_term: Optional[str] = None,
    expanded_term: Optional[str] = None
) -> Dict[str, Any]:
    """Staged lookup following Stages 1 -> 2 -> 3 -> 4 with fallback.
    
    Returns:
    {
        "concept": "fever",
        "code": "386661006",
        "display": "Fever (finding)",
        "confidence": 1.0,
        "source": "SNOMED-CT",
        "mapping_status": "MAPPED"
    }
    """
    term = (normalized_term or "").strip()
    if not term:
        return {
            "concept": "",
            "code": None,
            "display": None,
            "confidence": 0.0,
            "source": "unmapped",
            "mapping_status": "UNAVAILABLE"
        }

    # Check memory cache
    cache_key = term.lower()
    if cache_key in _CONCEPT_CACHE:
        return _CONCEPT_CACHE[cache_key]

    # Check SNOMED CT Core Reference tier
    norm_key = _normalize_term(term)
    if norm_key in SNOMED_CORE_REFERENCE:
        core_info = SNOMED_CORE_REFERENCE[norm_key]
        result = {
            "concept": term,
            "code": core_info["code"],
            "display": core_info["display"],
            "confidence": 1.0,
            "source": "SNOMED-CT",
            "mapping_status": "MAPPED"
        }
        _CONCEPT_CACHE[cache_key] = result
        return result

    # Check if a Snowstorm server endpoint is available
    base_url = get_snowstorm_base_url()
    if base_url and not base_url.startswith("https://browser.ihtsdotools.org"):
        search_terms = [term]
        if original_term and original_term.strip().lower() != term.lower():
            search_terms.append(original_term.strip())
        if expanded_term and expanded_term.strip().lower() not in [t.lower() for t in search_terms]:
            search_terms.append(expanded_term.strip())

        best_candidate = None
        best_score = -1.0

        for query_term in search_terms:
            for url in _candidate_urls(query_term):
                try:
                    response = requests.get(url, timeout=1.0)
                    if response.status_code != 200:
                        continue

                    candidates = _coerce_result(response.json())
                    for cand in candidates:
                        score = _score_candidate(query_term, cand)
                        if score > best_score:
                            best_score = score
                            best_candidate = cand

                    if best_score >= 0.85:
                        break
                except Exception:
                    continue
            if best_score >= 0.85:
                break

        if best_candidate is not None and best_score >= 0.5:
            code = _extract_code(best_candidate)
            display = _extract_display(best_candidate)
            if code and display:
                result = {
                    "concept": term,
                    "code": code,
                    "display": display,
                    "confidence": best_score,
                    "source": "Snowstorm",
                    "mapping_status": "MAPPED"
                }
                _CONCEPT_CACHE[cache_key] = result
                return result

    result = {
        "concept": term,
        "code": None,
        "display": None,
        "confidence": 0.0,
        "source": "unmapped",
        "mapping_status": "UNAVAILABLE"
    }
    _CONCEPT_CACHE[cache_key] = result
    return result


def map_concept(term: str) -> Dict[str, Any]:
    """Public helper for single concept mapping."""
    return staged_snomed_lookup(term)


def map_clinical_concepts(
    symptoms: List[str],
    medical_history: List[str],
    medications: List[str]
) -> List[Dict[str, Any]]:
    """Map only extracted clinical concepts (symptoms, diagnoses, medications) to SNOMED CT.
    
    Excludes non-clinical demographics like age, gender, duration, and raw numbers.
    """
    mappings = []
    seen = set()

    for concept in symptoms + medical_history + medications:
        clean = concept.strip()
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        mappings.append(staged_snomed_lookup(clean))

    return mappings
