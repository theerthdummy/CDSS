"""
condition_inferencer.py - Uses Groq primary extraction with local Ollama Mistral fallback.
- Formulates MeSH Boolean query for PubMed (Acute symptoms & Condition hint only).
- Formulates Tavily guideline prompt (incorporating chronic conditions as risk measures).
"""

import os
import json
import re
import logging
from typing import Dict, Any, List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("agent3.condition_inferencer")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")

# Groq round-robin configuration
GROQ_API_KEYS: List[str] = [
    k.strip() for k in [
        os.getenv("GROQ_API_KEY_1", ""),
        os.getenv("GROQ_API_KEY_2", ""),
        os.getenv("GROQ_API_KEY_3", ""),
        os.getenv("GROQ_API_KEY_4", ""),
    ] if k.strip()
]
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
GROQ_API_BASE = "https://api.groq.com/openai/v1"

NON_SYMPTOM_DESCRIPTORS = {
    "long", "standing", "long-standing", "acute", "chronic", "severe", 
    "mild", "moderate", "recent", "worsening", "progressive", "constant",
    "intermittent", "generalized", "isolated", "persistent", "recurrent",
    "sudden", "prior", "previous", "stable", "unstable", "primary", "secondary",
    "and", "or", "also", "with", "history", "patient", "presentation"
}

CHRONIC_RISK_MAP = {
    "diabetes": "increased infection/complication risk, glycemic volatility, risk of diabetic ketoacidosis",
    "ckd": "affects medication clearance/dosing, nephrotoxic drug contraindications, fluid balance risk",
    "chronic kidney disease": "affects renal clearance and dosing, nephrotoxic drug precautions",
    "heart failure": "fluid overload risk, contraindication for NSAIDs, caution with IV fluid boluses",
    "rheumatoid arthritis": "immunosuppression risk from DMARDs/biologics, atypical infection risk"
}


def filter_valid_symptoms(symptoms: List[str]) -> List[str]:
    """Ensures acute symptoms are valid clinical findings rather than descriptors."""
    validated = []
    for sym in symptoms:
        clean = sym.strip().lower()
        clean = re.sub(r'[^a-zA-Z0-9\s-]', '', clean)
        if clean and clean not in NON_SYMPTOM_DESCRIPTORS and len(clean) > 2:
            validated.append(clean)
    return validated


def quote_if_compound(term: str) -> str:
    """Encloses multi-word medical expressions in quotes."""
    clean = term.strip().strip('"')
    return f'"{clean}"' if " " in clean else clean


def build_chronic_risk_profiles(chronic_conditions: List[str]) -> List[str]:
    """Formats chronic conditions into explicit clinical risk measures."""
    risk_profiles = []
    for cond in chronic_conditions:
        norm = cond.lower().strip()
        matched_risk = None
        for key, desc in CHRONIC_RISK_MAP.items():
            if key in norm:
                matched_risk = f"{cond}: → {desc}"
                break
        if not matched_risk:
            matched_risk = f"{cond}: → comorbidity-related treatment adjustments and complications"
        risk_profiles.append(matched_risk)
    return risk_profiles


def _groq_profile_extraction(
    prompt: str,
) -> Optional[Dict[str, Any]]:
    """Try Groq round-robin keys to extract clinical profile JSON.
    
    Returns parsed dict on success, or None if all keys fail.
    """
    if not GROQ_API_KEYS:
        return None

    endpoint = f"{GROQ_API_BASE}/chat/completions"
    system_msg = (
        "You are an expert clinical NLP parser and medical information retrieval specialist. "
        "Return STRICT JSON with these exact keys: acute_symptoms (array of strings), "
        "chronic_conditions (array of strings), inferred_condition (string), "
        "pubmed_mesh_query (string), chronic_risk_measures (array of strings), "
        "tavily_guideline_query (string). No other keys or markdown."
    )
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    import time
    for key_idx, api_key in enumerate(GROQ_API_KEYS, start=1):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=15)

            if resp.status_code == 401:
                logger.warning(f"Groq key {key_idx}/{len(GROQ_API_KEYS)} auth failed (401). Trying next...")
                continue
            if resp.status_code == 429:
                logger.warning(f"Groq key {key_idx}/{len(GROQ_API_KEYS)} rate limited (429). Trying next...")
                time.sleep(1.0)
                continue
            if resp.status_code == 404:
                # Try alternative model name
                alt_model = "openai/gpt-oss-20b" if GROQ_MODEL == "openai/gpt-oss-120b" else "openai/gpt-oss-120b"
                logger.info(f"Groq model 404. Retrying with '{alt_model}'...")
                payload["model"] = alt_model
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Groq key {key_idx} returned HTTP {resp.status_code}. Trying next...")
                continue

            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"].strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            result = json.loads(raw_text.strip())

            # Validate required keys exist
            required_keys = {"acute_symptoms", "chronic_conditions", "inferred_condition",
                           "pubmed_mesh_query", "chronic_risk_measures", "tavily_guideline_query"}
            if required_keys.issubset(result.keys()):
                logger.info(f"Groq round-robin succeeded with key {key_idx}/{len(GROQ_API_KEYS)}.")
                return result
            else:
                logger.warning(f"Groq key {key_idx} returned incomplete JSON. Trying next...")
                continue

        except requests.exceptions.RequestException as e:
            logger.warning(f"Groq key {key_idx} connection error: {e}. Trying next...")
            continue
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Groq key {key_idx} parse error: {e}. Trying next...")
            continue

    logger.warning("All Groq API keys exhausted. Falling back to local Ollama Mistral.")
    return None

def extract_clinical_profile_and_mesh_queries(
    raw_text: str,
    acute_symptoms: Optional[List[str]] = None,
    chronic_conditions: Optional[List[str]] = None,
    condition_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parses clinical scenario and builds search payloads:
    1. PubMed: Queries ONLY acute symptoms + condition hint.
    2. Tavily: Queries acute symptoms + condition hint + chronic conditions as risk measures.
    """
    if not raw_text or not raw_text.strip():
        return _empty_profile()

    symptoms = acute_symptoms or []
    chronic = chronic_conditions or []
    hint = condition_hint or ""

    # === GROQ ROUND-ROBIN PRIMARY ATTEMPT ===
    if GROQ_API_KEYS:
        prompt = (
            f"Patient Presentation: \"{raw_text}\"\n"
            f"Known Symptoms: {symptoms}\n"
            f"Known Chronic Conditions: {chronic}\n"
            f"Condition Hint: {hint}\n\n"
            "Clinical NLP Formulation Tasks:\n"
            "1. Extract 'acute_symptoms'. Filter out pure adjectives/adverbs.\n"
            "2. Extract 'chronic_conditions'.\n"
            "3. Infer 'inferred_condition'.\n"
            "4. Formulate 'pubmed_mesh_query' using ONLY acute symptoms and inferred condition in [Title/Abstract] or [MeSH Terms] syntax. Double-quote compound phrases.\n"
            "5. Convert each chronic condition into explicit clinical risk measures (e.g., 'Diabetes: → increased infection risk', 'CKD: → affects medication selection/dosing').\n"
            "6. Formulate 'tavily_guideline_query' incorporating acute symptoms, inferred condition, and chronic conditions as risk modifiers for treatment selection & guidelines."
        )
        groq_result = _groq_profile_extraction(prompt)
        if groq_result is not None:
            groq_result["acute_symptoms"] = filter_valid_symptoms(groq_result.get("acute_symptoms", []))
            return groq_result

    # === OLLAMA FALLBACK ===
    try:
        prompt = (
            f"Patient Presentation: \"{raw_text}\"\n"
            f"Known Symptoms: {symptoms}\n"
            f"Known Chronic Conditions: {chronic}\n"
            f"Condition Hint: {hint}\n\n"
            "Clinical NLP Formulation Tasks:\n"
            "1. Extract 'acute_symptoms'. Filter out pure adjectives/adverbs.\n"
            "2. Extract 'chronic_conditions'.\n"
            "3. Infer 'inferred_condition'.\n"
            "4. Formulate 'pubmed_mesh_query' using ONLY acute symptoms and inferred condition in [Title/Abstract] or [MeSH Terms] syntax. Double-quote compound phrases.\n"
            "5. Convert each chronic condition into explicit clinical risk measures (e.g., 'Diabetes: → increased infection risk', 'CKD: → affects medication selection/dosing').\n"
            "6. Formulate 'tavily_guideline_query' incorporating acute symptoms, inferred condition, and chronic conditions as risk modifiers for treatment selection & guidelines.\n\n"
            "Return STRICT JSON with these exact keys: acute_symptoms, chronic_conditions, inferred_condition, pubmed_mesh_query, chronic_risk_measures, tavily_guideline_query."
        )

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            raw_text = data.get("response", "").strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            result = json.loads(raw_text.strip())
            required_keys = {"acute_symptoms", "chronic_conditions", "inferred_condition", "pubmed_mesh_query", "chronic_risk_measures", "tavily_guideline_query"}
            if required_keys.issubset(result.keys()):
                result["acute_symptoms"] = filter_valid_symptoms(result.get("acute_symptoms", []))
                return result
    except Exception as e:
        logger.warning(f"Ollama fallback failed: {e}")

    logger.info("Falling back to heuristic profile extraction.")

    return _heuristic_profile_fallback(raw_text, symptoms, chronic, hint)


def _heuristic_profile_fallback(
    raw_text: str,
    symptoms: List[str],
    chronic: List[str],
    hint: str
) -> Dict[str, Any]:
    text_lower = raw_text.lower()

    symptom_map = {
        "fever": "fever",
        "cough": "cough",
        "productive cough": "productive cough",
        "shortness of breath": "shortness of breath",
        "sob": "shortness of breath",
        "chest pain": "chest pain",
        "chills": "chills",
        "wheezing": "wheezing",
        "fatigue": "fatigue"
    }
    chronic_map = {
        "diabetes": "diabetes mellitus",
        "type 2 diabetes": "type 2 diabetes mellitus",
        "ckd": "chronic kidney disease",
        "chronic kidney disease": "chronic kidney disease",
        "heart failure": "heart failure",
        "weak heart": "heart failure",
        "rheumatoid arthritis": "rheumatoid arthritis",
        "copd": "chronic obstructive pulmonary disease"
    }

    extracted_symptoms = list(symptoms)
    extracted_chronic = list(chronic)

    for k, v in symptom_map.items():
        if k in text_lower and v not in extracted_symptoms:
            extracted_symptoms.append(v)

    for k, v in chronic_map.items():
        if k in text_lower and v not in extracted_chronic:
            extracted_chronic.append(v)

    extracted_symptoms = filter_valid_symptoms(extracted_symptoms)
    if not extracted_symptoms:
        extracted_symptoms = ["fever"]

    inferred_hint = hint or ("Pneumonia" if "cough" in extracted_symptoms else "Acute Condition")
    risk_profiles = build_chronic_risk_profiles(extracted_chronic)

    pm_sym = " OR ".join([f'{quote_if_compound(s)}[Title/Abstract]' for s in extracted_symptoms])
    pubmed_q = f"({pm_sym}) AND {quote_if_compound(inferred_hint)}[Title/Abstract]"

    risk_str = " | ".join(risk_profiles) if risk_profiles else "None"
    tavily_q = (
        f"Clinical practice guidelines and treatment safety for {inferred_hint} presenting with {', '.join(extracted_symptoms)}. "
        f"Comorbidity Risk Profiles: {risk_str}. Detail drug dosing adjustments and contraindications."
    )

    return {
        "acute_symptoms": extracted_symptoms,
        "chronic_conditions": extracted_chronic,
        "inferred_condition": inferred_hint,
        "pubmed_mesh_query": pubmed_q,
        "chronic_risk_measures": risk_profiles,
        "tavily_guideline_query": tavily_q
    }


def _empty_profile() -> Dict[str, Any]:
    return {
        "acute_symptoms": [],
        "chronic_conditions": [],
        "inferred_condition": "",
        "pubmed_mesh_query": "",
        "chronic_risk_measures": [],
        "tavily_guideline_query": ""
    }