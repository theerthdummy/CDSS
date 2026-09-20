"""
abbrevs.py - Medical Abbreviation Dictionary, Dynamic MeSH Lookup, and Query Formatter.
Provides centralized acronym resolution and query expansion for Agent 3.
"""

import os
import re
from typing import Dict, Optional
from Bio import Entrez

NCBI_EMAIL = os.getenv("NCBI_EMAIL", "adarshreddy261@gmail.com")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

Entrez.email = NCBI_EMAIL
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY

# Centralized Medical Abbreviation and Comorbidity Map
MEDICAL_ABBREV_MAP: Dict[str, str] = {
    # Acute Symptoms & Signs
    "sob": "shortness of breath",
    "cp": "chest pain",
    "fever": "febrile illness",
    "cough": "cough",
    "chills": "chills",
    "dyspnea": "shortness of breath",
    "diaphoresis": "sweating",
    
    # Chronic Comorbidities & Pre-existing Conditions
    "dm": "diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "diabeties": "diabetes mellitus",
    "diabetes": "diabetes mellitus",
    "htn": "hypertension",
    "hf": "heart failure",
    "chf": "congestive heart failure",
    "weakhart": "heart failure",
    "weak heart": "heart failure",
    "cad": "coronary artery disease",
    "mi": "myocardial infarction",
    "ra": "rheumatoid arthritis",
    "romatide": "rheumatoid arthritis",
    "rheumatoid": "rheumatoid arthritis",
    "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease",
    "aki": "acute kidney injury",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "afib": "atrial fibrillation",
    "gerd": "gastroesophageal reflux disease"
}


def lookup_mesh_synonym(term: str) -> Optional[str]:
    """Dynamically queries the NCBI MeSH database for unmapped medical acronyms/terms."""
    if not term or len(term) < 2:
        return None
    try:
        handle = Entrez.esearch(db="mesh", term=term, retmax=1)
        record = Entrez.read(handle)
        handle.close()
        id_list = record.get("IdList", [])
        if not id_list:
            return None

        summary_handle = Entrez.esummary(db="mesh", id=id_list[0])
        summary_record = Entrez.read(summary_handle)
        summary_handle.close()

        if summary_record and len(summary_record) > 0:
            return summary_record[0].get("DS_MeshTerms", [None])[0]
        return None
    except Exception:
        return None


def sanitize_and_expand_entity(entity: str) -> str:
    """
    Cleans raw entity, expands abbreviations via MEDICAL_ABBREV_MAP,
    and returns a Boolean OR group: (abbr OR "expansion").
    """
    if not entity:
        return ""

    cleaned = re.sub(r'[^a-zA-Z0-9\s\-]', '', entity).strip().lower()
    if not cleaned:
        return ""

    # 1. Match in local centralized dictionary
    if cleaned in MEDICAL_ABBREV_MAP:
        expansion = MEDICAL_ABBREV_MAP[cleaned]
        return f'({cleaned} OR "{expansion}")'

    # 2. Dynamic MeSH lookup for unmapped abbreviations (2-5 alphabetical characters)
    if len(cleaned) <= 5 and cleaned.isalpha():
        mesh_term = lookup_mesh_synonym(cleaned)
        if mesh_term and mesh_term.lower() != cleaned:
            return f'({cleaned} OR "{mesh_term.lower()}")'

    return cleaned