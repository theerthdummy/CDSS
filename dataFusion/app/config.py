"""Application configuration module."""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Centralized application settings loaded from environment variables."""

    app_name: str = os.getenv("APP_NAME", "Agent 4 Clinical Data Fusion API")
    app_version: str = os.getenv("APP_VERSION", "1.0")
    app_description: str = os.getenv("APP_DESCRIPTION", "Clinical Data Fusion service for the multi-agent CDSS.")
    
    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8004"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    env: str = os.getenv("ENV", "production")
    
    # Correlation header configuration
    correlation_header: str = os.getenv("CORRELATION_HEADER", "X-Correlation-ID")

    # LLM Settings (Groq primary + Ollama fallback)
    llm_enabled: bool = os.getenv("LLM_ENABLED", "True").lower() in ("true", "1", "yes")
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    llm_api_key: str = (
        os.getenv("LLM_API_KEY") or
        os.getenv("GROQ_API_KEY") or
        os.getenv("OPENAI_API_KEY") or
        ""
    )
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "gpt-oss-120b")
    groq_api_key_pool: list = [
        k.strip() for k in [
            os.getenv("GROQ_API_KEY_1", ""),
            os.getenv("GROQ_API_KEY_2", ""),
            os.getenv("GROQ_API_KEY_3", ""),
            os.getenv("GROQ_API_KEY_4", ""),
        ] if k.strip()
    ]
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral:latest")
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "15.0"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1500"))

    # Evidence priorities from highest to lowest
    @property
    def evidence_priorities(self) -> List[str]:
        raw_priorities = os.getenv(
            "EVIDENCE_PRIORITIES",
            "Latest Medical Evidence,Knowledge Graph,Biomedical RAG"
        )
        return [p.strip() for p in raw_priorities.split(",") if p.strip()]

    # Configurable evidence source priority weights
    source_priority_weights: dict[str, float] = {
        "Latest Medical Evidence": 1.0,
        "Knowledge Graph": 0.8,
        "Biomedical RAG": 0.7,
    }

    # Configurable standard medical term normalization dictionary
    normalization_map: dict[str, str] = {
        "ami": "Acute Myocardial Infarction",
        "acute myocardial infarction": "Acute Myocardial Infarction",
        "sob": "Shortness of Breath",
        "shortness of breath": "Shortness of Breath",
        "dyspnea": "Shortness of Breath",
        "acs": "Acute Coronary Syndrome",
        "ecg": "Electrocardiogram (ECG)",
        "electrocardiogram": "Electrocardiogram (ECG)",
        "cad": "Coronary Artery Disease",
        "htn": "Hypertension",
        "angina pectoris": "Angina",
    }

    # Generic clinical keywords dictionary for comprehensive cross-specialty clinical entity extraction
    clinical_keywords_map: dict[str, List[str]] = {
        "symptoms": [
            "chest pain", "shortness of breath", "dyspnea", "diaphoresis",
            "sweating", "nausea", "fever", "pain", "fatigue", "dizziness",
            "hypotension", "tachycardia", "tachypnea", "altered mental status",
            "confusion", "cough", "hemoptysis", "edema", "syncope"
        ],
        "diagnostic_tests": [
            "ecg", "electrocardiogram", "troponin", "chest x-ray",
            "cardiac enzymes", "angiography", "blood test", "ct scan", "mri",
            "blood cultures", "serum lactate", "lactate", "procalcitonin",
            "cbc", "complete blood count", "urinalysis", "echocardiogram",
            "d-dimer", "bnp", "abg", "arterial blood gas", "lumbar puncture"
        ],
        "treatments": [
            "aspirin", "nitroglycerin", "oxygen", "thrombolysis", "heparin",
            "morphine", "beta-blocker", "statin", "antibiotic", "analgesic",
            "crystalloid", "fluid resuscitation", "norepinephrine", "vasopressor",
            "antimicrobial", "intubation", "mechanical ventilation", "dialysis"
        ],
        "conditions": [
            "myocardial infarction", "angina", "coronary syndrome", "heart failure",
            "hypertension", "pneumonia", "diabetes", "stroke", "sepsis",
            "septic shock", "systemic inflammatory response syndrome",
            "pulmonary embolism", "aortic dissection", "acute kidney injury",
            "respiratory failure", "ketoacidosis", "anaphylaxis", "meningitis", "appendicitis"
        ],
    }


settings = Settings()