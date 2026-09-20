"""
Central configuration for the Medical RAG Vector Database project.
All tunable parameters, API settings, and search queries are defined here.
"""

# =============================================================================
# API CREDENTIALS (User must set these)
# =============================================================================

import os
from dotenv import load_dotenv

load_dotenv()


def _parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer environment variable.

    Accepts empty strings and literal 'None'/'none' as None.
    """
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "none":
        return None
    parsed = int(cleaned)
    if parsed <= 0:
        raise ValueError("MAX_SAMPLES must be a positive integer or None.")
    return parsed

NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "your.email@example.com")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")

# =============================================================================
# EMBEDDING MODEL
# =============================================================================

EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"  # BioBERT fine-tuned for sentence similarity
DENSE_VECTOR_DIM = 768               # BioBERT outputs 768-dimensional dense vectors
EMBEDDING_BATCH_SIZE = 32            # BioBERT is lighter than BGE-M3, can handle larger batches

# =============================================================================
# QDRANT DATABASE (Docker-based)
# =============================================================================

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_CONTAINER_NAME = "vectordb-cdss"
QDRANT_DOCKER_IMAGE = "qdrant/qdrant:latest"
QDRANT_HOST_PORT = 6333
QDRANT_STORAGE_VOLUME = "vectordb-cdss-data"

COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "medical_knowledge")
UPSERT_BATCH_SIZE = 500

# =============================================================================
# TEXT PROCESSING
# =============================================================================

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MIN_CHUNK_LENGTH = 50

# =============================================================================
# MEMORY MANAGEMENT (16GB RAM Safe)
# =============================================================================

# Maximum chunks to embed+store in a single sub-batch before freeing memory.
# With 16GB RAM, ~10,000 chunks × (768 floats × 4 bytes) ≈ 30MB for dense vectors alone.
# Keeping this at 10,000 keeps peak RAM well under control.
PROCESS_SUB_BATCH_SIZE = 10000

# =============================================================================
# DATA INGESTION VOLUMES
# =============================================================================

PMC_MAX_ARTICLES = 5000
EPMC_MAX_ARTICLES = 1500
CLINICAL_TRIALS_MAX = 2000
HUGGINGFACE_MAX_SAMPLES = 30000
PREPRINT_MAX_ARTICLES = 1000

# =============================================================================
# BENCHMARK EVALUATION SAMPLES
# =============================================================================

MAX_SAMPLES = _parse_optional_int(os.environ.get("MAX_SAMPLES"))

# =============================================================================
# RATE LIMITING (seconds between API requests)
# =============================================================================

PMC_DELAY = 0.12
EPMC_DELAY = 0.15
CT_DELAY = 1.2
PREPRINT_DELAY = 0.15

# =============================================================================
# MEDICAL SEARCH QUERIES (50 curated queries for broad clinical coverage)
# =============================================================================

SEARCH_QUERIES = [
    # --- Cardiovascular ---
    "chest pain diagnosis clinical findings",
    "myocardial infarction symptoms treatment",
    "cardiac arrhythmia symptom presentation",
    "heart failure clinical assessment management",
    "hypertension diagnosis treatment guidelines",
    "deep vein thrombosis pulmonary embolism diagnosis",
    "atrial fibrillation anticoagulation management",
    "valvular heart disease murmur assessment",

    # --- Respiratory ---
    "shortness of breath differential diagnosis",
    "pneumonia clinical diagnosis treatment",
    "asthma exacerbation assessment management",
    "chronic obstructive pulmonary disease symptoms",
    "pulmonary fibrosis interstitial lung disease",
    "pleural effusion thoracentesis management",

    # --- Neurological ---
    "headache neurological evaluation differential",
    "stroke symptoms acute management guidelines",
    "seizure epilepsy clinical evaluation",
    "dizziness vertigo differential diagnosis",
    "multiple sclerosis diagnosis clinical features",
    "parkinson disease tremor management",
    "meningitis encephalitis clinical diagnosis",

    # --- Gastrointestinal ---
    "abdominal pain emergency assessment diagnosis",
    "liver disease clinical findings hepatitis",
    "inflammatory bowel disease symptoms diagnosis",
    "gastrointestinal bleeding evaluation management",
    "pancreatitis acute chronic clinical management",
    "celiac disease malabsorption diagnosis",

    # --- Endocrine & Metabolic ---
    "diabetes mellitus clinical presentation management",
    "thyroid disorder symptom evaluation diagnosis",
    "adrenal insufficiency clinical assessment",
    "obesity metabolic syndrome clinical management",
    "diabetic ketoacidosis emergency treatment",

    # --- Renal & Urological ---
    "renal failure symptom assessment diagnosis",
    "urinary tract infection diagnosis treatment",
    "kidney stone clinical presentation management",
    "chronic kidney disease staging management",

    # --- Infectious Disease ---
    "fever infection differential diagnosis workup",
    "sepsis early recognition diagnosis treatment",
    "tuberculosis clinical diagnosis management",
    "HIV AIDS clinical presentation diagnosis",
    "malaria dengue tropical disease diagnosis",
    "antibiotic resistance antimicrobial stewardship",

    # --- Hematological ---
    "anemia differential diagnosis evaluation",
    "bleeding disorders clinical assessment",
    "leukemia lymphoma hematologic malignancy diagnosis",

    # --- Musculoskeletal ---
    "joint pain arthritis differential diagnosis",
    "back pain clinical evaluation red flags",
    "osteoporosis fracture risk assessment",

    # --- Dermatological ---
    "skin rash dermatitis differential diagnosis",
    "melanoma skin cancer screening diagnosis",

    # --- Psychiatric ---
    "depression anxiety clinical assessment screening",
    "psychosis schizophrenia clinical evaluation",

    # --- Obstetrics & Gynecology ---
    "pregnancy complications preeclampsia management",
    "polycystic ovary syndrome diagnosis treatment",

    # --- Pediatric ---
    "pediatric fever assessment management",
    "childhood asthma wheezing diagnosis",

    # --- Ophthalmology ---
    "acute vision loss differential diagnosis",
    "glaucoma screening optic nerve assessment",

    # --- Emergency & Trauma ---
    "trauma assessment emergency management",
    "poisoning toxicology clinical management",
    "allergic reaction anaphylaxis diagnosis treatment",
    "acute chest syndrome sickle cell management",
    "burn injury assessment fluid resuscitation",

    # --- Oncology ---
    "breast cancer screening staging treatment",
    "lung cancer non-small cell mutation targeted therapy",
    "colorectal cancer diagnosis surgical management",
    "prostate cancer psa screening biopsy",

    # --- Rheumatology & Immunology ---
    "rheumatoid arthritis joint pain biologics",
    "systemic lupus erythematosus flare management",
    "ankylosing spondylitis back pain diagnosis",
    "gout hyperuricemia acute flare treatment",

    # --- Infectious Disease (Expanded) ---
    "lyme disease tick bite erythema migrans",
    "syphilis neurosyphilis screening penicillin",
    "COVID-19 SARS-CoV-2 respiratory failure management",
    "hepatitis C viral load antiviral therapy",

    # --- Endocrinology (Expanded) ---
    "hyperthyroidism graves disease presentation",
    "hypothyroidism hashimoto fatigue evaluation",
    "cushing syndrome hypercortisolism diagnosis",

    # --- Neurology (Expanded) ---
    "alzheimer disease dementia cognitive decline",
    "amyotrophic lateral sclerosis weakness progression",
    "myasthenia gravis ptosis fatigue crisis",

    # --- Geriatrics ---
    "delirium elderly confused acute state",
    "polypharmacy falls risk assessment geriatrics",
]

# =============================================================================
# HUGGINGFACE DATASETS
# =============================================================================

HUGGINGFACE_DATASETS = [
    {
        "name": "gretelai/symptom_to_diagnosis",
        "text_field": "output_text",
        "label_field": "input_text",
        "split": "train",
    },
    {
        "name": "QuyenAnhDE/Diseases_Symptoms",
        "text_field": None,
        "label_field": None,
        "split": "train",
    },
]

# =============================================================================
# SOURCE PRIORITY (Score Re-Ranking Multipliers)
# =============================================================================

SOURCE_PRIORITY = {
    "clinical-trial":          1.15,
    "medical-paper":           1.10,
    "symptom-disease-dataset": 1.00,
    "preprint":                0.70,
}

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
