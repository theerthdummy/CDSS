"""config.py - Configuration for the CDSS Orchestrator and Conversational Assistant."""

import os
from dotenv import load_dotenv

load_dotenv()

# Microservice Endpoints
AGENT1_URL = os.getenv("AGENT1_URL", "http://localhost:8000")
AGENT2_URL = os.getenv("AGENT2_URL", "http://localhost:8002")
AGENT3_URL = os.getenv("AGENT3_URL", "http://localhost:8003")
AGENT4_URL = os.getenv("AGENT4_URL", "http://localhost:8004")
AGENT5_URL = os.getenv("AGENT5_URL", "http://localhost:8005")

# Orchestration Configuration
MAX_FEEDBACK_LOOPS = int(os.getenv("MAX_FEEDBACK_LOOPS", "3"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

# Hosted OpenAI GPT Configuration (Hard Requirement: Primary Conversational Model)
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("GROQ_API_KEY")
    or ""
).strip()

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.groq.com/openai/v1" if (OPENAI_API_KEY.startswith("gsk_") or not os.getenv("OPENAI_API_KEY")) else "https://api.openai.com/v1"
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "openai/gpt-oss-120b" if "groq.com" in OPENAI_BASE_URL else "gpt-4o"
)
