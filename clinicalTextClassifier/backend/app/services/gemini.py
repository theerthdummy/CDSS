"""LLM Service with Groq Round-Robin Primary + Gemini Fallback.

Execution Order:
1. Try Groq API keys 1-4 (round-robin) using Llama 3.1 via OpenAI-compatible endpoint.
2. If all Groq keys fail or are unconfigured, fall back to Google Gemini API.
"""

import json
import os
import time
import logging
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("agent1.llm")

# --- Groq Configuration ---
GROQ_API_KEYS: List[str] = [
    k.strip() for k in [
        os.getenv("GROQ_API_KEY_1", ""),
        os.getenv("GROQ_API_KEY_2", ""),
        os.getenv("GROQ_API_KEY_3", ""),
        os.getenv("GROQ_API_KEY_4", ""),
    ] if k.strip()
]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
GROQ_API_BASE = "https://api.groq.com/openai/v1"

# --- Gemini Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()


def is_groq_available() -> bool:
    """Check if any Groq API keys are configured."""
    return len(GROQ_API_KEYS) > 0


def is_gemini_available() -> bool:
    """Check if a Gemini API key is configured."""
    return bool(GEMINI_API_KEY)


def call_groq_json(
    prompt: str,
    system_prompt: str = "",
    model: str = GROQ_MODEL,
    temperature: float = 0.0,
    timeout: int = 15,
) -> Optional[Dict[str, Any]]:
    """Call Groq API with 4-key round-robin fallback. Returns parsed JSON dict or None if all keys fail."""
    if not GROQ_API_KEYS:
        return None

    endpoint = f"{GROQ_API_BASE}/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    for key_idx, api_key in enumerate(GROQ_API_KEYS, start=1):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)

            if response.status_code == 401:
                logger.warning(f"Groq key {key_idx}/{len(GROQ_API_KEYS)} auth failed (401). Trying next...")
                continue
            if response.status_code == 429:
                logger.warning(f"Groq key {key_idx}/{len(GROQ_API_KEYS)} rate limited (429). Trying next...")
                time.sleep(1.0)
                continue
            if response.status_code == 404 and model == "llama-3.1-8b-instruct":
                # Retry with alternative model name
                payload["model"] = "llama-3.1-8b-instant"
                response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)

            if response.status_code != 200:
                logger.warning(f"Groq key {key_idx} returned HTTP {response.status_code}. Trying next...")
                continue

            data = response.json()
            raw_text = data["choices"][0]["message"]["content"].strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            return json.loads(raw_text.strip())

        except requests.exceptions.RequestException as e:
            logger.warning(f"Groq key {key_idx} connection error: {e}. Trying next...")
            continue
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Groq key {key_idx} response parse error: {e}. Trying next...")
            continue

    logger.warning("All Groq API keys exhausted. Falling back to Gemini.")
    return None


def call_gemini_json(
    prompt: str,
    system_prompt: str = "",
    model: str = GEMINI_MODEL,
    temperature: float = 0.0,
    timeout: int = 15,
) -> Dict[str, Any]:
    """Call Google Gemini API with guaranteed structured JSON output."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"

    user_content = prompt
    if system_prompt:
        user_content = f"{system_prompt}\n\n---\n{prompt}"

    payload = {
        "contents": [
            {
                "parts": [{"text": user_content}]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": temperature,
        }
    }

    headers = {"Content-Type": "application/json"}

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if response.status_code == 429 and attempt < max_retries:
                time.sleep(2.5 * (attempt + 1))
                continue
            if response.status_code != 200:
                raise RuntimeError(f"Gemini API error ({response.status_code}): {response.text}")

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("No candidate returned by Gemini API")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise RuntimeError("Empty content parts in Gemini response")

            raw_text = parts[0].get("text", "{}").strip()
            return json.loads(raw_text)
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(2.0)
                continue
            raise RuntimeError(f"Gemini connection error: {e}")


def call_llm_json(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    timeout: int = 15,
) -> Optional[Dict[str, Any]]:
    """Unified LLM call: tries Groq keys first (round-robin), then falls back to Gemini.
    
    Returns parsed JSON dict, or None if both Groq and Gemini fail.
    """
    # 1. Try Groq round-robin first
    if is_groq_available():
        result = call_groq_json(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            timeout=timeout,
        )
        if result is not None:
            return result

    # 2. Fall back to Gemini
    if is_gemini_available():
        try:
            return call_gemini_json(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(f"Gemini fallback also failed: {e}")
            return None

    logger.error("No LLM providers configured. Set GROQ_API_KEY_1..4 or GEMINI_API_KEY.")
    return None
