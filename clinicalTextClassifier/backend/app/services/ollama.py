"""LLM Service - Ollama Integration (Local)

Provides a clean interface to Ollama running locally for clinical text processing.
Uses the official Ollama Python client.
"""

import json
import requests
from typing import Optional

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:4b-instruct"  # Using qwen3 4B instruction-tuned model

# Test if Ollama is available on startup
_ollama_available = None


def is_ollama_available() -> bool:
    """Check if Ollama service is running and accessible."""
    global _ollama_available

    # Keep successful checks, but retry after failures so Ollama can be started
    # while the application is already running.
    if _ollama_available is True:
        return _ollama_available
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        _ollama_available = response.status_code == 200
        return _ollama_available
    except Exception:
        _ollama_available = False
        return False


def verify_model_installed(model: str = OLLAMA_MODEL) -> bool:
    """Check if the specified model is installed in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            installed_models = [m.get('name', '') for m in data.get('models', [])]
            # Check if model name matches (handle version tags)
            return any(model in m or m.startswith(model.split(':')[0]) for m in installed_models)
    except Exception:
        pass
    return False


def call_ollama_text_only(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    model: str = OLLAMA_MODEL,
    timeout: int = 60,
    num_predict: Optional[int] = None
) -> str:
    """
    Call Ollama with a text prompt and return text response.
    
    Args:
        prompt: The main prompt/input
        system_prompt: Optional system prompt for context
        temperature: Sampling temperature (0 = deterministic)
        model: Model name to use
        timeout: Request timeout in seconds
    
    Returns:
        The response text from Ollama
    
    Raises:
        RuntimeError: If Ollama is not available
        ValueError: If model is not installed
        Exception: If Ollama API call fails
    """
    if not is_ollama_available():
        raise RuntimeError(
            "Ollama is not running. Please start Ollama with:\n"
            "  ollama serve\n"
            "or download from https://ollama.ai"
        )
    
    if not verify_model_installed(model):
        raise ValueError(
            f"Model '{model}' is not installed in Ollama.\n"
            f"Install it with: ollama pull {model}"
        )
    
    full_prompt = prompt
    if system_prompt:
        full_prompt = system_prompt + "\n\n" + prompt
    
    try:
        request_body = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "temperature": temperature,
        }
        if num_predict is not None:
            request_body["options"] = {"num_predict": num_predict}

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=request_body,
            timeout=timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        
        data = response.json()
        return data.get("response", "").strip()
    
    except requests.exceptions.Timeout:
        raise Exception(f"Ollama request timed out after {timeout}s. Model may be processing slowly.")
    except Exception as e:
        raise Exception(f"Ollama API call failed: {str(e)}")


def call_ollama_json(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    model: str = OLLAMA_MODEL,
    timeout: int = 60
) -> dict:
    """
    Call Ollama with a text prompt and expect JSON response.
    
    Args:
        prompt: The main prompt/input
        system_prompt: Optional system prompt for context
        temperature: Sampling temperature (0 = deterministic)
        model: Model name to use
        timeout: Request timeout in seconds
    
    Returns:
        Parsed JSON response as dictionary
    
    Raises:
        RuntimeError: If Ollama is not available
        ValueError: If model is not installed
        Exception: If Ollama API call fails or response is not valid JSON
    """
    response_text = call_ollama_text_only(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        model=model,
        timeout=timeout
    )
    
    # Clean up markdown formatting if present
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse Ollama JSON response: {str(e)}\nResponse was: {response_text}")
