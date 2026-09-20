"""
MedBullets MCQ Benchmark Evaluation.

Uses the LangAGI-Lab/medbullets dataset to evaluate the RAG pipeline
on clinical case-based multiple-choice questions (USMLE Step 2/3 style).
Each question is searched against our Qdrant database, context is retrieved,
and Ollama selects the best answer from 4-5 options.

Usage:
    python -m test.evaluate_medbullets
"""

import logging
import sys
import os
from itertools import islice

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import requests
from datasets import load_dataset
from query.search import MedicalSearchEngine

logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


def _load_dataset(max_samples: int | None):
    if max_samples is None:
        return load_dataset("LangAGI-Lab/medbullets", split="train")
    stream = load_dataset("LangAGI-Lab/medbullets", split="train", streaming=True)
    return list(islice(stream, max_samples))

# Map option keys to letters
OPTION_KEYS = ["opa", "opb", "opc", "opd", "ope"]
OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def generate_answer(question: str, options_text: str, context: str) -> str:
    """Send an MCQ question to Ollama with retrieved context."""
    prompt = (
        "You are a medical AI assistant taking a clinical exam. "
        "Based on the provided context and your medical knowledge, "
        "select the single best answer to the following clinical question. "
        "You must respond with ONLY the letter of the correct answer "
        "(A, B, C, D, or E). Do not explain.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Answer:"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
        response.raise_for_status()
        return response.json().get("response", "").strip().upper()
    except Exception as e:
        logger.error("Error querying Ollama: %s", e)
        return "ERROR"


def extract_letter(prediction: str) -> str:
    """Extract the answer letter (A-E) from the model's response."""
    prediction = prediction.strip().upper()
    # Check if first character is a valid letter
    for letter in OPTION_LETTERS:
        if prediction.startswith(letter):
            return letter
    # Fallback: search for any letter in the response
    for letter in OPTION_LETTERS:
        if letter in prediction:
            return letter
    return prediction


def main():
    logger.info("Loading MedBullets dataset...")
    try:
        dataset = _load_dataset(config.MAX_SAMPLES)
    except Exception as e:
        logger.error("Failed to load MedBullets dataset: %s", e)
        return

    samples = dataset
    sample_count = len(samples)

    print(f"Evaluation dataset: MedBullets | Samples: {sample_count}")

    logger.info("Initializing Search Engine...")
    engine = MedicalSearchEngine()

    correct = 0
    total = sample_count

    print("\n" + "=" * 80)
    print(f"  MedBullets MCQ EVALUATION (Model: {MODEL_NAME})")
    print("=" * 80 + "\n")

    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        ground_truth = sample["answer_idx"].strip().upper()

        # Build options text
        options_lines = []
        for key, letter in zip(OPTION_KEYS, OPTION_LETTERS):
            option_text = sample.get(key, "")
            if option_text and str(option_text).lower() != "nan":
                options_lines.append(f"{letter}. {option_text}")
        options_text = "\n".join(options_lines)

        # Retrieve context from our database
        results = engine.search_with_fallback(question, limit=5)

        if not results:
            context = "No relevant context found."
        else:
            parts = []
            for r in results:
                text = r["text"][:400] + "..." if len(r["text"]) > 400 else r["text"]
                parts.append(text)
            context = "\n\n".join(parts)

        # Generate answer
        raw_prediction = generate_answer(question, options_text, context)
        prediction = extract_letter(raw_prediction)

        is_correct = prediction == ground_truth
        if is_correct:
            correct += 1

        # Truncate question for display
        q_display = question[:70].replace("\n", " ")
        print(f"[{i}/{total}] Q: {q_display}...")
        print(
            f"       Truth: {ground_truth:<5} | "
            f"Pred: {prediction:<5} -> "
            f"{'[CORRECT]' if is_correct else '[INCORRECT]'}"
        )

    engine.close()

    accuracy = (correct / total) * 100
    print("\n" + "=" * 80)
    print(f"  MedBullets FINAL ACCURACY: {correct}/{total} ({accuracy:.2f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
