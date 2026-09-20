"""
BioASQ Yes/No Benchmark Evaluation.

Uses the jmhb/BioASQ dataset (yesno split) to evaluate the RAG pipeline
on biomedical yes/no questions. Each question is searched against our
Qdrant database, context is retrieved, and Ollama generates a yes/no answer.

Usage:
    python -m test.evaluate_bioasq
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
        return load_dataset("jmhb/BioASQ", split="yesno")
    stream = load_dataset("jmhb/BioASQ", split="yesno", streaming=True)
    return list(islice(stream, max_samples))


def generate_answer(question: str, context: str) -> str:
    """Send a yes/no question to Ollama with retrieved context."""
    prompt = (
        "You are a biomedical AI assistant. Based ONLY on the provided context, "
        "answer the following biomedical question. "
        "You must answer with exactly one word: 'yes' or 'no'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip().lower()
    except Exception as e:
        logger.error("Error querying Ollama: %s", e)
        return "error"


def main():
    logger.info("Loading BioASQ dataset (yesno split)...")
    try:
        dataset = _load_dataset(config.MAX_SAMPLES)
    except Exception as e:
        logger.error("Failed to load BioASQ dataset: %s", e)
        return

    samples = dataset
    sample_count = len(samples)

    print(f"Evaluation dataset: BioASQ | Samples: {sample_count}")

    logger.info("Initializing Search Engine...")
    engine = MedicalSearchEngine()

    correct = 0
    total = sample_count

    print("\n" + "=" * 80)
    print(f"  BioASQ Yes/No EVALUATION (Model: {MODEL_NAME})")
    print("=" * 80 + "\n")

    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        ground_truth = sample["answer"].strip().lower()

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
        prediction = generate_answer(question, context)

        # Clean prediction
        clean_pred = prediction.replace(".", "").replace(",", "").strip()
        if "yes" in clean_pred:
            clean_pred = "yes"
        elif "no" in clean_pred:
            clean_pred = "no"

        is_correct = clean_pred == ground_truth
        if is_correct:
            correct += 1

        print(f"[{i}/{total}] Q: {question[:70]}...")
        print(
            f"       Truth: {ground_truth.upper():<5} | "
            f"Pred: {clean_pred.upper():<5} -> "
            f"{'[CORRECT]' if is_correct else '[INCORRECT]'}"
        )

    engine.close()

    accuracy = (correct / total) * 100
    print("\n" + "=" * 80)
    print(f"  BioASQ FINAL ACCURACY: {correct}/{total} ({accuracy:.2f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
