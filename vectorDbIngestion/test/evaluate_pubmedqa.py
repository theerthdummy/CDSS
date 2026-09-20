"""
PubMedQA Benchmark Evaluation.

Uses the qiaojin/PubMedQA dataset (pqa_labeled split) to evaluate the RAG
pipeline on expert-level yes/no/maybe medical questions.

Usage:
    python -m test.evaluate_pubmedqa
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
        return load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
    stream = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train', streaming=True)
    return list(islice(stream, max_samples))

def generate_answer(question: str, context: str) -> str:
    prompt = (
        f"You are a medical AI assistant. Based ONLY on the provided context, answer the following "
        f"medical question. You must answer with exactly one word: 'yes', 'no', or 'maybe'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip().lower()
    except Exception as e:
        logger.error(f"Error querying Ollama: {e}")
        return "error"

def main():
    logger.info("Loading PubMedQA dataset (pqa_labeled)...")
    try:
        dataset = _load_dataset(config.MAX_SAMPLES)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    samples = dataset
    sample_count = len(samples)

    print(f"Evaluation dataset: PubMedQA | Samples: {sample_count}")
    
    logger.info("Initializing Search Engine...")
    engine = MedicalSearchEngine()
    
    correct_count = 0
    total = sample_count
    
    logger.info(f"Starting evaluation on {total} samples...")
    print("\n" + "=" * 80)
    print(f"  PubMedQA EVALUATION (Model: {MODEL_NAME})")
    print("=" * 80 + "\n")
    
    for i, sample in enumerate(samples, 1):
        question = sample['question']
        ground_truth = sample['final_decision'].lower()
        
        # 1. Retrieve Context
        results = engine.search_with_fallback(question, limit=5)
        
        if not results:
            context = "No relevant context found."
        else:
            context_parts = []
            for r in results:
                text = r['text']
                if len(text) > 400:
                    text = text[:400] + "..."
                context_parts.append(text)
            context = "\n\n".join(context_parts)
            
        # 2. Generate Answer via Ollama
        prediction = generate_answer(question, context)
        
        # Clean prediction
        clean_pred = prediction.replace(".", "").replace(",", "").strip()
        if "yes" in clean_pred:
            clean_pred = "yes"
        elif "no" in clean_pred:
            clean_pred = "no"
        elif "maybe" in clean_pred:
            clean_pred = "maybe"
            
        # 3. Check correctness
        is_correct = clean_pred == ground_truth
        if is_correct:
            correct_count += 1
            
        print(f"[{i}/{total}] Q: {question[:60]}...")
        print(f"       Truth: {ground_truth.upper():<5} | Pred: {clean_pred.upper():<5} -> {'[CORRECT]' if is_correct else '[INCORRECT]'}")
        
    engine.close()
    
    accuracy = (correct_count / total) * 100
    print("\n" + "=" * 80)
    print(f"  FINAL ACCURACY: {correct_count}/{total} ({accuracy:.2f}%)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
