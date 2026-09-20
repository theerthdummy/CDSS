"""
Manual Test Script for Ollama Integration

Run this script to test the Clinical Text Clarifier with Ollama.
Make sure Ollama is running before executing this.

Usage:
    python check_ollama_output.py

Requirements:
    - Ollama service running on http://localhost:11434
    - Model: qwen3:4b-instruct
"""

from app.agent.agent import ClinicalTextClarifierAgent
from app.agent.schema import Agent1Request
from app.services import ollama
import json


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def test_case_1():
    """TEST 1: Complex presentation with high fever 103°F"""
    print_section("TEST 1: Complex Presentation (103°F Fever)")
    
    text = (
        "Patient has had a high fever of 103°F for the past 2 days, "
        "along with severe headache, chills, body pain, and nausea. "
        "The patient also reports feeling very weak."
    )
    
    print(f"\nInput: {text}\n")
    
    agent = ClinicalTextClarifierAgent()
    request = Agent1Request(text=text)
    response = agent.process(request)
    
    print("OUTPUT:")
    print(json.dumps(response.model_dump(), indent=2))
    
    # Validation
    print("\n[VALIDATIONS]:")
    print(f"  [OK] Temperature: {response.measurements.get('temperature')} (should be 103°F, not age)")
    print(f"  [OK] Symptoms: {response.symptoms}")
    print(f"  [OK] No 'chest pain' conversion: {'chest pain' not in response.symptoms}")
    print(f"  [OK] Duration: {response.duration}")
    print(f"  [OK] Severity: {response.severity}")
    print(f"  [OK] Clarification Q: {response.clarification_question}")
    
    return response


def test_case_2():
    """TEST 2: Chest pain for 3 hours"""
    print_section("TEST 2: Chest Pain (3 hours)")
    
    text = "Patient has chest pain for 3 hours."
    
    print(f"\nInput: {text}\n")
    
    agent = ClinicalTextClarifierAgent()
    request = Agent1Request(text=text)
    response = agent.process(request)
    
    print("OUTPUT:")
    print(json.dumps(response.model_dump(), indent=2))
    
    print("\n[VALIDATIONS]:")
    print(f"  [OK] Symptoms: {response.symptoms}")
    print(f"  [OK] Duration: {response.duration}")
    print(f"  [OK] Clarification Q: {response.clarification_question}")
    print(f"  [OK] No redundant 'how long' question: {'how long' not in (response.clarification_question or '').lower()}")
    
    return response


def test_case_3():
    """TEST 3: Multiple symptoms with severity"""
    print_section("TEST 3: Severe Chest Pain with Multiple Symptoms")
    
    text = (
        "Patient has severe chest pain for 3 hours "
        "with sweating and shortness of breath."
    )
    
    print(f"\nInput: {text}\n")
    
    agent = ClinicalTextClarifierAgent()
    request = Agent1Request(text=text)
    response = agent.process(request)
    
    print("OUTPUT:")
    print(json.dumps(response.model_dump(), indent=2))
    
    print("\n[VALIDATIONS]:")
    print(f"  [OK] Symptoms: {response.symptoms}")
    print(f"  [OK] Severity: {response.severity}")
    print(f"  [OK] Duration: {response.duration}")
    print(f"  [OK] Clarification Q: {response.clarification_question}")
    print(f"  [OK] No redundant severity question: {'severity' not in (response.clarification_question or '').lower()}")
    
    return response


def test_case_4():
    """TEST 4: Simple fever"""
    print_section("TEST 4: Simple Fever (Missing Info)")
    
    text = "Patient has fever."
    
    print(f"\nInput: {text}\n")
    
    agent = ClinicalTextClarifierAgent()
    request = Agent1Request(text=text)
    response = agent.process(request)
    
    print("OUTPUT:")
    print(json.dumps(response.model_dump(), indent=2))
    
    print("\n[VALIDATIONS]:")
    print(f"  [OK] Fever detected: {'fever' in response.symptoms}")
    print(f"  [OK] Missing info detected: {response.missing_information}")
    print(f"  [OK] Requires clarification: {response.requires_clarification}")
    print(f"  [OK] Clarification Q: {response.clarification_question}")
    
    return response


def check_ollama_status():
    """Check if Ollama is available and model is installed."""
    print_section("CHECKING OLLAMA STATUS")
    
    # Check if Ollama is running
    if not ollama.is_ollama_available():
        print("\n[ERROR] Ollama is not running!")
        print("\nTo start Ollama:")
        print("  1. Download from https://ollama.ai")
        print("  2. Run: ollama serve")
        return False
    
    print("[OK] Ollama service is running")
    
    # Check if model is installed
    model = ollama.OLLAMA_MODEL
    if not ollama.verify_model_installed(model):
        print(f"\n[WARNING] Model '{model}' is not installed")
        print(f"\nTo install the model:")
        print(f"  ollama pull {model}")
        print("\nNote: You can also use:")
        print("  ollama pull qwen3:4b-instruct")
        return False
    
    print(f"[OK] Model '{model}' is installed")
    return True


def main():
    """Run all test cases."""
    print("\n" + "="*80)
    print("  CLINICAL TEXT CLARIFIER - OLLAMA INTEGRATION TESTS")
    print("="*80)
    
    # First check if Ollama is available
    if not check_ollama_status():
        print("\n[WARNING] TESTS CANNOT RUN - Ollama not available")
        return
    
    try:
        # Run all 4 test cases
        test_case_1()
        test_case_2()
        test_case_3()
        test_case_4()
        
        print_section("ALL TESTS COMPLETED SUCCESSFULLY [OK]")
        
    except Exception as e:
        print_section("ERROR DURING TESTING")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
