"""Live verification script demonstrating:
1. Non-blocking asynchronous hosted OpenAI GPT call.
2. Parallel Agent 2 & Agent 3 execution with timestamp logs.
3. Conversational UX: bullet-point-first response, no raw tables, no literal <br>, no literal \\*\\*.
"""

import httpx
import json
import sys
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def test_live_conversation():
    print("--- 1. Testing GET /health ---")
    with httpx.Client(timeout=10.0) as client:
        r = client.get("http://localhost:9000/health")
        print(f"Health Status: {r.status_code}")
        print(json.dumps(r.json(), indent=2))
        assert r.status_code == 200

    print("\n--- 2. Testing POST /chat (Hosted OpenAI GPT + Parallel Retrieval) ---")
    session_id = f"live_verify_{int(time.time())}"
    
    turn1_payload = {
        "message": "I am 55 years old, female. I have had a fever of 102 degrees for the past 2 days. No previous medical conditions and no current medications.",
        "session_id": session_id,
    }

    t0 = time.perf_counter()
    with httpx.Client(timeout=45.0) as client:
        r = client.post("http://localhost:9000/chat", json=turn1_payload)
        t_dur = time.perf_counter() - t0
        print(f"\nResponse Code: {r.status_code} in {t_dur:.2f}s")
        data = r.json()
        
        print("\n================ USER-FACING RENDERED RESPONSE ================")
        print(data["response"])
        print("================================================================")
        
        print("\nLLM Metadata:", json.dumps(data["llm_metadata"], indent=2))
        print("\nPatient State:", json.dumps(data["patient_state"], indent=2))
        print("\nFollow-up Questions Count:", len(data.get("follow_up_questions", [])))
        print("Validation Result:", json.dumps(data.get("validation", {}), indent=2))

        # Check formatting acceptance criteria
        resp = data["response"]
        assert "|----------|" not in resp, "Raw table separator should not appear!"
        assert "<br>" not in resp and "\\<br>" not in resp, "Literal <br> should not appear!"
        assert "\\*\\*" not in resp, "Escaped asterisks should not appear!"

    print("\n--- 3. Testing Turn 2 (Answering inquiry without repeating full history) ---")
    turn2_payload = {
        "message": "No, I do not have any cough, sore throat, or trouble breathing. No headache either.",
        "session_id": session_id,
    }

    t0 = time.perf_counter()
    with httpx.Client(timeout=45.0) as client:
        r2 = client.post("http://localhost:9000/chat", json=turn2_payload)
        t2_dur = time.perf_counter() - t0
        print(f"\nTurn 2 Response Code: {r2.status_code} in {t2_dur:.2f}s")
        data2 = r2.json()

        print("\n================ TURN 2 RENDERED RESPONSE ================")
        print(data2["response"])
        print("==========================================================")
        print("\nUpdated Patient State:", json.dumps(data2["patient_state"], indent=2))

if __name__ == "__main__":
    test_live_conversation()
