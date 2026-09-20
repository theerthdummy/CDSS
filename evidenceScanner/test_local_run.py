"""
test_local_run.py - Integration Test for Agent 3 (PubMed + Tavily Architecture)
"""

import json
from src.scanner_agent import EvidenceScannerAgent, EvidenceRequest

if __name__ == "__main__":
    print("=" * 75)
    print("🚀 TESTING AGENT 3 (EVIDENCE SCANNER) - PUBMED + TAVILY ARCHITECTURE")
    print("=" * 75)

    # Test Case 1: Comorbid Patient (Routes to Tavily only with risk measures)
    print("\n--- TEST CASE 1: Comorbid Presentation (Diabetes + CKD + Fever) ---")
    case_chronic = EvidenceRequest(
        raw_text="Patient with type 2 diabetes and chronic kidney disease presenting with high fever and chills.",
        max_results=3
    )
    agent = EvidenceScannerAgent()
    resp_chronic = agent.run(case_chronic)
    print(json.dumps(agent.get_dict_for_agent4(resp_chronic), indent=2))

    # Test Case 2: Acute Only (Routes to PubMed + Tavily)
    print("\n--- TEST CASE 2: Acute Only (Cough + Fever) ---")
    case_acute = EvidenceRequest(
        raw_text="Patient presenting with acute high fever, cough, and chest pain.",
        max_results=3
    )
    resp_acute = agent.run(case_acute)
    print(json.dumps(agent.get_dict_for_agent4(resp_acute), indent=2))