import pytest
from src.scanner_agent import EvidenceScannerAgent, EvidenceRequest


def test_multi_source_evidence_scanner():
    agent = EvidenceScannerAgent()
    request = EvidenceRequest(entities=["fever", "sob"], target_count_per_source=2)
    
    response = agent.run(request)
    
    assert response["status"] in ["SUCCESS", "NO_EVIDENCE_FOUND"]
    assert "PubMed" in response["sources_queried"]
    assert "Europe PMC" in response["sources_queried"]
    assert isinstance(response["evidence_records"], list)
    
    if response["evidence_records"]:
        record = response["evidence_records"][0]
        assert "source" in record
        assert "id" in record
        assert "title" in record
        assert "abstract" in record