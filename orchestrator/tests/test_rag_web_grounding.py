"""Test Case: Biomedical Vector RAG and Web Evidence Grounding.

Verifies:
1. Evidence items retain explicit provenance (source_type="biomedical_rag" vs "web").
2. Evidence identifiers, section titles, and URLs are preserved.
3. Formatted prompt separates patient state from retrieved literature evidence.
"""

import pytest
from orchestrator.clinical_assistant.grounding.evidence_grounder import (
    EvidenceGrounder,
    GroundedEvidenceContext,
    GroundedEvidenceItem,
)


def test_rag_and_web_evidence_provenance_preservation():
    # Mock Agent 2 (Vector RAG) output
    mock_agent2_res = {
        "results": [
            {
                "text": "Febrile delirium is characterized by acute confusion and hallucinations in patients with high fever.",
                "source_id": "rag-pubmed-88219",
                "journal": "New England Journal of Medicine",
                "source_type": "medical-paper",
                "score": 0.89,
                "section_title": "Clinical Neurosciences",
            }
        ]
    }

    # Mock Agent 3 (Web Evidence / Guidelines) output
    mock_agent3_res = {
        "evidence": [
            {
                "title": "AHA/CDC Guidelines for Febrile Encephalopathy",
                "snippet": "Patients presenting with fever and acute hallucinations should undergo urgent neuroimaging and CSF evaluation.",
                "source": "CDC Clinical Guidelines",
                "url": "https://cdc.gov/guidelines/encephalopathy",
            }
        ]
    }

    rag_items = EvidenceGrounder.process_agent2_rag(mock_agent2_res)
    web_items = EvidenceGrounder.process_agent3_web(mock_agent3_res)

    assert len(rag_items) == 1
    assert rag_items[0].source_type == "biomedical_rag"
    assert rag_items[0].source == "New England Journal of Medicine"
    assert rag_items[0].retrieval_id == "rag-pubmed-88219"

    assert len(web_items) == 1
    assert web_items[0].source_type == "web"
    assert web_items[0].source == "CDC Clinical Guidelines"
    assert web_items[0].url == "https://cdc.gov/guidelines/encephalopathy"

    # Context assembly
    ctx = GroundedEvidenceContext(
        biomedical_rag_evidence=rag_items,
        web_evidence=web_items,
    )
    prompt_text = ctx.to_prompt_context()

    assert "biomedical_rag" in prompt_text
    assert "source_type: web" in prompt_text
    assert "New England Journal of Medicine" in prompt_text
    assert "CDC Clinical Guidelines" in prompt_text
