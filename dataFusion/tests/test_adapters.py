"""Unit and integration tests for Upstream Agent Output Adapters (app/adapters.py)."""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.adapters import adapt_agent2_output, adapt_agent3_output
from app.main import app
from app.schemas import Agent2Output, Agent3Output
from tests.test_data import VALID_AGENT2_OUTPUT, VALID_AGENT3_OUTPUT, VALID_LLM_FUSION_RESULT

client = TestClient(app)

# --- Mock Real Agent 2 Raw Payload ---
REAL_AGENT2_RAW_PAYLOAD = {
    "success": True,
    "message": "Biomedical RAG and KG retrieval successful",
    "data": {
        "query": "ST-elevation Myocardial Infarction treatment",
        "results": [
            {
                "text": "ST-elevation myocardial infarction (STEMI) requires immediate emergency reperfusion therapy.",
                "score": 0.95,
                "confidence": 0.92,
                "source": "Biomedical PubMed Vector Store",
                "document_type": "literature",
            },
            {
                "disease": "Acute Myocardial Infarction",
                "document_type": "knowledge_graph",
                "confidence": 0.88,
                "graph_context": [
                    {"source": "Acute Myocardial Infarction", "target": "Shortness of Breath", "type": "associated_symptom"},
                    {"source": "Acute Myocardial Infarction", "target": "Aspirin", "type": "treatment_option"},
                ],
            },
        ],
        "retrieval_metadata": {
            "execution_time_ms": 12.4,
            "total_documents": 2,
        },
    },
    "metadata": {
        "environment": "production",
    },
}

# --- Mock Real Agent 3 Raw Payload ---
REAL_AGENT3_RAW_PAYLOAD = {
    "entities_queried": ["Acute Myocardial Infarction", "STEMI"],
    "sources_queried": ["PubMed", "EuropePMC", "ClinicalGuidelines"],
    "fallback_used": False,
    "summary": "Administer Aspirin 325 mg orally immediately. Contraindicated in severe bleeding or active aspirin allergy.",
    "evidence": [
        {
            "source": "PubMed",
            "title": "Efficacy of early antiplatelet therapy in acute myocardial infarction.",
            "abstract": "Studies show 325mg aspirin reduces 30-day mortality in STEMI patients.",
            "doi": "10.1016/j.jacc.2025.01.001",
            "id": "38192011",
            "publication_year": 2025,
            "url": "https://pubmed.ncbi.nlm.nih.gov/38192011/",
        },
        {
            "source": "EuropePMC",
            "title": "European Society of Cardiology STEMI Guidelines",
            "abstract": "12-lead ECG within 10 minutes of first medical contact.",
            "id": "PMC9912044",
            "publication_year": 2024,
            "url": "https://europepmc.org/articles/PMC9912044",
        },
        {
            "source": "Clinical Guidelines",
            "title": "A 12-lead Electrocardiogram (ECG) must be performed immediately for chest pain.",
            "abstract": "ECG diagnostic protocols for ACS.",
        },
    ],
    "metadata": {
        "agent": "Agent3_EvidenceScanner",
    },
}


def test_adapt_agent2_output_real_raw_payload():
    """Verify that adapt_agent2_output converts real Agent 2 raw JSON into Agent2Output."""
    adapted = adapt_agent2_output(REAL_AGENT2_RAW_PAYLOAD)

    assert isinstance(adapted, Agent2Output)
    assert len(adapted.rag_evidence) == 1
    assert len(adapted.kg_evidence) == 1

    rag = adapted.rag_evidence[0]
    assert len(rag.retrieved_literature) == 1
    assert "reperfusion therapy" in rag.retrieved_literature[0]
    assert rag.similarity_score == 0.95
    assert rag.confidence_score == 0.92

    kg = adapted.kg_evidence[0]
    assert kg.disease_info == "Acute Myocardial Infarction"
    assert len(kg.relationships) == 2
    assert kg.relationships[0].source == "Acute Myocardial Infarction"
    assert kg.relationships[0].target == "Shortness of Breath"

    assert adapted.metadata is not None
    assert adapted.metadata.get("query") == "ST-elevation Myocardial Infarction treatment"


def test_adapt_agent3_output_real_raw_payload():
    """Verify that adapt_agent3_output converts real Agent 3 raw JSON into Agent3Output."""
    adapted = adapt_agent3_output(REAL_AGENT3_RAW_PAYLOAD)

    assert isinstance(adapted, Agent3Output)
    assert len(adapted.medical_evidence) == 1

    med = adapted.medical_evidence[0]
    assert "Administer Aspirin 325 mg" in med.latest_evidence
    assert len(med.pubmed_articles) == 1
    assert med.pubmed_articles[0].pmid == "38192011"
    assert med.pubmed_articles[0].publication_year == 2025

    assert len(med.europe_pmc_articles) == 1
    assert med.europe_pmc_articles[0].pmcid == "PMC9912044"

    assert len(med.clinical_guidelines) >= 1
    assert "12-lead Electrocardiogram (ECG)" in med.clinical_guidelines[0]

    assert med.source_info.get("fallback_used") is False


def test_adapt_agent2_missing_optional_fields():
    """Verify that missing optional fields in Agent 2 payload do not crash adapter and assign fallbacks."""
    raw = {
        "data": {
            "results": [
                {"text": "Sample clinical literature text."}
            ]
        }
    }
    adapted = adapt_agent2_output(raw)
    assert isinstance(adapted, Agent2Output)
    assert len(adapted.rag_evidence) == 1
    assert adapted.rag_evidence[0].similarity_score == 0.85
    assert adapted.rag_evidence[0].confidence_score == 0.85


def test_adapt_agent3_missing_summary_fallback():
    """Verify that missing 'summary' field in Agent 3 payload falls back gracefully to extracted guideline text."""
    raw = {
        "evidence": [
            {
                "source": "Clinical Guidelines",
                "title": "Perform immediate 12-lead ECG.",
            }
        ]
    }
    adapted = adapt_agent3_output(raw)
    assert isinstance(adapted, Agent3Output)
    assert adapted.medical_evidence[0].latest_evidence == "Perform immediate 12-lead ECG."


def test_adapt_unexpected_extra_fields():
    """Verify that unexpected extra fields in raw payloads do not cause errors and are stored in metadata."""
    raw_agent2 = {
        "custom_vendor_field": "vendor_123",
        "data": {
            "results": [{"text": "Sample text"}],
            "query": "chest pain",
        },
        "metadata": {"extra_param": 999},
    }
    adapted = adapt_agent2_output(raw_agent2)
    assert adapted.metadata is not None
    assert adapted.metadata.get("extra_param") == 999


def test_adapt_malformed_payloads_raise_value_error():
    """Verify that non-dict or malformed raw payloads raise descriptive ValueError exceptions."""
    with pytest.raises(ValueError, match="Invalid Agent 2 raw input type"):
        adapt_agent2_output("this is a string, not a dict")

    with pytest.raises(ValueError, match="Invalid Agent 3 raw input type"):
        adapt_agent3_output(12345)


def test_adapter_pass_through_existing_models():
    """Verify that passing already-normalized Agent2Output / Agent3Output instances returns them unchanged."""
    res2 = adapt_agent2_output(VALID_AGENT2_OUTPUT)
    assert res2 is VALID_AGENT2_OUTPUT

    res3 = adapt_agent3_output(VALID_AGENT3_OUTPUT)
    assert res3 is VALID_AGENT3_OUTPUT


@patch("app.fusion.enhance_clinical_context_with_llama")
def test_end_to_end_route_with_real_raw_payloads(mock_llm):
    """Integration test verifying /api/v1/fuse accepts raw real Agent 2 and Agent 3 payloads, normalizes, and fuses context."""
    mock_stats = {"latency_ms": 25.0, "token_usage": {"total_tokens": 120}}
    mock_llm.return_value = (VALID_LLM_FUSION_RESULT, mock_stats)

    request_payload = {
        "agent2_output": REAL_AGENT2_RAW_PAYLOAD,
        "agent3_output": REAL_AGENT3_RAW_PAYLOAD,
    }

    response = client.post(
        "/api/v1/fuse",
        json=request_payload,
        headers={"X-Correlation-ID": "test-adapter-e2e"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "unified_context" in data
    assert "metadata" in data

    metadata = data["metadata"]
    assert metadata["correlation_id"] == "test-adapter-e2e"
    assert metadata["fusion_mode"] == "llama_enhanced"


def test_adapt_raw_agent_payload_pydantic_models():
    """Verify that RawAgent2Payload and RawAgent3Payload Pydantic instances adapt cleanly."""
    from app.schemas import RawAgent2Payload, RawAgent3Payload

    raw2_model = RawAgent2Payload.model_validate(REAL_AGENT2_RAW_PAYLOAD)
    raw3_model = RawAgent3Payload.model_validate(REAL_AGENT3_RAW_PAYLOAD)

    adapted2 = adapt_agent2_output(raw2_model)
    adapted3 = adapt_agent3_output(raw3_model)

    assert isinstance(adapted2, Agent2Output)
    assert isinstance(adapted3, Agent3Output)
    assert len(adapted2.rag_evidence[0].retrieved_literature) > 0
    assert len(adapted3.medical_evidence[0].pubmed_articles) > 0
