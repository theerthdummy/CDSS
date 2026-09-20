"""
adapters.py - Schema Transformation Layer for CDSS Orchestrator

This module bridges the data schema gaps between independently developed agents.
It ensures that the output from one agent is perfectly formatted for the next
agent in the pipeline.
"""

from typing import Dict, Any, Tuple

def agent1_to_search_query(agent1_output: Dict[str, Any]) -> str:
    """
    Transforms Agent 1's structured output into a flat query string for Agent 2 (RAG).
    Combines symptoms, medical history, and demographics.
    """
    symptoms = agent1_output.get("symptoms", [])
    history = agent1_output.get("medical_history", [])
    
    # Combine lists into a single space-separated query string
    query_parts = symptoms + history
    query = " ".join(query_parts)
    
    # Fallback if nothing was extracted
    if not query.strip():
        query = "general clinical presentation"
        
    return query


def agent1_to_evidence_request(agent1_output: Dict[str, Any], raw_patient_text: str) -> Dict[str, Any]:
    """
    Transforms Agent 1's structured output into an EvidenceRequest for Agent 3 (Scanner).
    """
    return {
        "raw_text": raw_patient_text,
        "entities": agent1_output.get("symptoms", []),
        "chronic_conditions": agent1_output.get("medical_history", []),
        "condition_hint": "", # Agent 3 will infer this using LLM if left blank
        "max_results": 3
    }


def agent2_to_fusion_input(search_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wraps Agent 2's SearchResponse into the expected RawAgent2Payload for Agent 4 (Data Fusion).
    Explicitly handles the rule: NO Knowledge Graph (kg_evidence = []).
    """
    # Agent 2 returns: {"query": "...", "top_k": 5, "total_found": 5, "results": [...]}
    results = search_response.get("results", [])
    
    mapped_results = []
    for r in results:
        mapped_results.append({
            "text": r.get("text", ""),
            "score": r.get("score", 0.0),
            "confidence": r.get("score", 0.0), # Use vector score as confidence
            "source": r.get("source_type", "RAG Database"),
            "document_type": "literature",
            "title": r.get("section_title", "Medical Evidence")
        })

    return {
        "success": True,
        "message": "Mapped successfully from Agent 2",
        "data": {
            "query": search_response.get("query", ""),
            "results": mapped_results
        },
        "kg_evidence": [], # Rule: Knowledge Graph is excluded
        "metadata": {"source_agent": "Agent 2"}
    }


def agent3_to_fusion_input(scanner_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wraps Agent 3's ScannerResponse into the expected RawAgent3Payload for Agent 4 (Data Fusion).
    """
    return {
        "entities_queried": scanner_response.get("acute_symptoms", []),
        "sources_queried": scanner_response.get("sources_queried", []),
        "fallback_used": scanner_response.get("fallback_used", False),
        "summary": scanner_response.get("summary", ""),
        "evidence": scanner_response.get("evidence", []),
        "metadata": {"source_agent": "Agent 3"}
    }


def feedback_to_queries(feedback_response: Dict[str, Any], agent1_output: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Parses Agent 5's FeedbackRequest and refines the queries for Agent 2 and Agent 3.
    Returns:
        - refined_agent2_query: str
        - refined_agent3_request: dict
    """
    # Agent 5 feedback typically requests missing information or flags low confidence.
    evidence_gaps = " ".join(feedback_response.get("evidence_gaps", []))
    
    # 1. Refine Agent 2 (RAG) query by appending the evidence gaps
    base_query = agent1_to_search_query(agent1_output)
    refined_agent2_query = f"{base_query} {evidence_gaps}".strip()
    
    # 2. Refine Agent 3 (Scanner) request by setting condition_hint
    refined_agent3_request = agent1_to_evidence_request(agent1_output, raw_patient_text="")
    
    # Inject feedback gaps as a hint to guide PubMed/Tavily search
    refined_agent3_request["condition_hint"] = evidence_gaps
    
    return refined_agent2_query, refined_agent3_request
