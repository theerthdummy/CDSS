"""Upstream Agent Output Adapters for Agent 4 (Clinical Data Fusion).

This module isolates Agent 4 from response schema changes in upstream Agent 2 and Agent 3.
It normalizes raw JSON payloads from Agent 2 (Biomedical RAG & Knowledge Graph) and Agent 3
(Evidence Scanner) into Agent 4 internal Pydantic models (Agent2Output, Agent3Output).
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import ValidationError

from app.schemas import (
    Agent2Output,
    Agent3Output,
    EuropePMCArticle,
    KnowledgeGraphEvidence,
    MedicalEvidence,
    PubMedArticle,
    RAGEvidence,
    Relationship,
)
from app.utils import logger


def _normalize_confidence_score(val: Any, default: float = 0.85) -> float:
    """Normalize score values to a float between 0.0 and 1.0."""
    if val is None:
        return default
    try:
        score = float(val)
        if score > 1.0 and score <= 100.0:
            score = score / 100.0
        return max(0.0, min(1.0, round(score, 2)))
    except (ValueError, TypeError):
        return default


def adapt_agent2_output(raw_input: Union[Dict[str, Any], Any]) -> Agent2Output:
    """Adapt raw Agent 2 output payload into Agent 4 internal Agent2Output model.

    Supports:
      1. Direct Agent2Output instances or internal dict structure (rag_evidence/kg_evidence keys).
      2. RawAgent2Payload Pydantic model or dict representation (success, message, data.results[], data.query).

    Raises:
        ValueError: If raw_input is malformed or missing critical structure.
    """
    if isinstance(raw_input, Agent2Output):
        logger.debug("Agent 2 input is already a normalized Agent2Output instance.")
        return raw_input

    if hasattr(raw_input, "model_dump"):
        raw_input = raw_input.model_dump()

    if not isinstance(raw_input, dict):
        error_msg = f"Invalid Agent 2 raw input type: expected dict or RawAgent2Payload, got {type(raw_input).__name__}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Executing Agent 2 payload adapter...")

    # Direct internal model dictionary format check
    if "rag_evidence" in raw_input or "kg_evidence" in raw_input:
        try:
            logger.info("Agent 2 raw payload matches internal schema. Validating directly.")
            return Agent2Output.model_validate(raw_input)
        except ValidationError as err:
            logger.warning(f"Direct validation of Agent 2 payload failed: {err}. Attempting flexible adaptation.")

    # Process Real Agent 2 Raw Payload Format
    data = raw_input.get("data") if isinstance(raw_input.get("data"), dict) else raw_input
    results = data.get("results", []) if isinstance(data, dict) else []

    retrieved_literature: List[str] = []
    kg_evidences: List[KnowledgeGraphEvidence] = []
    similarity_scores: List[float] = []
    confidence_scores: List[float] = []
    dropped_fields: List[str] = []

    # Process data results array
    if isinstance(results, list):
        for idx, item in enumerate(results):
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict item in Agent 2 data.results at index {idx}.")
                continue

            doc_type = str(item.get("document_type", "")).lower()
            text_content = item.get("text") or item.get("content") or item.get("summary")
            score = item.get("score") or item.get("similarity") or item.get("similarity_score")
            conf = item.get("confidence") or item.get("confidence_score")

            if text_content and isinstance(text_content, str) and text_content.strip():
                retrieved_literature.append(text_content.strip())
                if score is not None:
                    similarity_scores.append(_normalize_confidence_score(score))
                if conf is not None:
                    confidence_scores.append(_normalize_confidence_score(conf))

            # Process Knowledge Graph / disease relationships if present
            graph_context = item.get("graph_context") or item.get("relationships")
            disease_name = item.get("disease") or item.get("title") or data.get("query") or "Clinical Condition"

            if graph_context and isinstance(graph_context, list):
                relationships: List[Relationship] = []
                for rel in graph_context:
                    if isinstance(rel, dict):
                        rel_source = str(rel.get("source", disease_name)).strip()
                        rel_target = str(rel.get("target", "") or rel.get("object", "")).strip()
                        rel_type = str(rel.get("relation", "") or rel.get("type", "") or rel.get("predicate", "associated_with")).strip()
                        rel_conf = _normalize_confidence_score(rel.get("confidence"))
                        if rel_source and rel_target:
                            relationships.append(
                                Relationship(
                                    source=rel_source,
                                    target=rel_target,
                                    relation=rel_type,
                                    confidence=rel_conf,
                                )
                            )
                if relationships:
                    kg_evidences.append(
                        KnowledgeGraphEvidence(
                            disease_info=disease_name.strip(),
                            relationships=relationships,
                            confidence_score=_normalize_confidence_score(conf, default=0.85),
                        )
                    )

    # Build RAG Evidence if literature items were extracted
    rag_evidences: List[RAGEvidence] = []
    if retrieved_literature:
        avg_sim = round(sum(similarity_scores) / len(similarity_scores), 2) if similarity_scores else 0.85
        avg_conf = round(sum(confidence_scores) / len(confidence_scores), 2) if confidence_scores else 0.85
        rag_evidences.append(
            RAGEvidence(
                retrieved_literature=retrieved_literature,
                similarity_score=avg_sim,
                confidence_score=avg_conf,
                supporting_research_info=[f"Retrieved from Agent 2 query '{data.get('query', 'N/A')}'"] if data.get("query") else None,
            )
        )

    # Missing optional field default handling
    if not rag_evidences and not kg_evidences:
        logger.warning("No RAG text literature or Knowledge Graph relationships found in Agent 2 payload. Passing empty data.")

    # Combine metadata
    metadata: Dict[str, Any] = {}
    if isinstance(raw_input.get("metadata"), dict):
        metadata.update(raw_input["metadata"])
    if isinstance(data.get("retrieval_metadata"), dict):
        metadata["retrieval_metadata"] = data["retrieval_metadata"]
    if raw_input.get("message"):
        metadata["upstream_message"] = raw_input["message"]
    if data.get("query"):
        metadata["query"] = data["query"]

    logger.info(
        f"Agent 2 output adapted successfully. RAG Entries: {len(rag_evidences)}, "
        f"KG Entries: {len(kg_evidences)}, Literature Snippets: {len(retrieved_literature)}."
    )

    return Agent2Output(
        rag_evidence=rag_evidences,
        kg_evidence=kg_evidences,
        metadata=metadata if metadata else None,
    )


def adapt_agent3_output(raw_input: Union[Dict[str, Any], Any]) -> Agent3Output:
    """Adapt raw Agent 3 output payload into Agent 4 internal Agent3Output model.

    Supports:
      1. Direct Agent3Output instances or internal dict structure (medical_evidence key).
      2. RawAgent3Payload Pydantic model or dict representation (entities_queried, evidence[], summary).

    Raises:
        ValueError: If raw_input is malformed or missing critical structure.
    """
    if isinstance(raw_input, Agent3Output):
        logger.debug("Agent 3 input is already a normalized Agent3Output instance.")
        return raw_input

    if hasattr(raw_input, "model_dump"):
        raw_input = raw_input.model_dump()

    if not isinstance(raw_input, dict):
        error_msg = f"Invalid Agent 3 raw input type: expected dict or RawAgent3Payload, got {type(raw_input).__name__}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Executing Agent 3 payload adapter...")

    # Direct internal model dictionary format check
    if "medical_evidence" in raw_input:
        try:
            logger.info("Agent 3 raw payload matches internal schema. Validating directly.")
            return Agent3Output.model_validate(raw_input)
        except ValidationError as err:
            logger.warning(f"Direct validation of Agent 3 payload failed: {err}. Attempting flexible adaptation.")

    # Process Real Agent 3 Raw Payload Format
    evidence_items = raw_input.get("evidence", [])
    summary_text = raw_input.get("summary") or raw_input.get("latest_evidence") or ""

    pubmed_articles: List[PubMedArticle] = []
    europe_pmc_articles: List[EuropePMCArticle] = []
    clinical_guidelines: List[str] = []
    abstracts: List[str] = []

    if isinstance(evidence_items, list):
        for idx, item in enumerate(evidence_items):
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict item in Agent 3 evidence array at index {idx}.")
                continue

            src = str(item.get("source", "")).lower()
            title = str(item.get("title", "")).strip()
            abstract = str(item.get("abstract", "")).strip()
            url = str(item.get("url", "")).strip() if item.get("url") else None
            pub_year = item.get("publication_year") or item.get("year")
            try:
                pub_year = int(pub_year) if pub_year is not None else None
            except (ValueError, TypeError):
                pub_year = None

            item_id = str(item.get("id") or item.get("pmid") or item.get("pmcid") or "").strip() or None

            if "pubmed" in src:
                pubmed_articles.append(
                    PubMedArticle(
                        title=title or "PubMed Article",
                        pmid=item_id,
                        authors=item.get("authors") if isinstance(item.get("authors"), list) else None,
                        publication_year=pub_year,
                        journal=item.get("journal"),
                        url=url,
                    )
                )
            elif "europe" in src or "pmc" in src:
                europe_pmc_articles.append(
                    EuropePMCArticle(
                        title=title or "Europe PMC Article",
                        pmcid=item_id,
                        publication_year=pub_year,
                        url=url,
                    )
                )
            elif "guideline" in src:
                if title:
                    clinical_guidelines.append(title)
                if abstract and abstract != title:
                    clinical_guidelines.append(abstract)
            else:
                if title:
                    abstracts.append(title)
                if abstract and abstract != title:
                    abstracts.append(abstract)

    if not summary_text:
        if clinical_guidelines:
            summary_text = clinical_guidelines[0]
        elif abstracts:
            summary_text = abstracts[0]
        elif pubmed_articles:
            summary_text = pubmed_articles[0].title
        else:
            summary_text = "Latest evidence scanner response consolidated."
            logger.warning("Missing 'summary' field in Agent 3 payload. Applied fallback summary string.")

    source_info = {
        "entities_queried": raw_input.get("entities_queried", []),
        "sources_queried": raw_input.get("sources_queried", []),
        "fallback_used": raw_input.get("fallback_used", False),
    }

    metadata = raw_input.get("metadata") if isinstance(raw_input.get("metadata"), dict) else None

    medical_evidence_list = [
        MedicalEvidence(
            latest_evidence=summary_text.strip(),
            pubmed_articles=pubmed_articles,
            europe_pmc_articles=europe_pmc_articles,
            clinical_guidelines=clinical_guidelines,
            doi=str(raw_input.get("doi", "")).strip() if raw_input.get("doi") else None,
            abstracts=abstracts,
            source_info=source_info,
        )
    ]

    logger.info(
        f"Agent 3 output adapted successfully. PubMed: {len(pubmed_articles)}, "
        f"Europe PMC: {len(europe_pmc_articles)}, Guidelines: {len(clinical_guidelines)}, Abstracts: {len(abstracts)}."
    )

    return Agent3Output(
        medical_evidence=medical_evidence_list,
        metadata=metadata,
    )
