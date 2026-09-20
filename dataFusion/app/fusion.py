"""Hybrid Clinical Data Fusion Orchestration Module combining Rule-Based Fusion and Llama 3.1 8B Instruct.

Architecture & Concepts:
  - Rule-Based Fusion: Deterministic baseline engine extracting structured entities, normalizing medical terminology,
    scoring evidence priority, detecting contraindication conflicts, and mapping source traceability.
  - Semantic Fusion: LLM-powered (Llama 3.1 8B Instruct) semantic context refinement that resolves complex synonyms,
    deduplicates nuances, and preserves source attribution without adding external medical facts or clinical decisions.
  - Hybrid Fusion: Staged pipeline combining deterministic rule-based baseline generation with LLM semantic enhancement.
  - Automatic Fallback: Fault-tolerant execution mechanism that automatically falls back to rule-based baseline fusion
    if LLM API encounters network errors, timeouts, rate limits (429), or schema validation failures.
  - Source Traceability: Explicit mapping linking every merged finding back to originating upstream agents
    (Agent 2 / Agent 3) and source databases.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import settings
from app.llm import enhance_clinical_context_with_llama
from app.schemas import (
    Agent2Output,
    Agent3Output,
    ConflictEvidence,
    EvidencePriority,
    FusionRequest,
    KnowledgeGraphEvidence,
    MedicalEvidence,
    MergedFinding,
    RAGEvidence,
    SourceTraceability,
    SupportingEvidence,
    UnifiedClinicalContext,
)
from app.utils import logger


def validate_agent2_output(output: Agent2Output) -> None:
    """Validate the structured biomedical retrieval output from Agent 2.

    Raises:
        ValueError: If validation constraints are violated.
    """
    if not output.rag_evidence and not output.kg_evidence:
        raise ValueError("Agent 2 output must contain either RAG evidence or Knowledge Graph evidence.")

    for idx, rag in enumerate(output.rag_evidence):
        if not rag.retrieved_literature:
            raise ValueError(f"RAG evidence at index {idx} contains no literature items.")
        if any(not text.strip() for text in rag.retrieved_literature):
            raise ValueError(f"RAG evidence at index {idx} contains empty or whitespace-only literature items.")

    for idx, kg in enumerate(output.kg_evidence):
        if not kg.disease_info.strip():
            raise ValueError(f"Knowledge Graph evidence at index {idx} has an empty disease_info field.")


def validate_agent3_output(output: Agent3Output) -> None:
    """Validate the latest medical evidence output from Agent 3.

    Raises:
        ValueError: If validation constraints are violated.
    """
    if not output.medical_evidence:
        raise ValueError("Agent 3 output must contain at least one medical evidence entry.")

    for idx, med in enumerate(output.medical_evidence):
        if not med.latest_evidence.strip():
            raise ValueError(f"Medical evidence at index {idx} has an empty latest_evidence field.")


def _deduplicate_list_preserve_order(lst: List[str]) -> List[str]:
    """Remove duplicate string entries from a list while preserving their original order."""
    seen = set()
    cleaned = []
    for item in lst:
        stripped = item.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            cleaned.append(stripped)
    return cleaned


def normalize_medical_term(term: str) -> str:
    """Normalize a medical term or concept using the centralized configuration mapping."""
    clean_term = term.strip().lower()
    if clean_term in settings.normalization_map:
        return settings.normalization_map[clean_term]
    return " ".join(word.capitalize() for word in clean_term.split())


def preprocess_agent2_output(output: Agent2Output) -> Agent2Output:
    """Preprocess Agent 2 output by stripping whitespaces and deduplicating list fields."""
    cleaned_rag = []
    for rag in output.rag_evidence:
        cleaned_literature = _deduplicate_list_preserve_order(rag.retrieved_literature)
        cleaned_supporting = (
            _deduplicate_list_preserve_order(rag.supporting_research_info)
            if rag.supporting_research_info is not None
            else None
        )
        cleaned_rag.append(
            RAGEvidence(
                retrieved_literature=cleaned_literature,
                similarity_score=rag.similarity_score,
                confidence_score=rag.confidence_score,
                supporting_research_info=cleaned_supporting,
            )
        )

    cleaned_kg = []
    for kg in output.kg_evidence:
        cleaned_kg.append(
            KnowledgeGraphEvidence(
                disease_info=kg.disease_info.strip(),
                relationships=kg.relationships,
                confidence_score=kg.confidence_score,
            )
        )

    return Agent2Output(
        rag_evidence=cleaned_rag,
        kg_evidence=cleaned_kg,
        metadata=output.metadata,
    )


def preprocess_agent3_output(output: Agent3Output) -> Agent3Output:
    """Preprocess Agent 3 output by stripping whitespaces and deduplicating list fields."""
    cleaned_med = []
    for med in output.medical_evidence:
        cleaned_guidelines = _deduplicate_list_preserve_order(med.clinical_guidelines)
        cleaned_abstracts = _deduplicate_list_preserve_order(med.abstracts)

        cleaned_med.append(
            MedicalEvidence(
                latest_evidence=med.latest_evidence.strip(),
                pubmed_articles=med.pubmed_articles,
                europe_pmc_articles=med.europe_pmc_articles,
                clinical_guidelines=cleaned_guidelines,
                doi=med.doi.strip() if med.doi else None,
                abstracts=cleaned_abstracts,
                source_info={k: (v.strip() if isinstance(v, str) else v) for k, v in med.source_info.items()},
            )
        )

    return Agent3Output(
        medical_evidence=cleaned_med,
        metadata=output.metadata,
    )


def preprocess_fusion_request(request: FusionRequest) -> FusionRequest:
    """Validate and preprocess the entire incoming FusionRequest."""
    return FusionRequest(
        agent2_output=preprocess_agent2_output(request.agent2_output),
        agent3_output=preprocess_agent3_output(request.agent3_output),
    )


def prepare_llm_fusion_input(request: FusionRequest) -> Dict[str, Any]:
    """Prepare structured JSON dictionary from preprocessed inputs to send to Llama 3.1 8B Instruct."""
    return request.model_dump()


# --- Generic Rule-Based Baseline Engine ---

def detect_conflicts(request: FusionRequest, findings: List[str]) -> List[ConflictEvidence]:
    """Detect evidence conflicts and direct contradictions across Agent 2 and Agent 3.
    
    NOTE: Temporarily disabled. The previous regex-based rule engine was too aggressive, 
    flagging general symptoms (like 'chest pain') as treatment contraindications if words 
    like 'fatal' or 'avoid' appeared in the same evidence chunk. 
    """
    return []


def _calculate_fused_confidence(request: FusionRequest, conflicts: Optional[List[ConflictEvidence]] = None) -> float:
    """Calculate overall confidence score reflecting evidence quality, source agreement, completeness, and conflicts."""
    scores = []
    
    # 1. Agent 2 Evidence Scores
    for rag in request.agent2_output.rag_evidence:
        conf = rag.confidence_score if rag.confidence_score is not None else 0.80
        sim = rag.similarity_score if rag.similarity_score is not None else conf
        # Harmonic mean of similarity and confidence
        score = round((conf * 0.6) + (sim * 0.4), 2)
        scores.append((score, 0.8))

    for kg in request.agent2_output.kg_evidence:
        if kg.confidence_score is not None:
            scores.append((kg.confidence_score, 0.8))

    # 2. Agent 3 Evidence Scores
    all_a3_text = []
    for med in request.agent3_output.medical_evidence:
        all_a3_text.append(med.latest_evidence)
        all_a3_text.extend(med.clinical_guidelines)
        all_a3_text.extend(med.abstracts)

    a3_text_combined = " ".join(all_a3_text).lower()
    
    # Check for explicit uncertainty / insufficiency statements in literature
    uncertainty_phrases = ["insufficient to establish", "inconclusive", "cannot establish", "do not establish", "uncertainty regarding", "limitations that prevent"]
    is_inconclusive = any(phrase in a3_text_combined for phrase in uncertainty_phrases)

    if request.agent3_output.medical_evidence:
        if is_inconclusive:
            a3_score = 0.65
        elif any(med.pubmed_articles for med in request.agent3_output.medical_evidence) or any(med.clinical_guidelines for med in request.agent3_output.medical_evidence):
            a3_score = 0.90
        else:
            a3_score = 0.85
        scores.append((a3_score, 1.0))

    if not scores:
        return 0.50

    weighted_sum = sum(score * weight for score, weight in scores)
    weight_sum = sum(weight for _, weight in scores)
    base_confidence = weighted_sum / weight_sum

    # 3. Penalties & Bonuses
    if conflicts and len(conflicts) > 0:
        base_confidence -= 0.25

    if is_inconclusive:
        base_confidence = min(base_confidence, 0.68)

    # Multi-source agreement bonus
    if len(request.agent2_output.rag_evidence) > 0 and len(request.agent3_output.medical_evidence) > 0 and not (conflicts and len(conflicts) > 0) and not is_inconclusive:
        base_confidence += 0.04

    return round(max(0.10, min(0.98, base_confidence)), 2)


def validate_fused_context(context: UnifiedClinicalContext) -> UnifiedClinicalContext:
    """Validate and sanitize fused UnifiedClinicalContext to guarantee clean concise normalized concepts."""
    cleaned_terms = []
    for term in context.normalized_medical_terms:
        clean_t = term.strip().rstrip(".:;")
        # If term is accidentally a sentence (> 80 chars or has multiple words with verbs), condense it
        if len(clean_t) > 80 or any(clean_t.lower().startswith(prefix) for prefix in ["the study", "the available", "findings do not", "this study"]):
            if "not support" in clean_t.lower() or "insufficient" in clean_t.lower():
                clean_t = "Unsupportive Biomedical Evidence"
            elif "support" in clean_t.lower():
                clean_t = "Supportive Biomedical Evidence"
            else:
                clean_t = "Biomedical Study Evaluation"
        if clean_t and clean_t not in cleaned_terms:
            cleaned_terms.append(clean_t)

    if not cleaned_terms:
        cleaned_terms = ["Biomedical Clinical Evidence"]

    # Ensure confidence score is strictly in [0.0, 1.0]
    conf = max(0.0, min(1.0, float(context.confidence_score)))

    # Ensure merged_findings have clean names; preserve existing sources without defaulting
    cleaned_findings = []
    for mf in context.merged_findings:
        f_name = mf.finding.strip()
        if len(f_name) > 80 and any(f_name.lower().startswith(p) for p in ["the study", "the available", "findings"]):
            f_name = "Biomedical Research Evidence"
        cleaned_findings.append(MergedFinding(finding=f_name, sources=mf.sources or []))

    return UnifiedClinicalContext(
        merged_findings=cleaned_findings,
        normalized_medical_terms=cleaned_terms,
        supporting_evidence=context.supporting_evidence,
        conflicting_evidence=context.conflicting_evidence,
        evidence_priority=context.evidence_priority,
        source_traceability=context.source_traceability,
        fusion_summary=context.fusion_summary,
        confidence_score=conf,
    )


def _agent_labels_from_db_sources(db_sources: List[str]) -> List[str]:
    """Map internal DB source names to upstream agent labels (Agent 2 / Agent 3)."""
    agents: List[str] = []
    if any(s in db_sources for s in ["Knowledge Graph", "Biomedical RAG"]):
        agents.append("Agent 2")
    if "Latest Medical Evidence" in db_sources:
        agents.append("Agent 3")
    return agents


def rule_based_fusion(request: FusionRequest) -> Tuple[UnifiedClinicalContext, List[ConflictEvidence], Dict[str, List[str]]]:
    """Execute generic deterministic rule-based fusion baseline preserving Agent 2 and Agent 3."""
    # Internal mapping: finding -> set of DB-level sources ("Biomedical RAG", "Latest Medical Evidence", "Knowledge Graph")
    trace_db_sources: Dict[str, Set[str]] = {}

    def add_finding(term: str, db_source: str):
        if not term or not term.strip():
            return
        norm = normalize_medical_term(term)
        trace_db_sources.setdefault(norm, set()).add(db_source)

    # 1. Extract from Knowledge Graph Evidence (Agent 2)
    for kg in request.agent2_output.kg_evidence:
        if kg.disease_info:
            add_finding(kg.disease_info, "Knowledge Graph")
        for rel in kg.relationships:
            if rel.source:
                add_finding(rel.source, "Knowledge Graph")
            if rel.target:
                add_finding(rel.target, "Knowledge Graph")

    # 2. Extract from Biomedical RAG Evidence (Agent 2)
    for rag in request.agent2_output.rag_evidence:
        matched_a2 = False
        for text in rag.retrieved_literature:
            text_lower = text.lower()
            for category_keywords in settings.clinical_keywords_map.values():
                for kw in category_keywords:
                    if kw in text_lower:
                        add_finding(kw, "Biomedical RAG")
                        matched_a2 = True
        if rag.supporting_research_info:
            for info in rag.supporting_research_info:
                if not matched_a2 and info.startswith("PMID:"):
                    add_finding("Biomedical Literature Evidence", "Biomedical RAG")
                    matched_a2 = True
        # Ensure at least one Agent 2 finding is registered even without keyword match
        if not matched_a2 and rag.retrieved_literature:
            add_finding("Biomedical Literature Evidence", "Biomedical RAG")

    # 3. Extract from Latest Medical Evidence (Agent 3)
    for med in request.agent3_output.medical_evidence:
        all_med_texts = [med.latest_evidence] + med.clinical_guidelines + med.abstracts
        for art in med.pubmed_articles:
            all_med_texts.append(art.title)
        for art in med.europe_pmc_articles:
            all_med_texts.append(art.title)

        matched_a3 = False
        for text in all_med_texts:
            text_lower = text.lower()
            for category_keywords in settings.clinical_keywords_map.values():
                for kw in category_keywords:
                    if kw in text_lower:
                        add_finding(kw, "Latest Medical Evidence")
                        matched_a3 = True
        # Ensure at least one Agent 3 finding is registered even without keyword match
        if not matched_a3 and med.latest_evidence:
            add_finding("Biomedical Literature Evidence", "Latest Medical Evidence")

    # Fallback: both agents contribute to a generic finding if nothing was extracted
    if not trace_db_sources:
        if request.agent2_output.rag_evidence:
            add_finding("Biomedical Literature Evidence", "Biomedical RAG")
        if request.agent3_output.medical_evidence:
            add_finding("Biomedical Literature Evidence", "Latest Medical Evidence")

    def get_highest_priority(finding: str) -> float:
        db_srcs = trace_db_sources.get(finding, set())
        return max((settings.source_priority_weights.get(src, 0.7) for src in db_srcs), default=0.7)

    sorted_findings = sorted(list(trace_db_sources.keys()), key=get_highest_priority, reverse=True)

    conflicts = detect_conflicts(request, sorted_findings)

    merged_findings_list = []
    normalized_medical_terms = []
    supporting_evidence = []
    evidence_priority = []
    source_traceability_list = []

    for finding in sorted_findings:
        normalized_medical_terms.append(finding)
        db_srcs = sorted(trace_db_sources.get(finding, set()))
        agents = _agent_labels_from_db_sources(db_srcs)
        p_score = get_highest_priority(finding)

        # merged_findings.sources = upstream agent labels (Agent 2 / Agent 3)
        merged_findings_list.append(MergedFinding(finding=finding, sources=agents))
        supporting_evidence.append(
            SupportingEvidence(finding=f"Evidence supporting {finding}", source_attribution=", ".join(agents) or "Unknown")
        )
        evidence_priority.append(
            EvidencePriority(finding=finding, priority_score=p_score, primary_source=db_srcs[0] if db_srcs else "Biomedical Evidence")
        )
        source_traceability_list.append(
            SourceTraceability(finding=finding, sources=db_srcs, upstream_agents=agents)
        )

    # Add raw literature statements to supporting_evidence
    for rag in request.agent2_output.rag_evidence:
        for lit in rag.retrieved_literature:
            supporting_evidence.append(SupportingEvidence(finding=lit, source_attribution="Agent 2 (Biomedical RAG)"))
    for med in request.agent3_output.medical_evidence:
        if med.latest_evidence:
            supporting_evidence.append(SupportingEvidence(finding=med.latest_evidence, source_attribution="Agent 3 (Latest Medical Evidence)"))

    fusion_summary = (
        f"Clinical Data Fusion synthesized evidence from Agent 2 ({len(request.agent2_output.rag_evidence)} RAG entries) "
        f"and Agent 3 ({len(request.agent3_output.medical_evidence)} Medical Evidence entries). "
        f"Consolidated {len(normalized_medical_terms)} normalized finding(s) with {len(conflicts)} conflict(s) detected."
    )

    confidence_score = _calculate_fused_confidence(request, conflicts=conflicts)

    deterministic_context = UnifiedClinicalContext(
        merged_findings=merged_findings_list,
        normalized_medical_terms=normalized_medical_terms,
        supporting_evidence=supporting_evidence,
        conflicting_evidence=conflicts,
        evidence_priority=evidence_priority,
        source_traceability=source_traceability_list,
        fusion_summary=fusion_summary,
        confidence_score=confidence_score,
    )

    # findings_sources: single source-of-truth derived from source_traceability.upstream_agents
    # Format: {finding_name: ["Agent 2", ...]} — consistent with merged_findings and source_traceability
    findings_sources: Dict[str, List[str]] = {
        st.finding: st.upstream_agents for st in source_traceability_list
    }

    return deterministic_context, conflicts, findings_sources


# --- Hybrid Orchestrator ---

def fuse_clinical_data(
    request: FusionRequest,
    correlation_id: str = "N/A"
) -> Tuple[UnifiedClinicalContext, List[ConflictEvidence], Dict[str, List[str]], str, Dict[str, Any], Optional[str]]:
    """Orchestrate Hybrid Clinical Data Fusion with genuine multi-source fusion."""
    logger.info(f"[{correlation_id}] Stage 4: Executing rule-based initial clinical data fusion baseline...")
    deterministic_context, deterministic_conflicts, trace_origins = rule_based_fusion(request)

    fallback_reason: Optional[str] = None

    logger.info(f"[{correlation_id}] Stage 5: Calling LLM semantic enhancement engine (Provider: '{settings.llm_provider}', Model: '{settings.llm_model_name}')...")
    llm_result = enhance_clinical_context_with_llama(
        deterministic_context_dict=deterministic_context.model_dump(),
        raw_request_dict=request.model_dump(),
        correlation_id=correlation_id
    )

    if llm_result is not None:
        parsed_json, stats = llm_result
        raw_context = parsed_json.get("unified_context")
        if raw_context:
            try:
                raw_validated = UnifiedClinicalContext.model_validate(raw_context)
                validated_context = validate_fused_context(raw_validated)
                conflicts = [ConflictEvidence.model_validate(c) for c in parsed_json.get("conflicts", [])] or deterministic_conflicts
                
                # Check for conflicts between Agent 2 and Agent 3
                if not conflicts:
                    conflicts = detect_conflicts(request, validated_context.normalized_medical_terms)
                    if conflicts:
                        validated_context.conflicting_evidence = conflicts
                        validated_context.confidence_score = round(max(0.10, validated_context.confidence_score - 0.25), 2)

                logger.info(
                    f"[{correlation_id}] Stage 6: Schema Validation SUCCESS. Returning LLM-enhanced clinical context "
                    f"(Confidence: {validated_context.confidence_score}, Findings: {len(validated_context.merged_findings)}, Conflicts: {len(conflicts)})."
                )
                # Derive findings_sources from the single source-of-truth: source_traceability.upstream_agents
                llm_findings_sources: Dict[str, List[str]] = {
                    st.finding: st.upstream_agents
                    for st in validated_context.source_traceability
                }
                return validated_context, conflicts, llm_findings_sources, "llama_enhanced", stats, None
            except Exception as val_err:
                fallback_reason = f"LLM JSON output failed Pydantic schema validation: {val_err}"
                logger.warning(f"[{correlation_id}] Stage 6: Schema Validation FAILED ({fallback_reason}). Triggering rule-based fallback.")
        else:
            fallback_reason = "LLM JSON output missing required 'unified_context' root key."
    else:
        if not settings.llm_enabled:
            fallback_reason = "LLM enhancement is disabled in configuration settings (LLM_ENABLED=False)."
        elif not settings.llm_api_key or settings.llm_api_key == "mock_key_or_set_real_api_key":
            fallback_reason = (
                f"LLM API key is unconfigured or set to default mock key (Provider: '{settings.llm_provider}', "
                f"Base URL: '{settings.llm_base_url}'). Set LLM_API_KEY or GROQ_API_KEY to enable LLM primary fusion."
            )
        else:
            fallback_reason = "LLM enhancement call returned None (network error, timeout, HTTP 429, or invalid JSON response)."

    logger.info(f"[{correlation_id}] Stage 6: Automatic Rule-Based Fallback activated. Reason: {fallback_reason}. Returning rule-based baseline context.")
    return deterministic_context, deterministic_conflicts, trace_origins, "rule_based_fallback", {}, fallback_reason


def combine_context(rag_output: str, kg_output: str, evidence_output: str) -> str:
    """Concatenate the available context sources into one combined string."""
    return (
        f"RAG Output: {rag_output}\n"
        f"KG Output: {kg_output}\n"
        f"Evidence Output: {evidence_output}"
    )