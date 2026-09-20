"""Pydantic schemas for the Clinical Data Fusion API."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class Relationship(BaseModel):
    """Represents a structured clinical relationship from a biomedical knowledge graph."""

    model_config = ConfigDict(populate_by_name=True)

    source: str = Field(..., description="Originating entity or disease concept.")
    target: str = Field(..., description="Target clinical concept, symptom, drug, or test.")
    relation: Optional[str] = Field(
        default=None,
        alias="type",
        description="Relationship predicate or interaction type.",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score for the relationship extraction.",
        ge=0.0,
        le=1.0,
    )


class PubMedArticle(BaseModel):
    """Represents peer-reviewed biomedical research publication details from PubMed."""

    title: str = Field(..., description="Article title.")
    pmid: Optional[str] = Field(default=None, description="PubMed unique identifier.")
    authors: Optional[List[str]] = Field(default=None, description="List of article authors.")
    publication_year: Optional[int] = Field(default=None, description="Year of publication.")
    journal: Optional[str] = Field(default=None, description="Publishing journal name.")
    url: Optional[str] = Field(default=None, description="Web URL or article permalink.")


class EuropePMCArticle(BaseModel):
    """Represents medical research publication details from Europe PMC."""

    title: str = Field(..., description="Article title.")
    pmcid: Optional[str] = Field(default=None, description="Europe PMC article identifier.")
    publication_year: Optional[int] = Field(default=None, description="Year of publication.")
    url: Optional[str] = Field(default=None, description="Web URL or article permalink.")


class RAGEvidence(BaseModel):
    """Represents retrieval-augmented generation evidence from biomedical literature."""

    retrieved_literature: List[str] = Field(
        ...,
        description="List of relevant clinical and scientific text pieces supporting the clinical query.",
    )
    similarity_score: float = Field(
        ...,
        description="Indicates how closely the retrieved literature matches the clinical query.",
        ge=0.0,
        le=1.0,
    )
    confidence_score: float = Field(
        ...,
        description="Reflects the reliability or certainty of the retrieval result.",
        ge=0.0,
        le=1.0,
    )
    supporting_research_info: Optional[List[str]] = Field(
        default=None,
        description="Additional supporting research info or citations that strengthen the interpretation.",
    )


class KnowledgeGraphEvidence(BaseModel):
    """Represents structured clinical relationships from a biomedical knowledge graph."""

    disease_info: str = Field(
        ...,
        description="Identifies the suspected or related disease concepts being investigated.",
    )
    relationships: List[Relationship] = Field(
        ...,
        description="Captured structured relationships between diseases, symptoms, drugs, and clinical concepts.",
    )
    confidence_score: float = Field(
        ...,
        description="Reflects the reliability or certainty of the knowledge graph relationship extraction.",
        ge=0.0,
        le=1.0,
    )


class Agent2Output(BaseModel):
    """Represents the structured biomedical retrieval output from Agent 2."""

    rag_evidence: List[RAGEvidence] = Field(
        ...,
        description="Evidence retrieved from biomedical literature databases.",
    )
    kg_evidence: List[KnowledgeGraphEvidence] = Field(
        ...,
        description="Evidence structured as knowledge graph relationships.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for tracking and debugging.",
    )


class MedicalEvidence(BaseModel):
    """Represents recent clinical evidence collected from guidelines and publications."""

    latest_evidence: str = Field(
        ...,
        description="The most recent clinically relevant findings for the condition under review.",
    )
    pubmed_articles: List[PubMedArticle] = Field(
        ...,
        description="List of peer-reviewed biomedical research and study details from PubMed.",
    )
    europe_pmc_articles: List[EuropePMCArticle] = Field(
        ...,
        description="List of medical research details from Europe PMC.",
    )
    clinical_guidelines: List[str] = Field(
        ...,
        description="Practice-oriented recommendations and consensus-based clinical directions.",
    )
    doi: Optional[str] = Field(
        default=None,
        description="Digital Object Identifier for the primary source publication.",
    )
    abstracts: List[str] = Field(
        ...,
        description="Abstracts summarizing the essential findings of the publications.",
    )
    source_info: Dict[str, Any] = Field(
        ...,
        description="Indicates where the evidence originated and how it should be interpreted.",
    )


class Agent3Output(BaseModel):
    """Represents the latest medical evidence output from Agent 3."""

    medical_evidence: List[MedicalEvidence] = Field(
        ...,
        description="Latest medical evidence gathered from recent publications and guidelines.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for tracking and debugging.",
    )


class MergedFinding(BaseModel):
    """Represents a consolidated clinical finding extracted across evidence sources."""

    finding: str = Field(..., description="Normalized clinical finding statement.")
    sources: List[str] = Field(..., description="List of evidence sources contributing this finding.")


class SupportingEvidence(BaseModel):
    """Represents a supporting clinical evidence entry with explicit source attribution."""

    finding: str = Field(..., description="Clinical finding or concept statement.")
    source_attribution: str = Field(..., description="Attributed evidence source origin.")


class ConflictEvidence(BaseModel):
    """Represents a contradictory or conflicting finding detected across evidence sources."""

    type: str = Field(..., description="Classification of evidence conflict (e.g., 'Treatment Conflict').")
    finding: str = Field(..., description="Name or summary of the conflicting concept.")
    description: str = Field(..., description="Detailed narrative explaining the opposing statements.")


class EvidencePriority(BaseModel):
    """Represents a clinical finding ranked according to source priority weights."""

    finding: str = Field(..., description="Clinical finding string.")
    priority_score: float = Field(..., description="Calculated priority score based on source weighting.")
    primary_source: str = Field(..., description="Primary originating evidence source.")


class SourceTraceability(BaseModel):
    """Maps a fused finding back to its originating upstream agents and sources."""

    finding: str = Field(..., description="Fused clinical finding string.")
    sources: List[str] = Field(..., description="List of specific evidence source names.")
    upstream_agents: List[str] = Field(..., description="Upstream agent identifiers (e.g., Agent 2, Agent 3).")


class UnifiedClinicalContext(BaseModel):
    """Represents the unified, sanitized, and merged clinical data context produced by Agent 4."""

    merged_findings: List[MergedFinding] = Field(
        ...,
        description="Consolidated clinical findings extracted and aggregated from all evidence sources.",
    )
    normalized_medical_terms: List[str] = Field(
        ...,
        description="Medical concepts and terms standardized by the clinical data fusion engine.",
    )
    supporting_evidence: List[SupportingEvidence] = Field(
        ...,
        description="Supporting evidence entries with source attribution.",
    )
    conflicting_evidence: List[ConflictEvidence] = Field(
        ...,
        description="Any contradictory or conflicting findings detected across evidence sources.",
    )
    evidence_priority: List[EvidencePriority] = Field(
        ...,
        description="Clinical findings categorized and ordered according to evidence priority.",
    )
    source_traceability: List[SourceTraceability] = Field(
        ...,
        description="Traceability mapping linking each fused finding back to Agent 2 or Agent 3.",
    )
    fusion_summary: str = Field(
        ...,
        description="Concise semantic summary produced by Llama semantic fusion or rule engine fallback.",
    )
    confidence_score: float = Field(
        ...,
        description="Overall confidence score in the fused clinical output.",
        ge=0.0,
        le=1.0,
    )


# --- Raw Upstream Agent Request Models (Part A - Swagger Display Schemas) ---

class RawAgent2ResultItem(BaseModel):
    """Raw result entry from Agent 2 Biomedical RAG and Knowledge Graph search."""

    text: Optional[str] = Field(default=None, description="Retrieved literature text or summary snippet.")
    content: Optional[str] = Field(default=None, description="Alternative field for retrieved text content.")
    score: Optional[float] = Field(default=None, description="Vector similarity or retrieval score.")
    similarity: Optional[float] = Field(default=None, description="Alternative field for similarity score.")
    confidence: Optional[float] = Field(default=None, description="Entity or relationship extraction confidence score.")
    source: Optional[str] = Field(default=None, description="Upstream database or vector store source.")
    document_type: Optional[str] = Field(default=None, description="Type of document ('literature' or 'knowledge_graph').")
    title: Optional[str] = Field(default=None, description="Document title or medical concept name.")
    disease: Optional[str] = Field(default=None, description="Primary disease or medical entity.")
    graph_context: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of knowledge graph relationship triples.")
    relationships: Optional[List[Dict[str, Any]]] = Field(default=None, description="Alternative field for graph relationships.")


class RawAgent2Data(BaseModel):
    """Data container object in Agent 2 output payload."""

    query: Optional[str] = Field(default=None, description="Clinical query submitted to Agent 2.")
    results: List[RawAgent2ResultItem] = Field(default_factory=list, description="Array of search and graph retrieval results.")
    retrieval_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Retrieval timing and metadata.")


class RawAgent2Payload(BaseModel):
    """Raw response payload produced by Agent 2 (Biomedical RAG + Knowledge Graph)."""

    success: Optional[bool] = Field(default=True, description="Success flag from Agent 2.")
    message: Optional[str] = Field(default=None, description="Status message from Agent 2.")
    data: Optional[RawAgent2Data] = Field(default=None, description="Structured retrieval data object.")
    rag_evidence: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional direct RAG evidence list for internal compatibility.")
    kg_evidence: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional direct KG evidence list for internal compatibility.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata dictionary.")


class RawAgent3EvidenceItem(BaseModel):
    """Raw evidence entry produced by Agent 3 Evidence Scanner."""

    source: Optional[str] = Field(default=None, description="Source database (e.g. 'PubMed', 'EuropePMC', 'Clinical Guidelines').")
    title: Optional[str] = Field(default=None, description="Article or guideline title.")
    abstract: Optional[str] = Field(default=None, description="Abstract or evidence summary text.")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier.")
    id: Optional[Union[str, int]] = Field(default=None, description="Article identifier (PMID or PMCID).")
    pmid: Optional[Union[str, int]] = Field(default=None, description="PubMed ID.")
    pmcid: Optional[Union[str, int]] = Field(default=None, description="Europe PMC ID.")
    query_matched: Optional[str] = Field(default=None, description="Query concept matched.")
    recency_sorted: Optional[bool] = Field(default=None, description="Recency sorting flag.")
    url: Optional[str] = Field(default=None, description="Publication permalink URL.")
    publication_year: Optional[int] = Field(default=None, description="Publication year.")
    journal: Optional[str] = Field(default=None, description="Journal name.")
    authors: Optional[List[str]] = Field(default=None, description="List of authors.")


class RawAgent3Payload(BaseModel):
    """Raw response payload produced by Agent 3 (Evidence Scanner)."""

    entities_queried: Optional[List[str]] = Field(default_factory=list, description="Medical entities searched.")
    sources_queried: Optional[List[str]] = Field(default=None, description="Upstream sources queried.")
    fallback_used: Optional[bool] = Field(default=False, description="Flag indicating if Agent 3 used fallback search.")
    summary: Optional[str] = Field(default=None, description="Consolidated latest evidence summary.")
    latest_evidence: Optional[str] = Field(default=None, description="Alternative field for summary text.")
    evidence: List[RawAgent3EvidenceItem] = Field(default_factory=list, description="Array of evidence articles and guidelines.")
    medical_evidence: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional direct medical evidence list for internal compatibility.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata dictionary.")


class FusionRequest(BaseModel):
    """Represents the incoming request payload containing RAW outputs from Agent 2 and Agent 3."""

    agent2_output: Union[RawAgent2Payload, Agent2Output, Dict[str, Any]] = Field(
        ...,
        description="Raw output payload from Agent 2 (Biomedical RAG + Knowledge Graph).",
    )
    agent3_output: Union[RawAgent3Payload, Agent3Output, Dict[str, Any]] = Field(
        ...,
        description="Raw output payload from Agent 3 (Evidence Scanner).",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "agent2_output": {
                    "success": True,
                    "message": "Biomedical RAG and Knowledge Graph evidence retrieved successfully.",
                    "data": {
                        "query": "ST-elevation Myocardial Infarction treatment",
                        "results": [
                            {
                                "text": "ST-elevation myocardial infarction (STEMI) requires immediate emergency reperfusion therapy via primary PCI or fibrinolysis.",
                                "score": 0.94,
                                "confidence": 0.91,
                                "source": "Biomedical PubMed Vector Store",
                                "document_type": "literature"
                            },
                            {
                                "disease": "Acute Myocardial Infarction",
                                "document_type": "knowledge_graph",
                                "confidence": 0.89,
                                "graph_context": [
                                    {"source": "Acute Myocardial Infarction", "target": "Shortness of Breath", "type": "associated_symptom"},
                                    {"source": "Acute Myocardial Infarction", "target": "Aspirin", "type": "treatment_option"}
                                ]
                            }
                        ],
                        "retrieval_metadata": {"total_results": 2, "latency_ms": 14.5}
                    },
                    "metadata": {"source_agent": "Agent 2"}
                },
                "agent3_output": {
                    "entities_queried": ["Acute Myocardial Infarction", "STEMI"],
                    "sources_queried": ["PubMed", "EuropePMC", "ClinicalGuidelines"],
                    "fallback_used": False,
                    "summary": "Administer Aspirin 325 mg orally immediately. Perform 12-lead ECG within 10 minutes of presentation.",
                    "evidence": [
                        {
                            "source": "PubMed",
                            "title": "Dual antiplatelet therapy timing in STEMI",
                            "abstract": "Early administration of aspirin 325 mg reduces acute 30-day mortality in STEMI.",
                            "doi": "10.1016/j.jacc.2025.01.001",
                            "id": "38192011",
                            "publication_year": 2025,
                            "url": "https://pubmed.ncbi.nlm.nih.gov/38192011/"
                        },
                        {
                            "source": "Clinical Guidelines",
                            "title": "A 12-lead Electrocardiogram (ECG) must be performed immediately for chest pain.",
                            "abstract": "Emergency ECG protocol for acute chest pain evaluation."
                        }
                    ],
                    "metadata": {"source_agent": "Agent 3"}
                }
            }
        }
    )


class FusionMetadata(BaseModel):
    """Structured execution metadata for tracking, auditing, and debugging data fusion requests."""

    correlation_id: str = Field(..., description="Unique request correlation ID for end-to-end tracing.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of execution.")
    processing_time_ms: float = Field(..., description="Total pipeline execution time in milliseconds.")
    fusion_mode: str = Field(..., description="Fusion mode applied ('llama_enhanced' or 'rule_based_fallback').")
    fusion_method: str = Field(..., description="Method identifier for backward compatibility.")
    llm_provider: str = Field(..., description="Configured LLM provider name.")
    llm_model: str = Field(..., description="Configured LLM model name.")
    llm_used: bool = Field(..., description="Indicates whether LLM enhancement was successfully executed.")
    fallback_used: bool = Field(..., description="Indicates whether rule-based fallback was triggered.")
    fallback_reason: Optional[str] = Field(default=None, description="Detailed reason for automatic fallback if triggered.")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics if LLM was called.")
    latency_ms: Optional[float] = Field(default=None, description="LLM call latency in milliseconds if executed.")
    execution_stats: Optional[Dict[str, Any]] = Field(default=None, description="Additional execution stats.")
    conflicts_detected: Optional[List[ConflictEvidence]] = Field(default=None, description="List of detected evidence conflicts.")
    findings_sources: Optional[Dict[str, List[str]]] = Field(default=None, description="Map of findings to source origins.")


class FusionResponse(BaseModel):
    """Represents the API response containing the fused clinical context and structured metadata."""

    status: str = Field(
        ...,
        description="Execution status of the fusion pipeline (e.g., 'success', 'partial_success', 'failed').",
    )
    message: str = Field(
        ...,
        description="Human-readable message regarding fusion processing results.",
    )
    unified_context: UnifiedClinicalContext = Field(
        ...,
        description="The unified clinical data context produced by Clinical Data Fusion.",
    )
    metadata: FusionMetadata = Field(
        ...,
        description="Structured execution metadata.",
    )


class ErrorResponse(BaseModel):
    """Standardized API response for error cases."""

    detail: str = Field(..., description="A clear, human-readable error message explaining the failure.")
    correlation_id: str = Field(..., description="The unique request correlation ID for tracing.")


class FusionInput(BaseModel):
    """Deprecated: Legacy input schema preserved for backward compatibility."""

    rag_output: str = Field(..., description="Legacy RAG output field.")
    kg_output: str = Field(..., description="Legacy KG output field.")
    evidence_output: str = Field(..., description="Legacy Evidence output field.")