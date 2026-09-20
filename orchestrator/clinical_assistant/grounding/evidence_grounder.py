"""Evidence Grounding Layer for Clinical Conversational Assistant.

Integrates Biomedical Vector RAG (Agent 2) and Live Web Intelligence (Agent 3)
while strictly preserving evidence provenance:
- source_type: "biomedical_rag" | "web"
- source identifier, retrieval date / ID, section title, score
- clearly separates Patient Facts, Medical Evidence, and Hypotheses
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("orchestrator.clinical_assistant.grounding")


class GroundedEvidenceItem(BaseModel):
    """Represents a single piece of retrieved medical evidence with full provenance."""
    claim: str
    source_type: str  # "biomedical_rag" | "web"
    source: str
    retrieval_id: Optional[str] = None
    retrieval_date: Optional[str] = None
    confidence_score: float = 1.0
    section_title: Optional[str] = None
    url: Optional[str] = None


class GroundedEvidenceContext(BaseModel):
    """Combined evidence package for context assembly."""
    biomedical_rag_evidence: List[GroundedEvidenceItem] = Field(default_factory=list)
    web_evidence: List[GroundedEvidenceItem] = Field(default_factory=list)
    retrieval_summary: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def get_all_items(self) -> List[GroundedEvidenceItem]:
        return self.biomedical_rag_evidence + self.web_evidence

    def to_prompt_context(self) -> str:
        """Format grounded evidence into structured text with explicit source citations."""
        lines = ["[RETRIEVED MEDICAL EVIDENCE - EVIDENCE GROUNDING ONLY]"]
        
        if self.biomedical_rag_evidence:
            lines.append("--- Biomedical Vector RAG Evidence (Peer-Reviewed Literature) ---")
            for idx, item in enumerate(self.biomedical_rag_evidence, start=1):
                lines.append(
                    f"[{idx}] [source_type: biomedical_rag | source: {item.source} | id: {item.retrieval_id or 'N/A'}]"
                )
                lines.append(f"    Evidence: {item.claim}")
                if item.section_title:
                    lines.append(f"    Context: {item.section_title}")

        if self.web_evidence:
            lines.append("--- Live Web Intelligence (Clinical Guidelines & Treatment Evidence) ---")
            for idx, item in enumerate(self.web_evidence, start=len(self.biomedical_rag_evidence) + 1):
                lines.append(
                    f"[{idx}] [source_type: web | source: {item.source} | date: {item.retrieval_date or 'Recent'}]"
                )
                lines.append(f"    Guideline/Evidence: {item.claim}")

        if not self.biomedical_rag_evidence and not self.web_evidence:
            lines.append("No external medical literature retrieved for this turn.")

        return "\n".join(lines)


class EvidenceGrounder:
    """Adapts raw responses from Agent 2 and Agent 3 into grounded evidence items."""

    @classmethod
    def process_agent2_rag(cls, agent2_response: Optional[Dict[str, Any]]) -> List[GroundedEvidenceItem]:
        """Convert Agent 2 search results into grounded items."""
        if not agent2_response:
            return []

        items = []
        results = agent2_response.get("results", [])
        for idx, res in enumerate(results):
            text = res.get("text", "").strip()
            if not text:
                continue
            source_id = res.get("source_id", f"rag-doc-{idx+1}")
            source = res.get("journal") or res.get("source_type") or "Biomedical Literature DB"
            score = float(res.get("score", 0.8))
            items.append(
                GroundedEvidenceItem(
                    claim=text,
                    source_type="biomedical_rag",
                    source=source,
                    retrieval_id=source_id,
                    confidence_score=score,
                    section_title=res.get("section_title"),
                )
            )
        return items

    @classmethod
    def process_agent3_web(cls, agent3_response: Optional[Dict[str, Any]]) -> List[GroundedEvidenceItem]:
        """Convert Agent 3 scanner results into grounded web items."""
        if not agent3_response:
            return []

        items = []
        evidence_list = agent3_response.get("evidence", [])
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for idx, ev in enumerate(evidence_list):
            title = ev.get("title", "")
            snippet = ev.get("snippet", "")
            source = ev.get("source") or ev.get("url") or "Tavily Clinical Guidelines"
            claim_text = f"{title}: {snippet}".strip() if title else snippet.strip()
            if not claim_text:
                continue
            items.append(
                GroundedEvidenceItem(
                    claim=claim_text,
                    source_type="web",
                    source=source,
                    retrieval_date=now_date,
                    confidence_score=0.9,
                    url=ev.get("url"),
                )
            )
        return items
