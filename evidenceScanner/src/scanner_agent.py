"""
scanner_agent.py - Master Orchestrator for Agent 3 (Evidence-Based Scanner).
- PubMed: Queried for acute presentation & disease patterns (Symptoms + Hint only).
- Tavily: Queried for clinical management guidelines with comorbidity risk measures.
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.pubmed_client import fetch_pubmed_evidence_with_recency_and_fallback
from src.web_scraper import WebScraper
from src.condition_inferencer import extract_clinical_profile_and_mesh_queries


class EvidenceRequest(BaseModel):
    raw_text: Optional[str] = Field(default="", description="Patient clinical presentation")
    entities: List[str] = Field(default_factory=list, description="Acute symptoms")
    chronic_conditions: List[str] = Field(default_factory=list, description="Pre-existing chronic conditions")
    condition_hint: str = Field(default="", description="Suspected acute condition")
    max_results: int = Field(default=3, ge=1, le=10)


class ScannerResponse(BaseModel):
    input_text_processed: str
    acute_symptoms: List[str]
    chronic_conditions: List[str]
    inferred_condition_hint: str
    chronic_risk_measures: List[str]
    total_found: int
    sources_queried: List[str]
    fallback_used: bool
    evidence: List[Dict[str, Any]]
    summary: str


class EvidenceScannerAgent:
    def __init__(self):
        self.scraper = WebScraper()

    def normalize_title(self, title: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '', title.lower()) if title else ""

    def run(self, request: EvidenceRequest) -> ScannerResponse:
        combined_evidence: List[Dict[str, Any]] = []
        seen_titles = set()
        sources_used = []

        # 1. Profile Extraction
        input_text = request.raw_text or " ".join(request.entities + request.chronic_conditions)
        profile = extract_clinical_profile_and_mesh_queries(
            raw_text=input_text,
            acute_symptoms=request.entities,
            chronic_conditions=request.chronic_conditions,
            condition_hint=request.condition_hint
        )

        acute_symptoms = profile.get("acute_symptoms", []) or request.entities
        chronic_conditions = profile.get("chronic_conditions", []) or request.chronic_conditions
        inferred_condition = profile.get("inferred_condition", "") or request.condition_hint
        chronic_risks = profile.get("chronic_risk_measures", [])

        pubmed_mesh_q = profile.get("pubmed_mesh_query", "")
        tavily_q = profile.get("tavily_guideline_query", "")

        # 2. Dynamic Routing Execution
        has_chronic = len(chronic_conditions) > 0

        if has_chronic:
            # ROUTE A: Chronic Condition Present -> Tavily Scraper ONLY
            print("[Router] Chronic condition detected. Executing Tavily Guidelines ONLY...")
            tavily_res = self.scraper.crawl_and_extract(
                start_url="",
                topic=tavily_q or f"{inferred_condition} management with {', '.join(acute_symptoms)}",
                chronic_risks=chronic_risks
            )
            if tavily_res:
                norm_title = self.normalize_title(tavily_res.get("title", ""))
                if norm_title not in seen_titles:
                    seen_titles.add(norm_title)
                    combined_evidence.append(tavily_res)
                    sources_used.append("Tavily Guidelines & Treatment Safety")

        else:
            # ROUTE B: No Chronic Condition -> Query Both PubMed and Tavily
            print("[Router] No chronic condition detected. Querying PubMed and Tavily...")
            
            # A. PubMed
            print(f"[PubMed] Querying Acute/Hint: '{pubmed_mesh_q}'...")
            pubmed_results = fetch_pubmed_evidence_with_recency_and_fallback(
                entities=acute_symptoms,
                condition_hint=inferred_condition,
                target_count=request.max_results,
                strict_query=pubmed_mesh_q
            )
            for item in pubmed_results:
                norm_title = self.normalize_title(item.get("title", ""))
                if norm_title and norm_title not in seen_titles:
                    seen_titles.add(norm_title)
                    combined_evidence.append(item)
                    if "PubMed" not in sources_used:
                        sources_used.append("PubMed")

            # B. Tavily
            print("[Tavily / WebScraper] Querying General Guidelines...")
            tavily_res = self.scraper.crawl_and_extract(
                start_url="",
                topic=tavily_q or f"{inferred_condition} management with {', '.join(acute_symptoms)}",
                chronic_risks=[]
            )
            if tavily_res:
                norm_title = self.normalize_title(tavily_res.get("title", ""))
                if norm_title not in seen_titles:
                    seen_titles.add(norm_title)
                    combined_evidence.append(tavily_res)
                    sources_used.append("Tavily Guidelines & Treatment Safety")

        summary = (
            f"Retrieved {len(combined_evidence)} record(s) across [{', '.join(sources_used)}]. "
            f"Mode: {'Tavily-Only (Chronic Risk Mitigation)' if has_chronic else 'Dual PubMed + Tavily'}."
        )

        return ScannerResponse(
            input_text_processed=input_text,
            acute_symptoms=acute_symptoms,
            chronic_conditions=chronic_conditions,
            inferred_condition_hint=inferred_condition,
            chronic_risk_measures=chronic_risks,
            total_found=len(combined_evidence),
            sources_queried=sources_used,
            fallback_used=any(item.get("publication_window") == "10_year" for item in combined_evidence),
            evidence=combined_evidence,
            summary=summary
        )

    def get_dict_for_agent4(self, response: ScannerResponse) -> Dict[str, Any]:
        """Converts ScannerResponse Pydantic model into a dictionary payload for Agent 4."""
        dump_fn = getattr(response, "model_dump", getattr(response, "dict", None))
        if dump_fn and callable(dump_fn):
            return dump_fn()
        return response if isinstance(response, dict) else {}