"""
pubmed_client.py - NCBI Entrez Client for Agent 3 (Evidence-Based Scanner).
Integrates strict UI filters (Abstract, 5 Article Types), MeSH syntax execution,
and dynamic 1-Year -> 10-Year Publication Date fallback.
"""

import os
import itertools
from typing import List, Dict, Set, Any, Optional
from Bio import Entrez
from tenacity import retry, stop_after_attempt, wait_exponential

# Package import aligned with agent3_scanner/src structure
from src.abbrevs import sanitize_and_expand_entity

# NCBI Entrez Configuration
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "adarshreddy261@gmail.com")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

Entrez.email = NCBI_EMAIL
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY

# Filter 1: Text Availability Requirement
ABSTRACT_FILTER = "hasabstract"

# Filter 2: Selected Article Types (Adaptive Clinical Trial, Case Reports, 
# Evaluation Study, Evidence Synthesis, Validation Study)
ARTICLE_TYPES_FILTER = (
    '("adaptive clinical trial"[pt] OR "case reports"[pt] OR '
    '"evaluation study"[pt] OR "evidence synthesis"[pt] OR "validation study"[pt])'
)


def truncate_text(text: str, max_words: int = 250) -> str:
    """Truncates abstract text to protect downstream LLM context windows."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "... [truncated]"
    return text


def build_filter_clause(years_back: int = 1) -> str:
    """
    Builds the combined PubMed filter query clause:
    - Publication Date: 'last 1 year' (or 'last 10 years')
    - Text Availability: hasabstract
    - Article Types: Adaptive Clinical Trial, Case Reports, Evaluation Study, Evidence Synthesis, Validation Study
    - Language: English
    """
    date_filter = f'"last {years_back} year"[dp]' if years_back == 1 else f'"last {years_back} years"[dp]'
    return f"AND {date_filter} AND {ABSTRACT_FILTER} AND {ARTICLE_TYPES_FILTER} AND english[Language]"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=lambda retry_state: []
)
def execute_pubmed_query(
    query: str, 
    max_results: int, 
    sort_by_date: bool = True, 
    years_back: int = 1
) -> List[Dict[str, Any]]:
    """
    Executes a PubMed E-Utilities search query with automatic retry backoff,
    dynamic publication date windowing, and UI-aligned article type filters.
    """
    try:
        filter_clause = build_filter_clause(years_back=years_back)
        formatted_query = f"({query}) {filter_clause}"
        sort_param = "pub_date" if sort_by_date else "relevance"

        handle = Entrez.esearch(
            db="pubmed",
            term=formatted_query,
            retmax=max_results,
            sort=sort_param
        )
        record = Entrez.read(handle)
        handle.close()

        id_list = record.get("IdList", [])
        if not id_list:
            return []

        fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), retmode="xml")
        articles = Entrez.read(fetch_handle)
        fetch_handle.close()

        results = []
        for article in articles.get("PubmedArticle", []):
            medline = article.get("MedlineCitation", {})
            pmid = str(medline.get("PMID", ""))

            article_data = medline.get("Article", {})
            title = str(article_data.get("ArticleTitle", "No title available."))

            # Extract DOI if present
            doi = ""
            for id_node in article_data.get("ELocationID", []):
                if getattr(id_node, "attributes", {}).get("EIdType") == "doi":
                    doi = str(id_node)

            # Defensive Abstract Parsing
            abstract_data = article_data.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_data, list):
                raw_abstract = " ".join([str(part) for part in abstract_data if part])
            elif isinstance(abstract_data, str):
                raw_abstract = abstract_data
            else:
                raw_abstract = "No abstract available."

            if not raw_abstract.strip():
                raw_abstract = "No abstract available."

            clean_abstract = truncate_text(raw_abstract, max_words=250)

            results.append({
                "source": "PubMed",
                "id": pmid,
                "doi": doi,
                "title": title,
                "abstract": clean_abstract,
                "query_matched": formatted_query,
                "recency_sorted": sort_by_date,
                "publication_window": f"{years_back}_year"
            })
        return results

    except Exception as e:
        print(f"[PubMed Warning] Exception on query '{query}' (Window: {years_back}yr): {e}")
        return []


def fetch_pubmed_evidence_with_recency_and_fallback(
    entities: List[str],
    condition_hint: str = "",
    target_count: int = 3,
    strict_query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Robust Multi-Stage Retrieval Pipeline with Dynamic 1-Yr -> 10-Yr Fallback:
    1. Primary 1-Year Search (Recent first -> Relevance fallback).
    2. Fallback to 10-Year Search if 1-year window yields < target_count (or 0).
    3. Pairwise 2-entity combinations under 10-Year Window if still insufficient.
    4. Global PMID deduplication across all stages.
    """
    collected_evidence: List[Dict[str, Any]] = []
    seen_pmids: Set[str] = set()

    # Sanitize and expand entities using abbrevs
    cleaned_entities = [sanitize_and_expand_entity(e) for e in entities if e]
    cleaned_entities = [e for e in cleaned_entities if e]

    hint_suffix = ""
    if condition_hint:
        expanded_hint = sanitize_and_expand_entity(condition_hint)
        if expanded_hint:
            hint_suffix = f" AND {expanded_hint}"

    primary_query = strict_query or (" AND ".join(cleaned_entities) + hint_suffix if cleaned_entities else condition_hint)

    if not primary_query:
        return collected_evidence

    # =========================================================================
    # STAGE 1: 1-Year Publication Window (Recent -> Relevance)
    # =========================================================================
    print("[PubMed] Searching Stage 1 (1-Year Window | Recent)...")
    res_1yr_recent = execute_pubmed_query(primary_query, max_results=target_count, sort_by_date=True, years_back=1)
    for item in res_1yr_recent:
        if item["id"] and item["id"] not in seen_pmids:
            seen_pmids.add(item["id"])
            collected_evidence.append(item)

    if len(collected_evidence) >= target_count:
        return collected_evidence[:target_count]

    needed = target_count - len(collected_evidence)
    res_1yr_rel = execute_pubmed_query(primary_query, max_results=needed + 2, sort_by_date=False, years_back=1)
    for item in res_1yr_rel:
        if item["id"] and item["id"] not in seen_pmids:
            seen_pmids.add(item["id"])
            collected_evidence.append(item)

    if len(collected_evidence) >= target_count:
        return collected_evidence[:target_count]

    # =========================================================================
    # STAGE 2: 10-Year Window Dynamic Fallback (If 1-year window has < target_count)
    # =========================================================================
    print(f"[PubMed Fallback] Insufficient reports in 1-year window ({len(collected_evidence)}/{target_count}). Widening to 10-Year Window...")

    needed = target_count - len(collected_evidence)
    res_10yr_recent = execute_pubmed_query(primary_query, max_results=needed + 2, sort_by_date=True, years_back=10)
    for item in res_10yr_recent:
        if item["id"] and item["id"] not in seen_pmids:
            seen_pmids.add(item["id"])
            collected_evidence.append(item)

    if len(collected_evidence) >= target_count:
        return collected_evidence[:target_count]

    needed = target_count - len(collected_evidence)
    res_10yr_rel = execute_pubmed_query(primary_query, max_results=needed + 2, sort_by_date=False, years_back=10)
    for item in res_10yr_rel:
        if item["id"] and item["id"] not in seen_pmids:
            seen_pmids.add(item["id"])
            collected_evidence.append(item)

    if len(collected_evidence) >= target_count:
        return collected_evidence[:target_count]

    # =========================================================================
    # STAGE 3: Pairwise 2-Entity Combinations (Under 10-Year Window)
    # =========================================================================
    if len(cleaned_entities) >= 3:
        entity_pairs = list(itertools.combinations(cleaned_entities, 2))

        for pair in entity_pairs:
            if len(collected_evidence) >= target_count:
                break

            pair_query = " AND ".join(pair) + hint_suffix
            needed = target_count - len(collected_evidence)

            pair_recent = execute_pubmed_query(pair_query, max_results=needed, sort_by_date=True, years_back=10)
            for item in pair_recent:
                if item["id"] and item["id"] not in seen_pmids:
                    seen_pmids.add(item["id"])
                    collected_evidence.append(item)

            if len(collected_evidence) < target_count:
                needed = target_count - len(collected_evidence)
                pair_rel = execute_pubmed_query(pair_query, max_results=needed, sort_by_date=False, years_back=10)
                for item in pair_rel:
                    if item["id"] and item["id"] not in seen_pmids:
                        seen_pmids.add(item["id"])
                        collected_evidence.append(item)

    return collected_evidence[:target_count]