"""
Ingestion module for medRxiv and bioRxiv preprint articles.
Uses Europe PMC API for keyword-based discovery of preprints,
then fetches metadata (title, abstract) from the bioRxiv/medRxiv REST API.
"""

import time
import logging
import requests
from typing import List, Dict, Any, Set

import config

logger = logging.getLogger(__name__)


def _search_europe_pmc_preprints(query: str, max_results: int, delay: float, seen_ids: Set[str]) -> List[Dict[str, Any]]:
    """
    Search Europe PMC for preprint articles matching a query.
    Europe PMC indexes both bioRxiv and medRxiv preprints.

    Args:
        query: Search query string.
        max_results: Maximum results to return for this query.
        delay: Delay between API requests in seconds.
        seen_ids: Set of DOIs already seen (for deduplication).

    Returns:
        List of preprint document dicts.
    """
    documents = []
    search_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    try:
        params = {
            "query": f'{query} AND (SRC:PPR)',
            "resultType": "core",
            "format": "json",
            "pageSize": min(max_results, 100),
        }
        time.sleep(delay)
        response = requests.get(search_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = data.get("resultList", {}).get("result", [])
        count = 0

        for item in results:
            if count >= max_results:
                break

            doi = item.get("doi", "")
            if not doi:
                continue
            if doi in seen_ids:
                continue

            seen_ids.add(doi)

            title = item.get("title", "")
            abstract = item.get("abstractText", "")
            pub_year = item.get("firstPublicationDate", "")[:4] if item.get("firstPublicationDate") else ""
            authors = item.get("authorString", "")
            source = item.get("bookOrReportDetails", {}).get("publisher", "") if item.get("bookOrReportDetails") else ""

            # Determine source server (bioRxiv vs medRxiv)
            server = "medRxiv" if "medrxiv" in doi.lower() or "medrxiv" in str(item.get("source", "")).lower() else "bioRxiv"

            # Build the document
            if not abstract and not title:
                continue

            full_text = ""
            sections = []
            if abstract:
                sections.append({"section_title": "Abstract", "text": abstract})
                full_text = abstract

            doc = {
                'title': title,
                'abstract': abstract,
                'publication_year': pub_year,
                'journal': f"{server} (Preprint)",
                'keywords': [],
                'sections': sections,
                'full_body_text': full_text,
                'source_id': f"preprint_{doi}",
                'source_type': 'preprint',
                'evidence_level': 'Preprint',
            }
            documents.append(doc)
            count += 1

    except Exception as e:
        logger.warning(f"Error searching Europe PMC preprints for '{query}': {e}")

    return documents


def ingest_preprints(queries: List[str], max_per_query: int, delay: float) -> List[Dict[str, Any]]:
    """
    Ingest preprint articles from medRxiv and bioRxiv.

    Uses Europe PMC as the search engine (since the native bioRxiv/medRxiv
    API does not support keyword search), then collects metadata.

    Args:
        queries: List of search query strings.
        max_per_query: Maximum number of preprints to fetch per query.
        delay: Delay between API requests in seconds.

    Returns:
        List of standardized document dictionaries with source_type='preprint'.
    """
    all_documents = []
    seen_ids: Set[str] = set()

    for query in queries:
        logger.info(f"Searching preprints for query: {query}")
        docs = _search_europe_pmc_preprints(
            query=query,
            max_results=max_per_query,
            delay=delay,
            seen_ids=seen_ids,
        )
        all_documents.extend(docs)
        logger.info(f"Found {len(docs)} preprints for query: {query}")

    logger.info(f"Total preprints ingested: {len(all_documents)}")
    return all_documents
