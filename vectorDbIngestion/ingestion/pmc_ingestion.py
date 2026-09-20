"""
Ingestion module for PubMed Central Open Access subset.
Uses Biopython's Entrez API to search and fetch full-text XML,
then parses the JATS XML to extract structured text.
"""

import time
import logging
import xml.etree.ElementTree as ET
from Bio import Entrez
import config

logger = logging.getLogger(__name__)

def ingest_pmc(queries: list[str], max_per_query: int, delay: float) -> list[dict]:
    """
    Search and fetch articles from PubMed Central.

    Args:
        queries: List of search query strings.
        max_per_query: Maximum number of articles to fetch per query.
        delay: Delay in seconds between API requests.

    Returns:
        List of standardized document dictionaries.
    """
    if config.NCBI_EMAIL:
        Entrez.email = config.NCBI_EMAIL
    if config.NCBI_API_KEY:
        Entrez.api_key = config.NCBI_API_KEY

    documents = []
    seen_ids = set()

    for query in queries:
        logger.info(f"Searching PMC for query: {query}")
        try:
            # Search PMC
            time.sleep(delay)
            search_handle = Entrez.esearch(
                db="pmc",
                term=f'{query} AND "open access"[filter]',
                retmax=max_per_query
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            pmc_ids = search_results.get("IdList", [])
            
            for pmc_id in pmc_ids:
                if pmc_id in seen_ids:
                    continue
                    
                seen_ids.add(pmc_id)
                logger.debug(f"Fetching PMC article {pmc_id}")
                
                time.sleep(delay)
                fetch_handle = Entrez.efetch(
                    db="pmc",
                    id=pmc_id,
                    rettype="full",
                    retmode="xml"
                )
                xml_data = fetch_handle.read()
                fetch_handle.close()
                
                # Parse XML
                root = ET.fromstring(xml_data)
                
                # Extract metadata
                title = ""
                abstract = ""
                pub_year = ""
                journal = ""
                keywords = []
                sections = []
                
                article_meta = root.find(".//article-meta")
                if article_meta is not None:
                    title_elem = article_meta.find(".//article-title")
                    if title_elem is not None and title_elem.text:
                        title = "".join(title_elem.itertext()).strip()
                        
                    abstract_elem = article_meta.find(".//abstract")
                    if abstract_elem is not None:
                        abstract = "".join(abstract_elem.itertext()).strip()
                        
                    pub_date = article_meta.find(".//pub-date")
                    if pub_date is not None:
                        year_elem = pub_date.find("year")
                        if year_elem is not None and year_elem.text:
                            pub_year = year_elem.text.strip()
                            
                    for kwd in article_meta.findall(".//kwd"):
                        if kwd.text:
                            keywords.append("".join(kwd.itertext()).strip())
                            
                journal_meta = root.find(".//journal-meta")
                if journal_meta is not None:
                    journal_elem = journal_meta.find(".//journal-title")
                    if journal_elem is not None and journal_elem.text:
                        journal = "".join(journal_elem.itertext()).strip()
                        
                # Extract body sections
                body = root.find(".//body")
                full_body_text = ""
                if body is not None:
                    for sec in body.findall(".//sec"):
                        sec_title = ""
                        title_elem = sec.find("title")
                        if title_elem is not None and title_elem.text:
                            sec_title = "".join(title_elem.itertext()).strip()
                            
                        sec_text = ""
                        for p in sec.findall(".//p"):
                            sec_text += "".join(p.itertext()).strip() + "\n"
                            
                        if sec_text:
                            sections.append({"section_title": sec_title, "text": sec_text.strip()})
                            full_body_text += sec_text + "\n"
                            
                doc = {
                    'title': title,
                    'abstract': abstract,
                    'publication_year': pub_year,
                    'journal': journal,
                    'keywords': keywords,
                    'sections': sections,
                    'full_body_text': full_body_text.strip(),
                    'source_id': f"PMC{pmc_id}",
                    'source_type': 'medical-paper'
                }
                documents.append(doc)
                
        except Exception as e:
            logger.warning(f"Error processing query '{query}': {e}")
            continue

    return documents
